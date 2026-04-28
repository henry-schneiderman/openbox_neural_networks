"""
Fused Triton kernels for the shortwave Scattering module — T4 variant.

Uses mixed-precision dot products optimized for the T4 (Turing / SM75).
The T4 does not have TF32 tensor cores; with ``input_precision="tf32"``
Triton falls back to FP32 CUDA cores (~8 TFLOPS).

Strategy: the first MLP layer (which sees raw tau/mu inputs that can
exceed FP16's 65504 max) stays in FP32.  The hidden and output layers
(whose ReLU-bounded activations fit comfortably in FP16 range) cast to
FP16 to engage the T4's FP16 tensor cores (~65 TFLOPS).  The FP32
accumulator is preserved throughout.

In a single kernel launch (per direction: direct or diffuse) the kernel
performs, all in registers, the entire post-optical-depth scattering
pipeline:

  1. Build the per-row MLP input from tau (and mu_direct for the direct
     kernel).
  2. Run the 8-headed MLP (1 input + 2 hidden + output layers, all
     2D tl.dot in IEEE precision).
  3. Per-group softmax over the 24 outputs (8 groups of 3, encoded as
     8 groups of 4 with the 4th column biased to -inf so that
     exp(-inf) = 0 and the padding does not affect the softmax).
  4. Per-channel linear combination of the 8 basis vectors using the
     learned selection weights (a Conv2d in the original PyTorch code).
  5. Final softmax over the 3 components (e_t, e_r, e_a).
  6. Multiplication by (1 - t_full), where
     t_full = exp(-sum(tau)/mu).
  7. Storage of t_full and the three scaled e_* outputs.

The four output tensors are written directly so that the downstream
multireflection kernel can consume them as-is.

Author: Henry Schneiderman, henry@pittdata.com
"""

import torch
from torch import nn
import triton
import triton.language as tl
import math


# ---------------------------------------------------------------------------
# Direct kernel (input has mu_direct appended; mu_direct varies per row)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
        triton.Config({'BLOCK_B': 512}, num_warps=8),
    ],
    key=['B_total'],
)
@triton.jit
def _fused_scattering_direct_kernel(
    Tau_ptr, Mu_ptr,
    W0_ptr, B0_ptr, WH0_ptr, BH0_ptr, WH1_ptr, BH1_ptr, WO_ptr, BO_ptr,
    Wsel_ptr,
    Tfull_ptr, Et_ptr, Er_ptr, Ea_ptr,
    B_total,                        # = (n_samples * n_layers) * n_channels
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels      # sample-layer index
    c_idx = offs %  n_channels      # channel index

    c8  = tl.arange(0, 8)
    c16 = tl.arange(0, 16)
    c32 = tl.arange(0, 32)

    # ---- Load mu_direct (one per sample-layer) ----
    mu = tl.load(Mu_ptr + b_idx, mask=mask, other=1.0)
    inv_mu = 1.0 / (mu + 1.0e-7)

    # ---- Load tau[b, c, 0:8] padded into 16 columns ----
    # Layout: tau[b, c, k] is at flat offset (b*n_channels + c)*8 + k = offs*8 + k.
    tau_off = offs[:, None] * 8 + c16[None, :]
    x = tl.load(Tau_ptr + tau_off,
                mask=mask[:, None] & (c16[None, :] < 8), other=0.0)

    # Sum over the 8 constituents -> tau_sum
    tau_sum = tl.sum(x, axis=1)

    # Divide tau by mu (positions 8..15 are 0 already)
    x = x * inv_mu[:, None]
    # Place mu_direct at column 8 (the 9th input feature)
    x = tl.where(c16[None, :] == 8, mu[:, None], x)

    # ---- MLP forward: input layer (16 -> 32) ----
    w0 = tl.load(W0_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0_ptr + c32)
    h = tl.dot(x, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    # ---- Hidden layer 0 (32 -> 32) ----
    wh0 = tl.load(WH0_ptr + c32[:, None] * 32 + c32[None, :])
    bh0 = tl.load(BH0_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh0.to(tl.float16)) + bh0[None, :]
    h = tl.maximum(h, 0.0)

    # ---- Hidden layer 1 (32 -> 32) ----
    wh1 = tl.load(WH1_ptr + c32[:, None] * 32 + c32[None, :])
    bh1 = tl.load(BH1_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh1.to(tl.float16)) + bh1[None, :]
    h = tl.maximum(h, 0.0)

    # ---- Output layer (32 -> 32, last column of each group has -inf bias) ----
    wo = tl.load(WO_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BO_ptr + c32)
    out = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo.to(tl.float16)) + bo[None, :]

    # ---- Per-group softmax: 8 groups of 4 (4th col = padding -> 0) ----
    out_g = tl.reshape(out, [BLOCK_B * 8, 4])
    m = tl.max(out_g, axis=1)[:, None]
    out_g = tl.exp(out_g - m)
    s = tl.sum(out_g, axis=1)[:, None]
    out_g = out_g / s
    out_basis = tl.reshape(out_g, [BLOCK_B, 8, 4])  # last col is 0

    # ---- Linear combination over 8 basis vectors using per-channel weights ----
    wsel = tl.load(Wsel_ptr + c_idx[:, None] * 8 + c8[None, :],
                   mask=mask[:, None], other=0.0)
    # combined[b, p] = sum_k out_basis[b, k, p] * wsel[b, k]
    combined = tl.sum(out_basis * wsel[:, :, None], axis=1)  # [BLOCK_B, 4]

    # ---- Extract the 4 columns via tl.split ----
    # combined has shape [BLOCK_B, 4] with elements e_t (col 0),
    # e_r (col 1), e_a (col 2), dead-padding (col 3).
    #
    # tl.split splits along the LAST axis only, so after the reshape
    # [BLOCK_B, 4] -> [BLOCK_B, 2, 2] (row-major), the first split
    # along the inner-most axis yields:
    #     T0[b, i] = orig[b, 2*i + 0]   ->  cols {0, 2}  (e_t, e_a)
    #     T1[b, i] = orig[b, 2*i + 1]   ->  cols {1, 3}  (e_r, dead)
    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)   # each [BLOCK_B, 2]
    v_t, v_a = tl.split(T_even)             # cols 0 and 2  -> e_t, e_a
    v_r, _v_dead = tl.split(T_odd)          # cols 1 and 3  -> e_r, (dead)

    # ---- Final softmax over the 3 real components ----
    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    # ---- Scale by (1 - t_full_direct) ----
    t_full = tl.exp(-tau_sum * inv_mu)
    one_minus_t = 1.0 - t_full

    e_t = e_t_e * inv_s2 * one_minus_t
    e_r = e_r_e * inv_s2 * one_minus_t
    e_a = e_a_e * inv_s2 * one_minus_t

    # ---- Store outputs ----
    tl.store(Tfull_ptr + offs, t_full, mask=mask)
    tl.store(Et_ptr + offs, e_t, mask=mask)
    tl.store(Er_ptr + offs, e_r, mask=mask)
    tl.store(Ea_ptr + offs, e_a, mask=mask)


# ---------------------------------------------------------------------------
# Diffuse kernel (no mu in MLP input; mu_diffuse is a single scalar)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
        triton.Config({'BLOCK_B': 512}, num_warps=8),
    ],
    key=['B_total'],
)
@triton.jit
def _fused_scattering_diffuse_kernel(
    Tau_ptr,
    W0_ptr, B0_ptr, WH0_ptr, BH0_ptr, WH1_ptr, BH1_ptr, WO_ptr, BO_ptr,
    Wsel_ptr,
    Tfull_ptr, Et_ptr, Er_ptr, Ea_ptr,
    inv_mu_diffuse,                 # scalar (1 / (mu_diffuse + eps))
    B_total,                        # = (n_samples * n_layers) * n_channels
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    c_idx = offs %  n_channels      # channel index

    c8  = tl.arange(0, 8)
    c16 = tl.arange(0, 16)
    c32 = tl.arange(0, 32)

    # ---- Load tau[b, c, 0:8] padded into 16 columns ----
    tau_off = offs[:, None] * 8 + c16[None, :]
    x = tl.load(Tau_ptr + tau_off,
                mask=mask[:, None] & (c16[None, :] < 8), other=0.0)

    # Sum over the 8 constituents -> tau_sum
    tau_sum = tl.sum(x, axis=1)

    # ---- MLP forward: input layer (16 -> 32) ----
    w0 = tl.load(W0_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0_ptr + c32)
    h = tl.dot(x, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    # ---- Hidden layer 0 (32 -> 32) ----
    wh0 = tl.load(WH0_ptr + c32[:, None] * 32 + c32[None, :])
    bh0 = tl.load(BH0_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh0.to(tl.float16)) + bh0[None, :]
    h = tl.maximum(h, 0.0)

    # ---- Hidden layer 1 (32 -> 32) ----
    wh1 = tl.load(WH1_ptr + c32[:, None] * 32 + c32[None, :])
    bh1 = tl.load(BH1_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh1.to(tl.float16)) + bh1[None, :]
    h = tl.maximum(h, 0.0)

    # ---- Output layer ----
    wo = tl.load(WO_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BO_ptr + c32)
    out = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo.to(tl.float16)) + bo[None, :]

    # ---- Per-group softmax: 8 groups of 4 ----
    out_g = tl.reshape(out, [BLOCK_B * 8, 4])
    m = tl.max(out_g, axis=1)[:, None]
    out_g = tl.exp(out_g - m)
    s = tl.sum(out_g, axis=1)[:, None]
    out_g = out_g / s
    out_basis = tl.reshape(out_g, [BLOCK_B, 8, 4])

    # ---- Per-channel linear combination ----
    wsel = tl.load(Wsel_ptr + c_idx[:, None] * 8 + c8[None, :],
                   mask=mask[:, None], other=0.0)
    combined = tl.sum(out_basis * wsel[:, :, None], axis=1)

    # See the direct kernel for the explanation of this split pattern.
    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)   # each [BLOCK_B, 2]
    v_t, v_a = tl.split(T_even)             # cols 0 and 2  -> e_t, e_a
    v_r, _v_dead = tl.split(T_odd)          # cols 1 and 3  -> e_r, (dead)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    t_full = tl.exp(-tau_sum * inv_mu_diffuse)
    one_minus_t = 1.0 - t_full

    e_t = e_t_e * inv_s2 * one_minus_t
    e_r = e_r_e * inv_s2 * one_minus_t
    e_a = e_a_e * inv_s2 * one_minus_t

    tl.store(Tfull_ptr + offs, t_full, mask=mask)
    tl.store(Et_ptr + offs, e_t, mask=mask)
    tl.store(Er_ptr + offs, e_r, mask=mask)
    tl.store(Ea_ptr + offs, e_a, mask=mask)


# ---------------------------------------------------------------------------
# Combined direct + diffuse kernel.
#
# The per-row inputs of the direct and diffuse paths overlap completely
# in `tau` — both kernels above read the same 8-element tau row from
# global memory.  Fusing them into a single kernel:
#
#   * loads `tau` ONCE per row (~16 GB DRAM read saved per year),
#   * shares `tau_sum` between the two t_full computations,
#   * halves the per-launch overhead for the scattering stage,
#   * keeps the two MLPs sequential within the same program so the
#     compiler can pipeline weight loads and matmul issue.
#
# All eight outputs (4 direct + 4 diffuse) are written by the same
# kernel, in the same per-row layout the two split kernels used.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
    ],
    key=['B_total'],
)
@triton.jit
def _fused_scattering_both_kernel(
    Tau_ptr, Mu_ptr,
    # Direct MLP weights
    W0d_ptr, B0d_ptr, WH0d_ptr, BH0d_ptr, WH1d_ptr, BH1d_ptr, WOd_ptr, BOd_ptr,
    Wsel_dir_ptr,
    # Diffuse MLP weights
    W0f_ptr, B0f_ptr, WH0f_ptr, BH0f_ptr, WH1f_ptr, BH1f_ptr, WOf_ptr, BOf_ptr,
    Wsel_dif_ptr,
    # Direct outputs
    Tfull_dir_ptr, Et_dir_ptr, Er_dir_ptr, Ea_dir_ptr,
    # Diffuse outputs
    Tfull_dif_ptr, Et_dif_ptr, Er_dif_ptr, Ea_dif_ptr,
    inv_mu_diffuse,                 # scalar
    B_total,
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels
    c_idx = offs %  n_channels

    c8  = tl.arange(0, 8)
    c16 = tl.arange(0, 16)
    c32 = tl.arange(0, 32)

    # ---- Load mu_direct (one per sample-layer) ----
    mu = tl.load(Mu_ptr + b_idx, mask=mask, other=1.0)
    inv_mu = 1.0 / (mu + 1.0e-7)

    # ---- Load tau[b, c, 0:8] padded into 16 columns (ONCE) ----
    tau_off = offs[:, None] * 8 + c16[None, :]
    tau_pad = tl.load(Tau_ptr + tau_off,
                      mask=mask[:, None] & (c16[None, :] < 8), other=0.0)

    # Sum over the 8 constituents (used by both directions)
    tau_sum = tl.sum(tau_pad, axis=1)

    # Both transmissions share tau_sum and differ only in mu
    t_full_direct  = tl.exp(-tau_sum * inv_mu)
    t_full_diffuse = tl.exp(-tau_sum * inv_mu_diffuse)
    one_minus_t_dir = 1.0 - t_full_direct
    one_minus_t_dif = 1.0 - t_full_diffuse

    # =====================================================================
    # ---------------------- DIRECT branch ---------------------------------
    # =====================================================================
    # Build direct input: tau/mu in cols 0..7, mu in col 8
    x_d = tau_pad * inv_mu[:, None]
    x_d = tl.where(c16[None, :] == 8, mu[:, None], x_d)

    # ---- Direct MLP forward ----
    w0 = tl.load(W0d_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0d_ptr + c32)
    h = tl.dot(x_d, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    wh0 = tl.load(WH0d_ptr + c32[:, None] * 32 + c32[None, :])
    bh0 = tl.load(BH0d_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh0.to(tl.float16)) + bh0[None, :]
    h = tl.maximum(h, 0.0)

    wh1 = tl.load(WH1d_ptr + c32[:, None] * 32 + c32[None, :])
    bh1 = tl.load(BH1d_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh1.to(tl.float16)) + bh1[None, :]
    h = tl.maximum(h, 0.0)

    wo = tl.load(WOd_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BOd_ptr + c32)
    out = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo.to(tl.float16)) + bo[None, :]

    # ---- Per-group softmax (8 groups of 4) ----
    out_g = tl.reshape(out, [BLOCK_B * 8, 4])
    m = tl.max(out_g, axis=1)[:, None]
    out_g = tl.exp(out_g - m)
    s = tl.sum(out_g, axis=1)[:, None]
    out_g = out_g / s
    out_basis = tl.reshape(out_g, [BLOCK_B, 8, 4])

    # ---- Per-channel linear combination (direct selection weights) ----
    wsel = tl.load(Wsel_dir_ptr + c_idx[:, None] * 8 + c8[None, :],
                   mask=mask[:, None], other=0.0)
    combined = tl.sum(out_basis * wsel[:, :, None], axis=1)  # [BLOCK_B, 4]

    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    e_t_dir = e_t_e * inv_s2 * one_minus_t_dir
    e_r_dir = e_r_e * inv_s2 * one_minus_t_dir
    e_a_dir = e_a_e * inv_s2 * one_minus_t_dir

    tl.store(Tfull_dir_ptr + offs, t_full_direct, mask=mask)
    tl.store(Et_dir_ptr   + offs, e_t_dir, mask=mask)
    tl.store(Er_dir_ptr   + offs, e_r_dir, mask=mask)
    tl.store(Ea_dir_ptr   + offs, e_a_dir, mask=mask)

    # =====================================================================
    # ---------------------- DIFFUSE branch --------------------------------
    # =====================================================================
    # Diffuse input is just tau (no division by mu, no mu append).
    # tau_pad is still in registers (cols 0..7 hold the 8 constituents,
    # cols 8..15 are zero — exactly the padded MLP input format).

    # ---- Diffuse MLP forward ----
    w0 = tl.load(W0f_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0f_ptr + c32)
    h = tl.dot(tau_pad, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    wh0 = tl.load(WH0f_ptr + c32[:, None] * 32 + c32[None, :])
    bh0 = tl.load(BH0f_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh0.to(tl.float16)) + bh0[None, :]
    h = tl.maximum(h, 0.0)

    wh1 = tl.load(WH1f_ptr + c32[:, None] * 32 + c32[None, :])
    bh1 = tl.load(BH1f_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh1.to(tl.float16)) + bh1[None, :]
    h = tl.maximum(h, 0.0)

    wo = tl.load(WOf_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BOf_ptr + c32)
    out = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo.to(tl.float16)) + bo[None, :]

    # ---- Per-group softmax ----
    out_g = tl.reshape(out, [BLOCK_B * 8, 4])
    m = tl.max(out_g, axis=1)[:, None]
    out_g = tl.exp(out_g - m)
    s = tl.sum(out_g, axis=1)[:, None]
    out_g = out_g / s
    out_basis = tl.reshape(out_g, [BLOCK_B, 8, 4])

    # ---- Per-channel linear combination (diffuse selection weights) ----
    wsel = tl.load(Wsel_dif_ptr + c_idx[:, None] * 8 + c8[None, :],
                   mask=mask[:, None], other=0.0)
    combined = tl.sum(out_basis * wsel[:, :, None], axis=1)

    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    e_t_dif = e_t_e * inv_s2 * one_minus_t_dif
    e_r_dif = e_r_e * inv_s2 * one_minus_t_dif
    e_a_dif = e_a_e * inv_s2 * one_minus_t_dif

    tl.store(Tfull_dif_ptr + offs, t_full_diffuse, mask=mask)
    tl.store(Et_dif_ptr   + offs, e_t_dif, mask=mask)
    tl.store(Er_dif_ptr   + offs, e_r_dif, mask=mask)
    tl.store(Ea_dif_ptr   + offs, e_a_dif, mask=mask)


# ---------------------------------------------------------------------------
# Combined optical-depth + direct + diffuse scattering kernel.
#
# Identical to `_fused_scattering_both_kernel` except that the per-row
# tau vector is computed inline from three small inputs instead of
# being loaded from a materialized [B, n_channels, 8] tensor:
#
#     tau[b, c, k] = W_eff_filter[c, k] * mass[b, k] * ke_pad[b, k]
#
# where:
#   * `W_eff_filter[c, k]` is precomputed once at reconfigure time as
#         W_eff_filter[c, k] = exp(net_<gas_k>.weight[c, 0])
#                              * (filter_<gas_k>[c]   if k >= 2 else 1.0)
#     and stored as a [n_channels, 8] tensor.
#   * `mass[b, k]` is the constituent mass tensor passed in directly.
#   * `ke_pad[b, k]` is 1.0 for the two cloud constituents (k = 0, 1)
#     and `ke[b, k - 2]` for the six gas constituents (k = 2..7), where
#     `ke[b, 0..5]` is the per-row sigmoid output of the six small
#     temperature/pressure ke MLPs.  The ke tensor is computed once
#     per batch by a tiny torch.compile'd helper outside this kernel.
#
# This eliminates ~32 GB / year of DRAM traffic that previously came
# from materializing then immediately re-reading the tau tensor, and
# also folds the optical-depth stage into the same kernel.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
    ],
    key=['B_total'],
)
@triton.jit
def _fused_optical_scattering_kernel(
    Mass_ptr,            # [B, 8]              constituent_mass, contiguous
    Ke_ptr,              # [B, 6]              precomputed sigmoid(ke MLP)
    WeffFilter_ptr,      # [n_channels, 8]     precomputed exp(W) * filter
    Mu_ptr,              # [B]
    # Direct MLP weights
    W0d_ptr, B0d_ptr, WH0d_ptr, BH0d_ptr, WH1d_ptr, BH1d_ptr, WOd_ptr, BOd_ptr,
    Wsel_dir_ptr,
    # Diffuse MLP weights
    W0f_ptr, B0f_ptr, WH0f_ptr, BH0f_ptr, WH1f_ptr, BH1f_ptr, WOf_ptr, BOf_ptr,
    Wsel_dif_ptr,
    # Direct outputs
    Tfull_dir_ptr, Et_dir_ptr, Er_dir_ptr, Ea_dir_ptr,
    # Diffuse outputs
    Tfull_dif_ptr, Et_dif_ptr, Er_dif_ptr, Ea_dif_ptr,
    inv_mu_diffuse,                 # scalar
    B_total,
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels
    c_idx = offs %  n_channels

    c8  = tl.arange(0, 8)
    c16 = tl.arange(0, 16)
    c32 = tl.arange(0, 32)

    # ---- Load mu_direct (one per sample-layer) ----
    mu = tl.load(Mu_ptr + b_idx, mask=mask, other=1.0)
    inv_mu = 1.0 / (mu + 1.0e-7)

    # ---- Load mass[b, 0..7] padded into 16 columns ----
    mass_pad = tl.load(Mass_ptr + b_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)

    # ---- Load W_eff_filter[c, 0..7] padded into 16 columns ----
    W_ef_pad = tl.load(WeffFilter_ptr + c_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)

    # ---- Load ke shifted: 1.0 in cols 0,1 (clouds, no ke); ke[b, k-2]
    #      in cols 2..7; arbitrary in cols 8..15 (mass is 0 there). ----
    ke_offsets = tl.where(c16 >= 2, c16 - 2, 0)
    # Clamp out-of-range positions to 0 to avoid OOB pointer arithmetic
    # (the load itself is masked, but the address still gets computed).
    ke_offsets = tl.where(c16 < 8, ke_offsets, 0)
    ke_load_mask = mask[:, None] & (c16[None, :] >= 2) & (c16[None, :] < 8)
    ke_pad = tl.load(Ke_ptr + b_idx[:, None] * 6 + ke_offsets[None, :],
                     mask=ke_load_mask, other=1.0)

    # ---- Compute tau in registers (intermediates die immediately) ----
    tau_pad = W_ef_pad * mass_pad
    tau_pad = tau_pad * ke_pad           # [BLOCK_B, 16]; cols 8..15 = 0

    # Sum over the 8 constituents (cols 8..15 contribute 0)
    tau_sum = tl.sum(tau_pad, axis=1)

    # Both transmissions share tau_sum and differ only in mu
    t_full_direct  = tl.exp(-tau_sum * inv_mu)
    t_full_diffuse = tl.exp(-tau_sum * inv_mu_diffuse)
    one_minus_t_dir = 1.0 - t_full_direct
    one_minus_t_dif = 1.0 - t_full_diffuse

    # =====================================================================
    # ---------------------- DIRECT branch ---------------------------------
    # =====================================================================
    # Build direct input: tau/mu in cols 0..7, mu in col 8
    x_d = tau_pad * inv_mu[:, None]
    x_d = tl.where(c16[None, :] == 8, mu[:, None], x_d)

    # ---- Direct MLP forward ----
    w0 = tl.load(W0d_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0d_ptr + c32)
    h = tl.dot(x_d, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    wh0 = tl.load(WH0d_ptr + c32[:, None] * 32 + c32[None, :])
    bh0 = tl.load(BH0d_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh0.to(tl.float16)) + bh0[None, :]
    h = tl.maximum(h, 0.0)

    wh1 = tl.load(WH1d_ptr + c32[:, None] * 32 + c32[None, :])
    bh1 = tl.load(BH1d_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh1.to(tl.float16)) + bh1[None, :]
    h = tl.maximum(h, 0.0)

    wo = tl.load(WOd_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BOd_ptr + c32)
    out = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo.to(tl.float16)) + bo[None, :]

    # ---- Per-group softmax (8 groups of 4) ----
    out_g = tl.reshape(out, [BLOCK_B * 8, 4])
    m = tl.max(out_g, axis=1)[:, None]
    out_g = tl.exp(out_g - m)
    s = tl.sum(out_g, axis=1)[:, None]
    out_g = out_g / s
    out_basis = tl.reshape(out_g, [BLOCK_B, 8, 4])

    # ---- Per-channel linear combination (direct selection weights) ----
    wsel = tl.load(Wsel_dir_ptr + c_idx[:, None] * 8 + c8[None, :],
                   mask=mask[:, None], other=0.0)
    combined = tl.sum(out_basis * wsel[:, :, None], axis=1)

    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    e_t_dir = e_t_e * inv_s2 * one_minus_t_dir
    e_r_dir = e_r_e * inv_s2 * one_minus_t_dir
    e_a_dir = e_a_e * inv_s2 * one_minus_t_dir

    tl.store(Tfull_dir_ptr + offs, t_full_direct, mask=mask)
    tl.store(Et_dir_ptr   + offs, e_t_dir, mask=mask)
    tl.store(Er_dir_ptr   + offs, e_r_dir, mask=mask)
    tl.store(Ea_dir_ptr   + offs, e_a_dir, mask=mask)

    # =====================================================================
    # ---------------------- DIFFUSE branch --------------------------------
    # =====================================================================
    # Diffuse input is just tau (no division, no mu append).  tau_pad is
    # still in registers — cols 0..7 hold the constituent optical depths
    # and cols 8..15 are zero.

    w0 = tl.load(W0f_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0f_ptr + c32)
    h = tl.dot(tau_pad, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    wh0 = tl.load(WH0f_ptr + c32[:, None] * 32 + c32[None, :])
    bh0 = tl.load(BH0f_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh0.to(tl.float16)) + bh0[None, :]
    h = tl.maximum(h, 0.0)

    wh1 = tl.load(WH1f_ptr + c32[:, None] * 32 + c32[None, :])
    bh1 = tl.load(BH1f_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh1.to(tl.float16)) + bh1[None, :]
    h = tl.maximum(h, 0.0)

    wo = tl.load(WOf_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BOf_ptr + c32)
    out = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo.to(tl.float16)) + bo[None, :]

    # ---- Per-group softmax ----
    out_g = tl.reshape(out, [BLOCK_B * 8, 4])
    m = tl.max(out_g, axis=1)[:, None]
    out_g = tl.exp(out_g - m)
    s = tl.sum(out_g, axis=1)[:, None]
    out_g = out_g / s
    out_basis = tl.reshape(out_g, [BLOCK_B, 8, 4])

    # ---- Per-channel linear combination (diffuse selection weights) ----
    wsel = tl.load(Wsel_dif_ptr + c_idx[:, None] * 8 + c8[None, :],
                   mask=mask[:, None], other=0.0)
    combined = tl.sum(out_basis * wsel[:, :, None], axis=1)

    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    e_t_dif = e_t_e * inv_s2 * one_minus_t_dif
    e_r_dif = e_r_e * inv_s2 * one_minus_t_dif
    e_a_dif = e_a_e * inv_s2 * one_minus_t_dif

    tl.store(Tfull_dif_ptr + offs, t_full_diffuse, mask=mask)
    tl.store(Et_dif_ptr   + offs, e_t_dif, mask=mask)
    tl.store(Er_dif_ptr   + offs, e_r_dif, mask=mask)
    tl.store(Ea_dif_ptr   + offs, e_a_dif, mask=mask)


# ---------------------------------------------------------------------------
# Fused ke MLP kernel.
#
# The six ke MLPs (h2o, o3, co2, o2, n2o, ch4) are each:
#     input (2) -> hidden1 (6) -> hidden2 (4) -> hidden3 (4) -> output (1)
#
# In eager PyTorch each MLP is ~48 separate op launches (4 Linear + 3 ReLU +
# 1 sigmoid per MLP × 6 MLPs).  A single Triton kernel loads t_p[b, 2] once
# per row, runs all 6 MLPs in registers with weights loaded as constants,
# applies sigmoid, and writes the [B, 6] output.
#
# Constraint: all 6 MLPs fit in the 24 KB of per-program register file only
# if we don't keep intermediate activations for more than one MLP at a time.
# We therefore compute them sequentially, reusing register space.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=1),
        triton.Config({'BLOCK_B': 128}, num_warps=2),
        triton.Config({'BLOCK_B': 256}, num_warps=2),
        triton.Config({'BLOCK_B': 512}, num_warps=4),
    ],
    key=['B'],
)
@triton.jit
def _fused_ke_mlps_kernel(
    TP_ptr,      # [B, 2] contiguous (temperature, pressure)
    Ke_ptr,      # [B, 6] output
    # Per-MLP weights (constants loaded by the kernel).
    # The weights are passed as pointers to allow them to be registered
    # buffers on the Python side.  Triton will load them once at kernel
    # startup (they're small).
    W0h2o_ptr, B0h2o_ptr, W1h2o_ptr, B1h2o_ptr, W2h2o_ptr, B2h2o_ptr, W3h2o_ptr, B3h2o_ptr,
    W0o3_ptr,  B0o3_ptr,  W1o3_ptr,  B1o3_ptr,  W2o3_ptr,  B2o3_ptr,  W3o3_ptr,  B3o3_ptr,
    W0co2_ptr, B0co2_ptr, W1co2_ptr, B1co2_ptr, W2co2_ptr, B2co2_ptr, W3co2_ptr, B3co2_ptr,
    W0o2_ptr,  B0o2_ptr,  W1o2_ptr,  B1o2_ptr,  W2o2_ptr,  B2o2_ptr,  W3o2_ptr,  B3o2_ptr,
    W0n2o_ptr, B0n2o_ptr, W1n2o_ptr, B1n2o_ptr, W2n2o_ptr, B2n2o_ptr, W3n2o_ptr, B3n2o_ptr,
    W0ch4_ptr, B0ch4_ptr, W1ch4_ptr, B1ch4_ptr, W2ch4_ptr, B2ch4_ptr, W3ch4_ptr, B3ch4_ptr,
    B,
    BLOCK_B: tl.constexpr,
):
    """Compute sigmoid(ke_mlp(t_p)) for all 6 constituents in one kernel.

    All tensor dimensions use power-of-2 sizes to satisfy Triton's constraints:
    - arange(0, 2), arange(0, 4), arange(0, 8) are OK
    - arange(0, 6) is NOT OK; we use arange(0, 8) and mask
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B

    # Load [B, 2] temperature-pressure once, using arange(0, 2)
    c2 = tl.arange(0, 2)
    tp = tl.load(TP_ptr + offs[:, None] * 2 + c2[None, :],
                 mask=mask[:, None], other=0.0)  # [BLOCK_B, 2]

    # Precompute indices for the repeated hidden layer dimension (4)
    c4 = tl.arange(0, 4)
    c8 = tl.arange(0, 8)

    # Run each of the 6 MLPs in sequence, reusing register space.
    # Each MLP: 2 -> 6 -> 4 -> 4 -> 1
    # We use arange(0, 8) for the hidden layer and mask out columns 6..7
    for mlp_idx in range(6):
        if mlp_idx == 0:
            w0 = tl.load(W0h2o_ptr + c2[:, None] * 8 + c8[None, :])
            b0 = tl.load(B0h2o_ptr + c8)
            w1 = tl.load(W1h2o_ptr + c8[:, None] * 4 + c4[None, :])
            b1 = tl.load(B1h2o_ptr + c4)
            w2 = tl.load(W2h2o_ptr + c4[:, None] * 4 + c4[None, :])
            b2 = tl.load(B2h2o_ptr + c4)
            w3 = tl.load(W3h2o_ptr + c4[:, None] * 1 + tl.arange(0, 1)[None, :])
            b3 = tl.load(B3h2o_ptr + tl.arange(0, 1))
        elif mlp_idx == 1:
            w0 = tl.load(W0o3_ptr + c2[:, None] * 8 + c8[None, :])
            b0 = tl.load(B0o3_ptr + c8)
            w1 = tl.load(W1o3_ptr + c8[:, None] * 4 + c4[None, :])
            b1 = tl.load(B1o3_ptr + c4)
            w2 = tl.load(W2o3_ptr + c4[:, None] * 4 + c4[None, :])
            b2 = tl.load(B2o3_ptr + c4)
            w3 = tl.load(W3o3_ptr + c4[:, None] * 1 + tl.arange(0, 1)[None, :])
            b3 = tl.load(B3o3_ptr + tl.arange(0, 1))
        elif mlp_idx == 2:
            w0 = tl.load(W0co2_ptr + c2[:, None] * 8 + c8[None, :])
            b0 = tl.load(B0co2_ptr + c8)
            w1 = tl.load(W1co2_ptr + c8[:, None] * 4 + c4[None, :])
            b1 = tl.load(B1co2_ptr + c4)
            w2 = tl.load(W2co2_ptr + c4[:, None] * 4 + c4[None, :])
            b2 = tl.load(B2co2_ptr + c4)
            w3 = tl.load(W3co2_ptr + c4[:, None] * 1 + tl.arange(0, 1)[None, :])
            b3 = tl.load(B3co2_ptr + tl.arange(0, 1))
        elif mlp_idx == 3:
            w0 = tl.load(W0o2_ptr + c2[:, None] * 8 + c8[None, :])
            b0 = tl.load(B0o2_ptr + c8)
            w1 = tl.load(W1o2_ptr + c8[:, None] * 4 + c4[None, :])
            b1 = tl.load(B1o2_ptr + c4)
            w2 = tl.load(W2o2_ptr + c4[:, None] * 4 + c4[None, :])
            b2 = tl.load(B2o2_ptr + c4)
            w3 = tl.load(W3o2_ptr + c4[:, None] * 1 + tl.arange(0, 1)[None, :])
            b3 = tl.load(B3o2_ptr + tl.arange(0, 1))
        elif mlp_idx == 4:
            w0 = tl.load(W0n2o_ptr + c2[:, None] * 8 + c8[None, :])
            b0 = tl.load(B0n2o_ptr + c8)
            w1 = tl.load(W1n2o_ptr + c8[:, None] * 4 + c4[None, :])
            b1 = tl.load(B1n2o_ptr + c4)
            w2 = tl.load(W2n2o_ptr + c4[:, None] * 4 + c4[None, :])
            b2 = tl.load(B2n2o_ptr + c4)
            w3 = tl.load(W3n2o_ptr + c4[:, None] * 1 + tl.arange(0, 1)[None, :])
            b3 = tl.load(B3n2o_ptr + tl.arange(0, 1))
        else:  # mlp_idx == 5
            w0 = tl.load(W0ch4_ptr + c2[:, None] * 8 + c8[None, :])
            b0 = tl.load(B0ch4_ptr + c8)
            w1 = tl.load(W1ch4_ptr + c8[:, None] * 4 + c4[None, :])
            b1 = tl.load(B1ch4_ptr + c4)
            w2 = tl.load(W2ch4_ptr + c4[:, None] * 4 + c4[None, :])
            b2 = tl.load(B2ch4_ptr + c4)
            w3 = tl.load(W3ch4_ptr + c4[:, None] * 1 + tl.arange(0, 1)[None, :])
            b3 = tl.load(B3ch4_ptr + tl.arange(0, 1))

        # Forward pass for this MLP:  2 -> 6 -> 4 -> 4 -> 1 + sigmoid
        # w0 is [2, 8] but we only use columns 0..5 (the 6 outputs);
        # columns 6..7 are padding and their contributions are zeroed out below.
        h = tl.dot(tp, w0) + b0[None, :]  # [BLOCK_B, 8]
        h = h[:, :6]  # Trim to actual 6 outputs
        h = tl.maximum(h, 0.0)
        h = tl.dot(h, w1) + b1[None, :]
        h = tl.maximum(h, 0.0)
        h = tl.dot(h, w2) + b2[None, :]
        h = tl.maximum(h, 0.0)
        h = tl.dot(h, w3) + b3[None, :]
        # Sigmoid: 1 / (1 + exp(-x))
        h = 1.0 / (1.0 + tl.exp(-h))
        h = tl.reshape(h, (BLOCK_B,))

        # Store in output column mlp_idx
        tl.store(Ke_ptr + offs * 6 + mlp_idx, h, mask=mask)


# ---------------------------------------------------------------------------
# Inline ke MLP helper (2→6→4→4→1→sigmoid) for fused ke+scattering.
#
# Packed weight layout per MLP (71 FP32 scalars):
#   [0..11]  W0.T [2, 6]  row-major   (12 elements)
#   [12..17] b0   [6]                  (6 elements)
#   [18..41] W1.T [6, 4]  row-major   (24 elements)
#   [42..45] b1   [4]                  (4 elements)
#   [46..61] W2.T [4, 4]  row-major   (16 elements)
#   [62..65] b2   [4]                  (4 elements)
#   [66..69] W3.T [4, 1]  row-major   (4 elements)
#   [70]     b3   [1]                  (1 element)
# Six MLPs packed sequentially: total 426 scalars (~1.7 KB).
# ---------------------------------------------------------------------------
@triton.jit
def _ke_mlp_inline(temp, pres, KW_ptr, base):
    """Compute one ke MLP: sigmoid(MLP(temp, pres)).

    temp, pres: [BLOCK_B] vectors
    KW_ptr: packed weights pointer
    base: offset for this MLP (0, 71, 142, 213, 284, 355)
    Returns: [BLOCK_B] sigmoid output
    """
    # Layer 0: [2] → [6], W0.T at base+0 [2,6] row-major, b0 at base+12
    w = base
    b = base + 12
    h0 = temp * tl.load(KW_ptr + w + 0) + pres * tl.load(KW_ptr + w + 6)  + tl.load(KW_ptr + b + 0)
    h1 = temp * tl.load(KW_ptr + w + 1) + pres * tl.load(KW_ptr + w + 7)  + tl.load(KW_ptr + b + 1)
    h2 = temp * tl.load(KW_ptr + w + 2) + pres * tl.load(KW_ptr + w + 8)  + tl.load(KW_ptr + b + 2)
    h3 = temp * tl.load(KW_ptr + w + 3) + pres * tl.load(KW_ptr + w + 9)  + tl.load(KW_ptr + b + 3)
    h4 = temp * tl.load(KW_ptr + w + 4) + pres * tl.load(KW_ptr + w + 10) + tl.load(KW_ptr + b + 4)
    h5 = temp * tl.load(KW_ptr + w + 5) + pres * tl.load(KW_ptr + w + 11) + tl.load(KW_ptr + b + 5)
    h0 = tl.maximum(h0, 0.0); h1 = tl.maximum(h1, 0.0); h2 = tl.maximum(h2, 0.0)
    h3 = tl.maximum(h3, 0.0); h4 = tl.maximum(h4, 0.0); h5 = tl.maximum(h5, 0.0)

    # Layer 1: [6] → [4], W1.T at base+18 [6,4] row-major, b1 at base+42
    w = base + 18
    b = base + 42
    g0 = (h0 * tl.load(KW_ptr + w + 0)  + h1 * tl.load(KW_ptr + w + 4)  +
          h2 * tl.load(KW_ptr + w + 8)  + h3 * tl.load(KW_ptr + w + 12) +
          h4 * tl.load(KW_ptr + w + 16) + h5 * tl.load(KW_ptr + w + 20) + tl.load(KW_ptr + b + 0))
    g1 = (h0 * tl.load(KW_ptr + w + 1)  + h1 * tl.load(KW_ptr + w + 5)  +
          h2 * tl.load(KW_ptr + w + 9)  + h3 * tl.load(KW_ptr + w + 13) +
          h4 * tl.load(KW_ptr + w + 17) + h5 * tl.load(KW_ptr + w + 21) + tl.load(KW_ptr + b + 1))
    g2 = (h0 * tl.load(KW_ptr + w + 2)  + h1 * tl.load(KW_ptr + w + 6)  +
          h2 * tl.load(KW_ptr + w + 10) + h3 * tl.load(KW_ptr + w + 14) +
          h4 * tl.load(KW_ptr + w + 18) + h5 * tl.load(KW_ptr + w + 22) + tl.load(KW_ptr + b + 2))
    g3 = (h0 * tl.load(KW_ptr + w + 3)  + h1 * tl.load(KW_ptr + w + 7)  +
          h2 * tl.load(KW_ptr + w + 11) + h3 * tl.load(KW_ptr + w + 15) +
          h4 * tl.load(KW_ptr + w + 19) + h5 * tl.load(KW_ptr + w + 23) + tl.load(KW_ptr + b + 3))
    g0 = tl.maximum(g0, 0.0); g1 = tl.maximum(g1, 0.0)
    g2 = tl.maximum(g2, 0.0); g3 = tl.maximum(g3, 0.0)

    # Layer 2: [4] → [4], W2.T at base+46 [4,4] row-major, b2 at base+62
    w = base + 46
    b = base + 62
    f0 = (g0 * tl.load(KW_ptr + w + 0)  + g1 * tl.load(KW_ptr + w + 4)  +
          g2 * tl.load(KW_ptr + w + 8)  + g3 * tl.load(KW_ptr + w + 12) + tl.load(KW_ptr + b + 0))
    f1 = (g0 * tl.load(KW_ptr + w + 1)  + g1 * tl.load(KW_ptr + w + 5)  +
          g2 * tl.load(KW_ptr + w + 9)  + g3 * tl.load(KW_ptr + w + 13) + tl.load(KW_ptr + b + 1))
    f2 = (g0 * tl.load(KW_ptr + w + 2)  + g1 * tl.load(KW_ptr + w + 6)  +
          g2 * tl.load(KW_ptr + w + 10) + g3 * tl.load(KW_ptr + w + 14) + tl.load(KW_ptr + b + 2))
    f3 = (g0 * tl.load(KW_ptr + w + 3)  + g1 * tl.load(KW_ptr + w + 7)  +
          g2 * tl.load(KW_ptr + w + 11) + g3 * tl.load(KW_ptr + w + 15) + tl.load(KW_ptr + b + 3))
    f0 = tl.maximum(f0, 0.0); f1 = tl.maximum(f1, 0.0)
    f2 = tl.maximum(f2, 0.0); f3 = tl.maximum(f3, 0.0)

    # Layer 3: [4] → [1], W3.T at base+66 [4,1], b3 at base+70
    w = base + 66
    out = (f0 * tl.load(KW_ptr + w + 0) + f1 * tl.load(KW_ptr + w + 1) +
           f2 * tl.load(KW_ptr + w + 2) + f3 * tl.load(KW_ptr + w + 3) +
           tl.load(KW_ptr + base + 70))

    # Sigmoid
    return 1.0 / (1.0 + tl.exp(-out))


# ---------------------------------------------------------------------------
# T4-optimized split kernels with FUSED ke computation.
#
# Identical to _optical_split_{direct,diffuse}_T4_kernel except the ke
# values are computed inline from temperature/pressure instead of loaded
# from a pre-computed [B, 6] tensor.  This eliminates the separate ke
# kernel launch and the ke DRAM round-trip.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 128}, num_warps=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
        # maxnreg-capped variants
        triton.Config({'BLOCK_B': 128}, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_B': 256}, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_B': 256}, num_warps=8, maxnreg=128),
        # num_stages=2 variants
        triton.Config({'BLOCK_B': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_B': 256}, num_warps=8, num_stages=2),
    ],
    key=['B_total'],
)
@triton.jit
def _optical_split_direct_T4_fke_kernel(
    Mass_ptr, TP_ptr, WeffFilter_ptr, Mu_ptr,
    KeW_ptr,                       # packed ke MLP weights [426]
    W0_ptr, B0_ptr,
    WH0_f16_ptr, BH0_ptr, WH1_f16_ptr, BH1_ptr, WO_f16_ptr, BO_ptr,
    Wsel_ptr,
    Tfull_ptr, Et_ptr, Er_ptr, Ea_ptr,
    B_total,
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels
    c_idx = offs %  n_channels

    c8  = tl.arange(0, 8)
    c16 = tl.arange(0, 16)
    c32 = tl.arange(0, 32)

    mu = tl.load(Mu_ptr + b_idx, mask=mask, other=1.0)
    inv_mu = 1.0 / (mu + 1.0e-7)

    mass_pad = tl.load(Mass_ptr + b_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    W_ef_pad = tl.load(WeffFilter_ptr + c_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)

    # ---- Inline ke computation from temperature/pressure ----
    temp = tl.load(TP_ptr + b_idx * 2,     mask=mask, other=0.0)
    pres = tl.load(TP_ptr + b_idx * 2 + 1, mask=mask, other=0.0)
    ke0 = _ke_mlp_inline(temp, pres, KeW_ptr, 0 * 71)    # h2o
    ke1 = _ke_mlp_inline(temp, pres, KeW_ptr, 1 * 71)    # o3
    ke2 = _ke_mlp_inline(temp, pres, KeW_ptr, 2 * 71)    # co2
    ke3 = _ke_mlp_inline(temp, pres, KeW_ptr, 3 * 71)    # o2
    ke4 = _ke_mlp_inline(temp, pres, KeW_ptr, 4 * 71)    # n2o
    ke5 = _ke_mlp_inline(temp, pres, KeW_ptr, 5 * 71)    # ch4
    # Construct ke_pad [BLOCK_B, 16]: columns 2-7 = ke values, rest = 1.0
    ke_pad = tl.where(c16[None, :] == 2, ke0[:, None], 1.0)
    ke_pad = tl.where(c16[None, :] == 3, ke1[:, None], ke_pad)
    ke_pad = tl.where(c16[None, :] == 4, ke2[:, None], ke_pad)
    ke_pad = tl.where(c16[None, :] == 5, ke3[:, None], ke_pad)
    ke_pad = tl.where(c16[None, :] == 6, ke4[:, None], ke_pad)
    ke_pad = tl.where(c16[None, :] == 7, ke5[:, None], ke_pad)

    tau_pad = W_ef_pad * mass_pad * ke_pad
    tau_sum = tl.sum(tau_pad, axis=1)

    t_full = tl.exp(-tau_sum * inv_mu)
    one_minus_t = 1.0 - t_full

    x_d = tau_pad * inv_mu[:, None]
    x_d = tl.where(c16[None, :] == 8, mu[:, None], x_d)

    # Layer 0 (FP32)
    w0 = tl.load(W0_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0_ptr + c32)
    h = tl.dot(x_d, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    # Hidden 0 (FP16 TC, native FP16 weights)
    wh0 = tl.load(WH0_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bh0 = tl.load(BH0_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh0, out_dtype=tl.float32) + bh0[None, :]
    h = tl.maximum(h, 0.0)

    # Hidden 1 (FP16 TC)
    wh1 = tl.load(WH1_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bh1 = tl.load(BH1_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh1, out_dtype=tl.float32) + bh1[None, :]
    h = tl.maximum(h, 0.0)

    # Output (FP16 TC)
    wo = tl.load(WO_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BO_ptr + c32)
    out = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo, out_dtype=tl.float32) + bo[None, :]

    out_g = tl.reshape(out, [BLOCK_B * 8, 4])
    m = tl.max(out_g, axis=1)[:, None]
    out_g = tl.exp(out_g - m)
    s = tl.sum(out_g, axis=1)[:, None]
    out_g = out_g / s
    out_basis = tl.reshape(out_g, [BLOCK_B, 8, 4])

    wsel = tl.load(Wsel_ptr + c_idx[:, None] * 8 + c8[None, :],
                   mask=mask[:, None], other=0.0)
    combined = tl.sum(out_basis * wsel[:, :, None], axis=1)

    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    tl.store(Tfull_ptr + offs, t_full, mask=mask)
    tl.store(Et_ptr + offs, e_t_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Er_ptr + offs, e_r_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Ea_ptr + offs, e_a_e * inv_s2 * one_minus_t, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 128}, num_warps=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
        triton.Config({'BLOCK_B': 128}, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_B': 256}, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_B': 256}, num_warps=8, maxnreg=128),
        triton.Config({'BLOCK_B': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_B': 256}, num_warps=8, num_stages=2),
    ],
    key=['B_total'],
)
@triton.jit
def _optical_split_diffuse_T4_fke_kernel(
    Mass_ptr, TP_ptr, WeffFilter_ptr,
    KeW_ptr,
    W0_ptr, B0_ptr,
    WH0_f16_ptr, BH0_ptr, WH1_f16_ptr, BH1_ptr, WO_f16_ptr, BO_ptr,
    Wsel_ptr,
    Tfull_ptr, Et_ptr, Er_ptr, Ea_ptr,
    inv_mu_diffuse,
    B_total,
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels
    c_idx = offs %  n_channels

    c8  = tl.arange(0, 8)
    c16 = tl.arange(0, 16)
    c32 = tl.arange(0, 32)

    mass_pad = tl.load(Mass_ptr + b_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    W_ef_pad = tl.load(WeffFilter_ptr + c_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)

    # ---- Inline ke computation ----
    temp = tl.load(TP_ptr + b_idx * 2,     mask=mask, other=0.0)
    pres = tl.load(TP_ptr + b_idx * 2 + 1, mask=mask, other=0.0)
    ke0 = _ke_mlp_inline(temp, pres, KeW_ptr, 0 * 71)
    ke1 = _ke_mlp_inline(temp, pres, KeW_ptr, 1 * 71)
    ke2 = _ke_mlp_inline(temp, pres, KeW_ptr, 2 * 71)
    ke3 = _ke_mlp_inline(temp, pres, KeW_ptr, 3 * 71)
    ke4 = _ke_mlp_inline(temp, pres, KeW_ptr, 4 * 71)
    ke5 = _ke_mlp_inline(temp, pres, KeW_ptr, 5 * 71)
    ke_pad = tl.where(c16[None, :] == 2, ke0[:, None], 1.0)
    ke_pad = tl.where(c16[None, :] == 3, ke1[:, None], ke_pad)
    ke_pad = tl.where(c16[None, :] == 4, ke2[:, None], ke_pad)
    ke_pad = tl.where(c16[None, :] == 5, ke3[:, None], ke_pad)
    ke_pad = tl.where(c16[None, :] == 6, ke4[:, None], ke_pad)
    ke_pad = tl.where(c16[None, :] == 7, ke5[:, None], ke_pad)

    tau_pad = W_ef_pad * mass_pad * ke_pad
    tau_sum = tl.sum(tau_pad, axis=1)

    t_full = tl.exp(-tau_sum * inv_mu_diffuse)
    one_minus_t = 1.0 - t_full

    # Layer 0 (FP32)
    w0 = tl.load(W0_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0_ptr + c32)
    h = tl.dot(tau_pad, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    # Hidden 0 (FP16 TC, native FP16 weights)
    wh0 = tl.load(WH0_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bh0 = tl.load(BH0_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh0, out_dtype=tl.float32) + bh0[None, :]
    h = tl.maximum(h, 0.0)

    # Hidden 1 (FP16 TC)
    wh1 = tl.load(WH1_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bh1 = tl.load(BH1_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh1, out_dtype=tl.float32) + bh1[None, :]
    h = tl.maximum(h, 0.0)

    # Output (FP16 TC)
    wo = tl.load(WO_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BO_ptr + c32)
    out = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo, out_dtype=tl.float32) + bo[None, :]

    out_g = tl.reshape(out, [BLOCK_B * 8, 4])
    m = tl.max(out_g, axis=1)[:, None]
    out_g = tl.exp(out_g - m)
    s = tl.sum(out_g, axis=1)[:, None]
    out_g = out_g / s
    out_basis = tl.reshape(out_g, [BLOCK_B, 8, 4])

    wsel = tl.load(Wsel_ptr + c_idx[:, None] * 8 + c8[None, :],
                   mask=mask[:, None], other=0.0)
    combined = tl.sum(out_basis * wsel[:, :, None], axis=1)

    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    tl.store(Tfull_ptr + offs, t_full, mask=mask)
    tl.store(Et_ptr + offs, e_t_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Er_ptr + offs, e_r_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Ea_ptr + offs, e_a_e * inv_s2 * one_minus_t, mask=mask)


# ---------------------------------------------------------------------------
# T4-optimized split kernels: separate direct and diffuse kernels so each
# kernel only runs ONE MLP, halving register pressure vs the combined kernel.
# Uses FP32 for all dot products (Triton does not engage SM75 tensor cores
# via in-kernel .to(tl.float16) casts).
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
        triton.Config({'BLOCK_B': 512}, num_warps=8),
    ],
    key=['B_total'],
)
@triton.jit
def _optical_scattering_direct_T4_kernel(
    Mass_ptr, Ke_ptr, WeffFilter_ptr, Mu_ptr,
    W0_ptr, B0_ptr, WH0_ptr, BH0_ptr, WH1_ptr, BH1_ptr, WO_ptr, BO_ptr,
    Wsel_ptr,
    Tfull_ptr, Et_ptr, Er_ptr, Ea_ptr,
    B_total,
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels
    c_idx = offs %  n_channels

    c8  = tl.arange(0, 8)
    c16 = tl.arange(0, 16)
    c32 = tl.arange(0, 32)

    # ---- Load mu_direct ----
    mu = tl.load(Mu_ptr + b_idx, mask=mask, other=1.0)
    inv_mu = 1.0 / (mu + 1.0e-7)

    # ---- Compute tau inline from mass, ke, W_eff_filter ----
    mass_pad = tl.load(Mass_ptr + b_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    W_ef_pad = tl.load(WeffFilter_ptr + c_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    ke_offsets = tl.where(c16 >= 2, c16 - 2, 0)
    ke_offsets = tl.where(c16 < 8, ke_offsets, 0)
    ke_load_mask = mask[:, None] & (c16[None, :] >= 2) & (c16[None, :] < 8)
    ke_pad = tl.load(Ke_ptr + b_idx[:, None] * 6 + ke_offsets[None, :],
                     mask=ke_load_mask, other=1.0)

    tau_pad = W_ef_pad * mass_pad * ke_pad
    tau_sum = tl.sum(tau_pad, axis=1)

    t_full = tl.exp(-tau_sum * inv_mu)
    one_minus_t = 1.0 - t_full

    # ---- Build direct MLP input: tau/mu in cols 0..7, mu in col 8 ----
    x_d = tau_pad * inv_mu[:, None]
    x_d = tl.where(c16[None, :] == 8, mu[:, None], x_d)

    # ---- Direct MLP forward (layer 1 FP32, layers 2-4 FP16 tensor cores) ----
    w0 = tl.load(W0_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0_ptr + c32)
    h = tl.dot(x_d, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    wh0 = tl.load(WH0_ptr + c32[:, None] * 32 + c32[None, :])
    bh0 = tl.load(BH0_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh0.to(tl.float16), out_dtype=tl.float32) + bh0[None, :]
    h = tl.maximum(h, 0.0)

    wh1 = tl.load(WH1_ptr + c32[:, None] * 32 + c32[None, :])
    bh1 = tl.load(BH1_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh1.to(tl.float16), out_dtype=tl.float32) + bh1[None, :]
    h = tl.maximum(h, 0.0)

    wo = tl.load(WO_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BO_ptr + c32)
    out = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo.to(tl.float16), out_dtype=tl.float32) + bo[None, :]

    # ---- Per-group softmax (8 groups of 4) ----
    out_g = tl.reshape(out, [BLOCK_B * 8, 4])
    m = tl.max(out_g, axis=1)[:, None]
    out_g = tl.exp(out_g - m)
    s = tl.sum(out_g, axis=1)[:, None]
    out_g = out_g / s
    out_basis = tl.reshape(out_g, [BLOCK_B, 8, 4])

    # ---- Per-channel linear combination ----
    wsel = tl.load(Wsel_ptr + c_idx[:, None] * 8 + c8[None, :],
                   mask=mask[:, None], other=0.0)
    combined = tl.sum(out_basis * wsel[:, :, None], axis=1)

    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    e_t = e_t_e * inv_s2 * one_minus_t
    e_r = e_r_e * inv_s2 * one_minus_t
    e_a = e_a_e * inv_s2 * one_minus_t

    tl.store(Tfull_ptr + offs, t_full, mask=mask)
    tl.store(Et_ptr + offs, e_t, mask=mask)
    tl.store(Er_ptr + offs, e_r, mask=mask)
    tl.store(Ea_ptr + offs, e_a, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
        triton.Config({'BLOCK_B': 512}, num_warps=8),
    ],
    key=['B_total'],
)
@triton.jit
def _optical_scattering_diffuse_T4_kernel(
    Mass_ptr, Ke_ptr, WeffFilter_ptr,
    W0_ptr, B0_ptr, WH0_ptr, BH0_ptr, WH1_ptr, BH1_ptr, WO_ptr, BO_ptr,
    Wsel_ptr,
    Tfull_ptr, Et_ptr, Er_ptr, Ea_ptr,
    inv_mu_diffuse,
    B_total,
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels
    c_idx = offs %  n_channels

    c8  = tl.arange(0, 8)
    c16 = tl.arange(0, 16)
    c32 = tl.arange(0, 32)

    # ---- Compute tau inline from mass, ke, W_eff_filter ----
    mass_pad = tl.load(Mass_ptr + b_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    W_ef_pad = tl.load(WeffFilter_ptr + c_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    ke_offsets = tl.where(c16 >= 2, c16 - 2, 0)
    ke_offsets = tl.where(c16 < 8, ke_offsets, 0)
    ke_load_mask = mask[:, None] & (c16[None, :] >= 2) & (c16[None, :] < 8)
    ke_pad = tl.load(Ke_ptr + b_idx[:, None] * 6 + ke_offsets[None, :],
                     mask=ke_load_mask, other=1.0)

    tau_pad = W_ef_pad * mass_pad * ke_pad
    tau_sum = tl.sum(tau_pad, axis=1)

    t_full = tl.exp(-tau_sum * inv_mu_diffuse)
    one_minus_t = 1.0 - t_full

    # ---- Diffuse MLP forward (layer 1 FP32, layers 2-4 FP16 tensor cores) ----
    w0 = tl.load(W0_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0_ptr + c32)
    h = tl.dot(tau_pad, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    wh0 = tl.load(WH0_ptr + c32[:, None] * 32 + c32[None, :])
    bh0 = tl.load(BH0_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh0.to(tl.float16), out_dtype=tl.float32) + bh0[None, :]
    h = tl.maximum(h, 0.0)

    wh1 = tl.load(WH1_ptr + c32[:, None] * 32 + c32[None, :])
    bh1 = tl.load(BH1_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh1.to(tl.float16), out_dtype=tl.float32) + bh1[None, :]
    h = tl.maximum(h, 0.0)

    wo = tl.load(WO_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BO_ptr + c32)
    out = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo.to(tl.float16), out_dtype=tl.float32) + bo[None, :]

    # ---- Per-group softmax (8 groups of 4) ----
    out_g = tl.reshape(out, [BLOCK_B * 8, 4])
    m = tl.max(out_g, axis=1)[:, None]
    out_g = tl.exp(out_g - m)
    s = tl.sum(out_g, axis=1)[:, None]
    out_g = out_g / s
    out_basis = tl.reshape(out_g, [BLOCK_B, 8, 4])

    # ---- Per-channel linear combination ----
    wsel = tl.load(Wsel_ptr + c_idx[:, None] * 8 + c8[None, :],
                   mask=mask[:, None], other=0.0)
    combined = tl.sum(out_basis * wsel[:, :, None], axis=1)

    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    e_t = e_t_e * inv_s2 * one_minus_t
    e_r = e_r_e * inv_s2 * one_minus_t
    e_a = e_a_e * inv_s2 * one_minus_t

    tl.store(Tfull_ptr + offs, t_full, mask=mask)
    tl.store(Et_ptr + offs, e_t, mask=mask)
    tl.store(Er_ptr + offs, e_r, mask=mask)
    tl.store(Ea_ptr + offs, e_a, mask=mask)


# ---------------------------------------------------------------------------
# T4-optimized split kernels with pre-stored FP16 weights.
# Hidden/output weights loaded as native FP16 (no in-kernel conversion).
# maxnreg=128 caps registers per thread: at 128 regs with BLOCK_B=128
# warps=4, we get 4 blocks/SM → 50% occupancy (vs 37.5% at 149 regs).
# Test 1: include maxnreg-capped variants so autotune can pick whichever
# wins empirically; falls back to uncapped if capping causes spills.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 128}, num_warps=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
        # maxnreg-capped variants — target 50% occupancy on T4
        triton.Config({'BLOCK_B': 128}, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_B': 256}, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_B': 256}, num_warps=8, maxnreg=128),
        # num_stages=2 variants — target L1-throughput bottleneck (Test A)
        triton.Config({'BLOCK_B': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_B': 256}, num_warps=8, num_stages=2),
    ],
    key=['B_total'],
)
@triton.jit
def _optical_split_direct_T4_kernel(
    Mass_ptr, Ke_ptr, WeffFilter_ptr, Mu_ptr,
    W0_ptr, B0_ptr,
    WH0_f16_ptr, BH0_ptr, WH1_f16_ptr, BH1_ptr, WO_f16_ptr, BO_ptr,
    Wsel_ptr,
    Tfull_ptr, Et_ptr, Er_ptr, Ea_ptr,
    B_total,
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels
    c_idx = offs %  n_channels

    c8  = tl.arange(0, 8)
    c16 = tl.arange(0, 16)
    c32 = tl.arange(0, 32)

    mu = tl.load(Mu_ptr + b_idx, mask=mask, other=1.0)
    inv_mu = 1.0 / (mu + 1.0e-7)

    mass_pad = tl.load(Mass_ptr + b_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    W_ef_pad = tl.load(WeffFilter_ptr + c_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    ke_offsets = tl.where(c16 >= 2, c16 - 2, 0)
    ke_offsets = tl.where(c16 < 8, ke_offsets, 0)
    ke_load_mask = mask[:, None] & (c16[None, :] >= 2) & (c16[None, :] < 8)
    ke_pad = tl.load(Ke_ptr + b_idx[:, None] * 6 + ke_offsets[None, :],
                     mask=ke_load_mask, other=1.0)

    tau_pad = W_ef_pad * mass_pad * ke_pad
    tau_sum = tl.sum(tau_pad, axis=1)

    t_full = tl.exp(-tau_sum * inv_mu)
    one_minus_t = 1.0 - t_full

    x_d = tau_pad * inv_mu[:, None]
    x_d = tl.where(c16[None, :] == 8, mu[:, None], x_d)

    # Layer 0 (FP32)
    w0 = tl.load(W0_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0_ptr + c32)
    h = tl.dot(x_d, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    # Hidden 0 (FP16 TC, native FP16 weights)
    wh0 = tl.load(WH0_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bh0 = tl.load(BH0_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh0, out_dtype=tl.float32) + bh0[None, :]
    h = tl.maximum(h, 0.0)

    # Hidden 1 (FP16 TC)
    wh1 = tl.load(WH1_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bh1 = tl.load(BH1_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh1, out_dtype=tl.float32) + bh1[None, :]
    h = tl.maximum(h, 0.0)

    # Output (FP16 TC)
    wo = tl.load(WO_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BO_ptr + c32)
    out = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo, out_dtype=tl.float32) + bo[None, :]

    out_g = tl.reshape(out, [BLOCK_B * 8, 4])
    m = tl.max(out_g, axis=1)[:, None]
    out_g = tl.exp(out_g - m)
    s = tl.sum(out_g, axis=1)[:, None]
    out_g = out_g / s
    out_basis = tl.reshape(out_g, [BLOCK_B, 8, 4])

    wsel = tl.load(Wsel_ptr + c_idx[:, None] * 8 + c8[None, :],
                   mask=mask[:, None], other=0.0)
    combined = tl.sum(out_basis * wsel[:, :, None], axis=1)

    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    tl.store(Tfull_ptr + offs, t_full, mask=mask)
    tl.store(Et_ptr + offs, e_t_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Er_ptr + offs, e_r_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Ea_ptr + offs, e_a_e * inv_s2 * one_minus_t, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 128}, num_warps=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
        # maxnreg-capped variants — target 50% occupancy on T4
        triton.Config({'BLOCK_B': 128}, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_B': 256}, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_B': 256}, num_warps=8, maxnreg=128),
        # num_stages=2 variants — target L1-throughput bottleneck (Test A)
        triton.Config({'BLOCK_B': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_B': 256}, num_warps=8, num_stages=2),
    ],
    key=['B_total'],
)
@triton.jit
def _optical_split_diffuse_T4_kernel(
    Mass_ptr, Ke_ptr, WeffFilter_ptr,
    W0_ptr, B0_ptr,
    WH0_f16_ptr, BH0_ptr, WH1_f16_ptr, BH1_ptr, WO_f16_ptr, BO_ptr,
    Wsel_ptr,
    Tfull_ptr, Et_ptr, Er_ptr, Ea_ptr,
    inv_mu_diffuse,
    B_total,
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels
    c_idx = offs %  n_channels

    c8  = tl.arange(0, 8)
    c16 = tl.arange(0, 16)
    c32 = tl.arange(0, 32)

    mass_pad = tl.load(Mass_ptr + b_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    W_ef_pad = tl.load(WeffFilter_ptr + c_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    ke_offsets = tl.where(c16 >= 2, c16 - 2, 0)
    ke_offsets = tl.where(c16 < 8, ke_offsets, 0)
    ke_load_mask = mask[:, None] & (c16[None, :] >= 2) & (c16[None, :] < 8)
    ke_pad = tl.load(Ke_ptr + b_idx[:, None] * 6 + ke_offsets[None, :],
                     mask=ke_load_mask, other=1.0)

    tau_pad = W_ef_pad * mass_pad * ke_pad
    tau_sum = tl.sum(tau_pad, axis=1)

    t_full = tl.exp(-tau_sum * inv_mu_diffuse)
    one_minus_t = 1.0 - t_full

    # Layer 0 (FP32)
    w0 = tl.load(W0_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0_ptr + c32)
    h = tl.dot(tau_pad, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    # Hidden 0 (FP16 TC, native FP16 weights)
    wh0 = tl.load(WH0_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bh0 = tl.load(BH0_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh0, out_dtype=tl.float32) + bh0[None, :]
    h = tl.maximum(h, 0.0)

    # Hidden 1 (FP16 TC)
    wh1 = tl.load(WH1_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bh1 = tl.load(BH1_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh1, out_dtype=tl.float32) + bh1[None, :]
    h = tl.maximum(h, 0.0)

    # Output (FP16 TC)
    wo = tl.load(WO_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BO_ptr + c32)
    out = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo, out_dtype=tl.float32) + bo[None, :]

    out_g = tl.reshape(out, [BLOCK_B * 8, 4])
    m = tl.max(out_g, axis=1)[:, None]
    out_g = tl.exp(out_g - m)
    s = tl.sum(out_g, axis=1)[:, None]
    out_g = out_g / s
    out_basis = tl.reshape(out_g, [BLOCK_B, 8, 4])

    wsel = tl.load(Wsel_ptr + c_idx[:, None] * 8 + c8[None, :],
                   mask=mask[:, None], other=0.0)
    combined = tl.sum(out_basis * wsel[:, :, None], axis=1)

    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    tl.store(Tfull_ptr + offs, t_full, mask=mask)
    tl.store(Et_ptr + offs, e_t_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Er_ptr + offs, e_r_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Ea_ptr + offs, e_a_e * inv_s2 * one_minus_t, mask=mask)


# ---------------------------------------------------------------------------
# T4-optimized split kernels with BLOCK-DIAGONAL hidden weights.
# The hidden 32x32 matrices are actually block-diagonal with 4 blocks of 8x8
# (enforced in MultipleMLPs_Ix via multiplication by a block-diagonal filter).
# Triton's tl.dot on SM75 requires K>=16, so we pair two 8x8 blocks into a
# [16,16] block-diagonal weight and issue two matmuls per layer:
#   tl.dot([BLOCK_B, 16], [16, 16]) -> [BLOCK_B, 16]
# Each paired matmul is 50% sparse (off-diagonal rows of the [16,16] weight
# are zero); full dense 32x32 matmul is 75% sparse.  Net: 2x fewer useful
# FMAs per hidden-layer matmul.
#
# Weights expected pre-packed as [2, 16, 16] FP16 at ScatteringFused
# reconfigure time: slice 0 = dense[:16,:16] (blocks 0&1),
# slice 1 = dense[16:,16:] (blocks 2&3).  Output layer stays dense.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 128}, num_warps=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
    ],
    key=['B_total'],
)
@triton.jit
def _optical_split_direct_T4_bd_kernel(
    Mass_ptr, Ke_ptr, WeffFilter_ptr, Mu_ptr,
    W0_ptr, B0_ptr,
    WH0_BD_ptr, BH0_ptr,       # [2, 16, 16] FP16 — paired-block hidden 0
    WH1_BD_ptr, BH1_ptr,       # [2, 16, 16] FP16 — paired-block hidden 1
    WO_f16_ptr, BO_ptr,        # [32, 32] FP16 — dense output
    Wsel_ptr,
    Tfull_ptr, Et_ptr, Er_ptr, Ea_ptr,
    B_total,
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels
    c_idx = offs %  n_channels

    c8  = tl.arange(0, 8)
    c16 = tl.arange(0, 16)
    c32 = tl.arange(0, 32)

    mu = tl.load(Mu_ptr + b_idx, mask=mask, other=1.0)
    inv_mu = 1.0 / (mu + 1.0e-7)

    mass_pad = tl.load(Mass_ptr + b_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    W_ef_pad = tl.load(WeffFilter_ptr + c_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    ke_offsets = tl.where(c16 >= 2, c16 - 2, 0)
    ke_offsets = tl.where(c16 < 8, ke_offsets, 0)
    ke_load_mask = mask[:, None] & (c16[None, :] >= 2) & (c16[None, :] < 8)
    ke_pad = tl.load(Ke_ptr + b_idx[:, None] * 6 + ke_offsets[None, :],
                     mask=ke_load_mask, other=1.0)

    tau_pad = W_ef_pad * mass_pad * ke_pad
    tau_sum = tl.sum(tau_pad, axis=1)

    t_full = tl.exp(-tau_sum * inv_mu)
    one_minus_t = 1.0 - t_full

    x_d = tau_pad * inv_mu[:, None]
    x_d = tl.where(c16[None, :] == 8, mu[:, None], x_d)

    # Layer 0 (FP32, dense — not block-diagonal)
    w0 = tl.load(W0_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0_ptr + c32)
    h = tl.dot(x_d, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    # ---- Hidden 0 (paired block-diagonal: 2 blocks of [16,16], FP16 TC) ----
    # h is [BLOCK_B, 32].  Split into two halves [BLOCK_B, 16]:
    #   h_left  = h[:, 0:16]   -> paired against wh_bd[0]  (blocks 0&1)
    #   h_right = h[:, 16:32]  -> paired against wh_bd[1]  (blocks 2&3)
    # Triton: reshape to [BLOCK_B, 2, 16], trans so split dim is last, split.
    bh0 = tl.load(BH0_ptr + c32)                              # [32]
    h_fp16 = tl.minimum(h, 65504.0).to(tl.float16)            # [BLOCK_B, 32]
    h_r = tl.reshape(h_fp16, [BLOCK_B, 2, 16])                # [b, p, j] = h[b, p*16+j]
    h_r = tl.trans(h_r, (0, 2, 1))                            # [BLOCK_B, 16, 2]
    h_left, h_right = tl.split(h_r)                           # each [BLOCK_B, 16]
    # Load two paired weights [16, 16] each:
    wh0_l = tl.load(WH0_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    wh0_r = tl.load(WH0_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    # Two tl.dots, each [BLOCK_B, 16] x [16, 16] -> [BLOCK_B, 16]
    o_left  = tl.dot(h_left,  wh0_l, out_dtype=tl.float32)
    o_right = tl.dot(h_right, wh0_r, out_dtype=tl.float32)
    # Rejoin (inverse of split/trans/reshape):
    o_joined = tl.join(o_left, o_right)                       # [BLOCK_B, 16, 2]
    o_joined = tl.trans(o_joined, (0, 2, 1))                  # [BLOCK_B, 2, 16]
    h = tl.reshape(o_joined, [BLOCK_B, 32]) + bh0[None, :]
    h = tl.maximum(h, 0.0)

    # ---- Hidden 1 (same pattern) ----
    bh1 = tl.load(BH1_ptr + c32)
    h_fp16 = tl.minimum(h, 65504.0).to(tl.float16)
    h_r = tl.reshape(h_fp16, [BLOCK_B, 2, 16])
    h_r = tl.trans(h_r, (0, 2, 1))
    h_left, h_right = tl.split(h_r)
    wh1_l = tl.load(WH1_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    wh1_r = tl.load(WH1_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    o_left  = tl.dot(h_left,  wh1_l, out_dtype=tl.float32)
    o_right = tl.dot(h_right, wh1_r, out_dtype=tl.float32)
    o_joined = tl.join(o_left, o_right)
    o_joined = tl.trans(o_joined, (0, 2, 1))
    h = tl.reshape(o_joined, [BLOCK_B, 32]) + bh1[None, :]
    h = tl.maximum(h, 0.0)

    # ---- Output (dense, FP16 TC — unchanged) ----
    wo = tl.load(WO_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BO_ptr + c32)
    out = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo,
                 out_dtype=tl.float32) + bo[None, :]

    out_g = tl.reshape(out, [BLOCK_B * 8, 4])
    m = tl.max(out_g, axis=1)[:, None]
    out_g = tl.exp(out_g - m)
    s = tl.sum(out_g, axis=1)[:, None]
    out_g = out_g / s
    out_basis = tl.reshape(out_g, [BLOCK_B, 8, 4])

    wsel = tl.load(Wsel_ptr + c_idx[:, None] * 8 + c8[None, :],
                   mask=mask[:, None], other=0.0)
    combined = tl.sum(out_basis * wsel[:, :, None], axis=1)

    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    tl.store(Tfull_ptr + offs, t_full, mask=mask)
    tl.store(Et_ptr + offs, e_t_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Er_ptr + offs, e_r_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Ea_ptr + offs, e_a_e * inv_s2 * one_minus_t, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 128}, num_warps=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
    ],
    key=['B_total'],
)
@triton.jit
def _optical_split_diffuse_T4_bd_kernel(
    Mass_ptr, Ke_ptr, WeffFilter_ptr,
    W0_ptr, B0_ptr,
    WH0_BD_ptr, BH0_ptr,
    WH1_BD_ptr, BH1_ptr,
    WO_f16_ptr, BO_ptr,
    Wsel_ptr,
    Tfull_ptr, Et_ptr, Er_ptr, Ea_ptr,
    inv_mu_diffuse,
    B_total,
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels
    c_idx = offs %  n_channels

    c8  = tl.arange(0, 8)
    c16 = tl.arange(0, 16)
    c32 = tl.arange(0, 32)

    mass_pad = tl.load(Mass_ptr + b_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    W_ef_pad = tl.load(WeffFilter_ptr + c_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    ke_offsets = tl.where(c16 >= 2, c16 - 2, 0)
    ke_offsets = tl.where(c16 < 8, ke_offsets, 0)
    ke_load_mask = mask[:, None] & (c16[None, :] >= 2) & (c16[None, :] < 8)
    ke_pad = tl.load(Ke_ptr + b_idx[:, None] * 6 + ke_offsets[None, :],
                     mask=ke_load_mask, other=1.0)

    tau_pad = W_ef_pad * mass_pad * ke_pad
    tau_sum = tl.sum(tau_pad, axis=1)

    t_full = tl.exp(-tau_sum * inv_mu_diffuse)
    one_minus_t = 1.0 - t_full

    # Diffuse MLP takes tau_pad directly (no mu scaling, no mu column)
    x_d = tau_pad

    # Layer 0 (FP32, dense)
    w0 = tl.load(W0_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0_ptr + c32)
    h = tl.dot(x_d, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    # Hidden 0 (paired block-diagonal, same pattern as direct kernel)
    bh0 = tl.load(BH0_ptr + c32)
    h_fp16 = tl.minimum(h, 65504.0).to(tl.float16)
    h_r = tl.reshape(h_fp16, [BLOCK_B, 2, 16])
    h_r = tl.trans(h_r, (0, 2, 1))
    h_left, h_right = tl.split(h_r)
    wh0_l = tl.load(WH0_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    wh0_r = tl.load(WH0_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    o_left  = tl.dot(h_left,  wh0_l, out_dtype=tl.float32)
    o_right = tl.dot(h_right, wh0_r, out_dtype=tl.float32)
    o_joined = tl.join(o_left, o_right)
    o_joined = tl.trans(o_joined, (0, 2, 1))
    h = tl.reshape(o_joined, [BLOCK_B, 32]) + bh0[None, :]
    h = tl.maximum(h, 0.0)

    # Hidden 1 (paired block-diagonal)
    bh1 = tl.load(BH1_ptr + c32)
    h_fp16 = tl.minimum(h, 65504.0).to(tl.float16)
    h_r = tl.reshape(h_fp16, [BLOCK_B, 2, 16])
    h_r = tl.trans(h_r, (0, 2, 1))
    h_left, h_right = tl.split(h_r)
    wh1_l = tl.load(WH1_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    wh1_r = tl.load(WH1_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    o_left  = tl.dot(h_left,  wh1_l, out_dtype=tl.float32)
    o_right = tl.dot(h_right, wh1_r, out_dtype=tl.float32)
    o_joined = tl.join(o_left, o_right)
    o_joined = tl.trans(o_joined, (0, 2, 1))
    h = tl.reshape(o_joined, [BLOCK_B, 32]) + bh1[None, :]
    h = tl.maximum(h, 0.0)

    # Output (dense, FP16 TC)
    wo = tl.load(WO_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BO_ptr + c32)
    out = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo,
                 out_dtype=tl.float32) + bo[None, :]

    out_g = tl.reshape(out, [BLOCK_B * 8, 4])
    m = tl.max(out_g, axis=1)[:, None]
    out_g = tl.exp(out_g - m)
    s = tl.sum(out_g, axis=1)[:, None]
    out_g = out_g / s
    out_basis = tl.reshape(out_g, [BLOCK_B, 8, 4])

    wsel = tl.load(Wsel_ptr + c_idx[:, None] * 8 + c8[None, :],
                   mask=mask[:, None], other=0.0)
    combined = tl.sum(out_basis * wsel[:, :, None], axis=1)

    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    tl.store(Tfull_ptr + offs, t_full, mask=mask)
    tl.store(Et_ptr + offs, e_t_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Er_ptr + offs, e_r_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Ea_ptr + offs, e_a_e * inv_s2 * one_minus_t, mask=mask)


# ---------------------------------------------------------------------------
# T4-optimized split kernels with BLOCK-DIAGONAL weights — V2 (split-path).
#
# Called by launch_optical_scattering_bd_v2
# Separate "split" kernels for direct and diffuse radiation
#
# Keeps TWO independent [BLOCK_B, 16] streams from input to output.
# The input layer is split into "left/right" halves at load time; the block-
# diagonal hidden layers and output layer each operate on their half
# independently; the two halves merge only at the wsel combination step.
#
# All four weight layers use pre-packed [2, 16, 16] buffers:
#   - w0_bd:      [2, 16, 16] FP32   (input: w0[:,:16] and w0[:,16:])
#   - wh0_f16_bd: [2, 16, 16] FP16   (hidden 0 diagonal blocks)
#   - wh1_f16_bd: [2, 16, 16] FP16   (hidden 1 diagonal blocks)
#   - wo_f16_bd:  [2, 16, 16] FP16   (output diagonal blocks)
#
# Net effect: zero reshape/trans/split/join; 2× fewer MMA tiles for hidden
# and output layers; same FLOPs for input layer.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 128}, num_warps=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
        triton.Config({'BLOCK_B': 512}, num_warps=8),
    ],
    key=['B_total'],
)
@triton.jit
def _optical_split_direct_T4_bd_v2_kernel(
    Mass_ptr, Ke_ptr, WeffFilter_ptr, Mu_ptr,
    W0_BD_ptr, B0_ptr,             # [2, 16, 16] FP32, [32]
    WH0_BD_ptr, BH0_ptr,           # [2, 16, 16] FP16, [32]
    WH1_BD_ptr, BH1_ptr,           # [2, 16, 16] FP16, [32]
    WO_BD_ptr, BO_ptr,             # [2, 16, 16] FP16, [32]
    Wsel_ptr,
    Tfull_ptr, Et_ptr, Er_ptr, Ea_ptr,
    B_total,
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels
    c_idx = offs %  n_channels

    c4  = tl.arange(0, 4)
    c16 = tl.arange(0, 16)

    mu = tl.load(Mu_ptr + b_idx, mask=mask, other=1.0)
    inv_mu = 1.0 / (mu + 1.0e-7)

    mass_pad = tl.load(Mass_ptr + b_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    W_ef_pad = tl.load(WeffFilter_ptr + c_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    ke_offsets = tl.where(c16 >= 2, c16 - 2, 0)
    ke_offsets = tl.where(c16 < 8, ke_offsets, 0)
    ke_load_mask = mask[:, None] & (c16[None, :] >= 2) & (c16[None, :] < 8)
    ke_pad = tl.load(Ke_ptr + b_idx[:, None] * 6 + ke_offsets[None, :],
                     mask=ke_load_mask, other=1.0)

    tau_pad = W_ef_pad * mass_pad * ke_pad
    tau_sum = tl.sum(tau_pad, axis=1)

    t_full = tl.exp(-tau_sum * inv_mu)
    one_minus_t = 1.0 - t_full

    x_d = tau_pad * inv_mu[:, None]
    x_d = tl.where(c16[None, :] == 8, mu[:, None], x_d)

    # ---- Layer 0 (FP32, split into left/right [BLOCK_B, 16] streams) ----
    w0_L = tl.load(W0_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    w0_R = tl.load(W0_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    b0_L = tl.load(B0_ptr + c16)
    b0_R = tl.load(B0_ptr + c16 + 16)
    h_L = tl.dot(x_d, w0_L, input_precision="ieee") + b0_L[None, :]
    h_R = tl.dot(x_d, w0_R, input_precision="ieee") + b0_R[None, :]
    h_L = tl.maximum(h_L, 0.0)
    h_R = tl.maximum(h_R, 0.0)

    # ---- Hidden 0 (FP16 TC, block-diagonal) ----
    wh0_L = tl.load(WH0_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    wh0_R = tl.load(WH0_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    bh0_L = tl.load(BH0_ptr + c16)
    bh0_R = tl.load(BH0_ptr + c16 + 16)
    h_L = tl.dot(tl.minimum(h_L, 65504.0).to(tl.float16), wh0_L, out_dtype=tl.float32) + bh0_L[None, :]
    h_R = tl.dot(tl.minimum(h_R, 65504.0).to(tl.float16), wh0_R, out_dtype=tl.float32) + bh0_R[None, :]
    h_L = tl.maximum(h_L, 0.0)
    h_R = tl.maximum(h_R, 0.0)

    # ---- Hidden 1 (FP16 TC, block-diagonal) ----
    wh1_L = tl.load(WH1_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    wh1_R = tl.load(WH1_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    bh1_L = tl.load(BH1_ptr + c16)
    bh1_R = tl.load(BH1_ptr + c16 + 16)
    h_L = tl.dot(tl.minimum(h_L, 65504.0).to(tl.float16), wh1_L, out_dtype=tl.float32) + bh1_L[None, :]
    h_R = tl.dot(tl.minimum(h_R, 65504.0).to(tl.float16), wh1_R, out_dtype=tl.float32) + bh1_R[None, :]
    h_L = tl.maximum(h_L, 0.0)
    h_R = tl.maximum(h_R, 0.0)

    # ---- Output (FP16 TC, block-diagonal) ----
    wo_L = tl.load(WO_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    wo_R = tl.load(WO_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    bo_L = tl.load(BO_ptr + c16)
    bo_R = tl.load(BO_ptr + c16 + 16)
    out_L = tl.dot(tl.minimum(h_L, 65504.0).to(tl.float16), wo_L, out_dtype=tl.float32) + bo_L[None, :]
    out_R = tl.dot(tl.minimum(h_R, 65504.0).to(tl.float16), wo_R, out_dtype=tl.float32) + bo_R[None, :]

    # ---- Per-group softmax (4 groups of 4 per half) ----
    out_gL = tl.reshape(out_L, [BLOCK_B * 4, 4])
    m = tl.max(out_gL, axis=1)[:, None]
    out_gL = tl.exp(out_gL - m)
    s = tl.sum(out_gL, axis=1)[:, None]
    out_gL = out_gL / s
    out_basis_L = tl.reshape(out_gL, [BLOCK_B, 4, 4])

    out_gR = tl.reshape(out_R, [BLOCK_B * 4, 4])
    m = tl.max(out_gR, axis=1)[:, None]
    out_gR = tl.exp(out_gR - m)
    s = tl.sum(out_gR, axis=1)[:, None]
    out_gR = out_gR / s
    out_basis_R = tl.reshape(out_gR, [BLOCK_B, 4, 4])

    # ---- wsel combination: merge left (groups 0-3) + right (groups 4-7) ----
    wsel_L = tl.load(Wsel_ptr + c_idx[:, None] * 8 + c4[None, :],
                     mask=mask[:, None], other=0.0)
    wsel_R = tl.load(Wsel_ptr + c_idx[:, None] * 8 + (c4[None, :] + 4),
                     mask=mask[:, None], other=0.0)
    combined = (tl.sum(out_basis_L * wsel_L[:, :, None], axis=1) +
                tl.sum(out_basis_R * wsel_R[:, :, None], axis=1))

    # ---- Final 3-way softmax (T, R, A) ----
    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    tl.store(Tfull_ptr + offs, t_full, mask=mask)
    tl.store(Et_ptr + offs, e_t_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Er_ptr + offs, e_r_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Ea_ptr + offs, e_a_e * inv_s2 * one_minus_t, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 128}, num_warps=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
        triton.Config({'BLOCK_B': 512}, num_warps=8),
    ],
    key=['B_total'],
)
@triton.jit
def _optical_split_diffuse_T4_bd_v2_kernel(
    Mass_ptr, Ke_ptr, WeffFilter_ptr,
    W0_BD_ptr, B0_ptr,
    WH0_BD_ptr, BH0_ptr,
    WH1_BD_ptr, BH1_ptr,
    WO_BD_ptr, BO_ptr,
    Wsel_ptr,
    Tfull_ptr, Et_ptr, Er_ptr, Ea_ptr,
    inv_mu_diffuse,
    B_total,
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels
    c_idx = offs %  n_channels

    c4  = tl.arange(0, 4)
    c16 = tl.arange(0, 16)

    mass_pad = tl.load(Mass_ptr + b_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    W_ef_pad = tl.load(WeffFilter_ptr + c_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    ke_offsets = tl.where(c16 >= 2, c16 - 2, 0)
    ke_offsets = tl.where(c16 < 8, ke_offsets, 0)
    ke_load_mask = mask[:, None] & (c16[None, :] >= 2) & (c16[None, :] < 8)
    ke_pad = tl.load(Ke_ptr + b_idx[:, None] * 6 + ke_offsets[None, :],
                     mask=ke_load_mask, other=1.0)

    tau_pad = W_ef_pad * mass_pad * ke_pad
    tau_sum = tl.sum(tau_pad, axis=1)

    t_full = tl.exp(-tau_sum * inv_mu_diffuse)
    one_minus_t = 1.0 - t_full

    x_d = tau_pad

    # ---- Layer 0 (FP32, split into left/right [BLOCK_B, 16] streams) ----
    w0_L = tl.load(W0_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    w0_R = tl.load(W0_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    b0_L = tl.load(B0_ptr + c16)
    b0_R = tl.load(B0_ptr + c16 + 16)
    h_L = tl.dot(x_d, w0_L, input_precision="ieee") + b0_L[None, :]
    h_R = tl.dot(x_d, w0_R, input_precision="ieee") + b0_R[None, :]
    h_L = tl.maximum(h_L, 0.0)
    h_R = tl.maximum(h_R, 0.0)

    # ---- Hidden 0 (FP16 TC, block-diagonal) ----
    wh0_L = tl.load(WH0_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    wh0_R = tl.load(WH0_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    bh0_L = tl.load(BH0_ptr + c16)
    bh0_R = tl.load(BH0_ptr + c16 + 16)
    h_L = tl.dot(tl.minimum(h_L, 65504.0).to(tl.float16), wh0_L, out_dtype=tl.float32) + bh0_L[None, :]
    h_R = tl.dot(tl.minimum(h_R, 65504.0).to(tl.float16), wh0_R, out_dtype=tl.float32) + bh0_R[None, :]
    h_L = tl.maximum(h_L, 0.0)
    h_R = tl.maximum(h_R, 0.0)

    # ---- Hidden 1 (FP16 TC, block-diagonal) ----
    wh1_L = tl.load(WH1_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    wh1_R = tl.load(WH1_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    bh1_L = tl.load(BH1_ptr + c16)
    bh1_R = tl.load(BH1_ptr + c16 + 16)
    h_L = tl.dot(tl.minimum(h_L, 65504.0).to(tl.float16), wh1_L, out_dtype=tl.float32) + bh1_L[None, :]
    h_R = tl.dot(tl.minimum(h_R, 65504.0).to(tl.float16), wh1_R, out_dtype=tl.float32) + bh1_R[None, :]
    h_L = tl.maximum(h_L, 0.0)
    h_R = tl.maximum(h_R, 0.0)

    # ---- Output (FP16 TC, block-diagonal) ----
    wo_L = tl.load(WO_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    wo_R = tl.load(WO_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    bo_L = tl.load(BO_ptr + c16)
    bo_R = tl.load(BO_ptr + c16 + 16)
    out_L = tl.dot(tl.minimum(h_L, 65504.0).to(tl.float16), wo_L, out_dtype=tl.float32) + bo_L[None, :]
    out_R = tl.dot(tl.minimum(h_R, 65504.0).to(tl.float16), wo_R, out_dtype=tl.float32) + bo_R[None, :]

    # ---- Per-group softmax (4 groups of 4 per half) ----
    out_gL = tl.reshape(out_L, [BLOCK_B * 4, 4])
    m = tl.max(out_gL, axis=1)[:, None]
    out_gL = tl.exp(out_gL - m)
    s = tl.sum(out_gL, axis=1)[:, None]
    out_gL = out_gL / s
    out_basis_L = tl.reshape(out_gL, [BLOCK_B, 4, 4])

    out_gR = tl.reshape(out_R, [BLOCK_B * 4, 4])
    m = tl.max(out_gR, axis=1)[:, None]
    out_gR = tl.exp(out_gR - m)
    s = tl.sum(out_gR, axis=1)[:, None]
    out_gR = out_gR / s
    out_basis_R = tl.reshape(out_gR, [BLOCK_B, 4, 4])

    # ---- wsel combination: merge left (groups 0-3) + right (groups 4-7) ----
    wsel_L = tl.load(Wsel_ptr + c_idx[:, None] * 8 + c4[None, :],
                     mask=mask[:, None], other=0.0)
    wsel_R = tl.load(Wsel_ptr + c_idx[:, None] * 8 + (c4[None, :] + 4),
                     mask=mask[:, None], other=0.0)
    combined = (tl.sum(out_basis_L * wsel_L[:, :, None], axis=1) +
                tl.sum(out_basis_R * wsel_R[:, :, None], axis=1))

    # ---- Final 3-way softmax (T, R, A) ----
    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    tl.store(Tfull_ptr + offs, t_full, mask=mask)
    tl.store(Et_ptr + offs, e_t_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Er_ptr + offs, e_r_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Ea_ptr + offs, e_a_e * inv_s2 * one_minus_t, mask=mask)


# ---------------------------------------------------------------------------
# BD v2 SPLIT-SERIAL variant: left and right halves in separate kernels.
#
# Each sub-kernel carries only ONE [BLOCK_B, 16] activation stream (half
# the register pressure of the unified BD v2 kernel).  The left kernel
# stores t_full and a partial wsel combination [B_total, 4] to DRAM;
# the right kernel reads the partial, adds its own, and writes final
# outputs.  Input data (mass/ke/w_eff/mu) is loaded in both (duplicated).
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 128}, num_warps=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
        triton.Config({'BLOCK_B': 512}, num_warps=8),
    ],
    key=['B_total'],
)
@triton.jit
def _optical_direct_T4_bd_v2_left_kernel(
    Mass_ptr, Ke_ptr, WeffFilter_ptr, Mu_ptr,
    W0_BD_ptr, B0_ptr,
    WH0_BD_ptr, BH0_ptr,
    WH1_BD_ptr, BH1_ptr,
    WO_BD_ptr, BO_ptr,
    Wsel_ptr,
    Tfull_ptr, Partial_ptr,        # [B_total], [B_total, 4]
    B_total,
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels
    c_idx = offs %  n_channels

    c4  = tl.arange(0, 4)
    c16 = tl.arange(0, 16)

    mu = tl.load(Mu_ptr + b_idx, mask=mask, other=1.0)
    inv_mu = 1.0 / (mu + 1.0e-7)

    mass_pad = tl.load(Mass_ptr + b_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    W_ef_pad = tl.load(WeffFilter_ptr + c_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    ke_offsets = tl.where(c16 >= 2, c16 - 2, 0)
    ke_offsets = tl.where(c16 < 8, ke_offsets, 0)
    ke_load_mask = mask[:, None] & (c16[None, :] >= 2) & (c16[None, :] < 8)
    ke_pad = tl.load(Ke_ptr + b_idx[:, None] * 6 + ke_offsets[None, :],
                     mask=ke_load_mask, other=1.0)

    tau_pad = W_ef_pad * mass_pad * ke_pad
    tau_sum = tl.sum(tau_pad, axis=1)
    t_full = tl.exp(-tau_sum * inv_mu)

    x_d = tau_pad * inv_mu[:, None]
    x_d = tl.where(c16[None, :] == 8, mu[:, None], x_d)

    # ---- Left path only: 4 dots through all layers ----
    w0_L = tl.load(W0_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    b0_L = tl.load(B0_ptr + c16)
    h = tl.dot(x_d, w0_L, input_precision="ieee") + b0_L[None, :]
    h = tl.maximum(h, 0.0)

    wh0_L = tl.load(WH0_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    bh0_L = tl.load(BH0_ptr + c16)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh0_L, out_dtype=tl.float32) + bh0_L[None, :]
    h = tl.maximum(h, 0.0)

    wh1_L = tl.load(WH1_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    bh1_L = tl.load(BH1_ptr + c16)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh1_L, out_dtype=tl.float32) + bh1_L[None, :]
    h = tl.maximum(h, 0.0)

    wo_L = tl.load(WO_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    bo_L = tl.load(BO_ptr + c16)
    out_L = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo_L, out_dtype=tl.float32) + bo_L[None, :]

    # Softmax over 4 groups of 4
    out_gL = tl.reshape(out_L, [BLOCK_B * 4, 4])
    m = tl.max(out_gL, axis=1)[:, None]
    out_gL = tl.exp(out_gL - m)
    s = tl.sum(out_gL, axis=1)[:, None]
    out_gL = out_gL / s
    out_basis_L = tl.reshape(out_gL, [BLOCK_B, 4, 4])

    # Partial wsel combination (left groups 0-3)
    wsel_L = tl.load(Wsel_ptr + c_idx[:, None] * 8 + c4[None, :],
                     mask=mask[:, None], other=0.0)
    partial_L = tl.sum(out_basis_L * wsel_L[:, :, None], axis=1)   # [BLOCK_B, 4]

    # Store t_full and partial combination
    tl.store(Tfull_ptr + offs, t_full, mask=mask)
    tl.store(Partial_ptr + offs[:, None] * 4 + c4[None, :], partial_L, mask=mask[:, None])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 128}, num_warps=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
        triton.Config({'BLOCK_B': 512}, num_warps=8),
    ],
    key=['B_total'],
)
@triton.jit
def _optical_direct_T4_bd_v2_right_kernel(
    Mass_ptr, Ke_ptr, WeffFilter_ptr, Mu_ptr,
    W0_BD_ptr, B0_ptr,
    WH0_BD_ptr, BH0_ptr,
    WH1_BD_ptr, BH1_ptr,
    WO_BD_ptr, BO_ptr,
    Wsel_ptr,
    Partial_ptr,                   # [B_total, 4] from left kernel
    Et_ptr, Er_ptr, Ea_ptr,
    B_total,
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels
    c_idx = offs %  n_channels

    c4  = tl.arange(0, 4)
    c16 = tl.arange(0, 16)

    # Recompute tau and t_full (duplicated from left kernel)
    mu = tl.load(Mu_ptr + b_idx, mask=mask, other=1.0)
    inv_mu = 1.0 / (mu + 1.0e-7)

    mass_pad = tl.load(Mass_ptr + b_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    W_ef_pad = tl.load(WeffFilter_ptr + c_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    ke_offsets = tl.where(c16 >= 2, c16 - 2, 0)
    ke_offsets = tl.where(c16 < 8, ke_offsets, 0)
    ke_load_mask = mask[:, None] & (c16[None, :] >= 2) & (c16[None, :] < 8)
    ke_pad = tl.load(Ke_ptr + b_idx[:, None] * 6 + ke_offsets[None, :],
                     mask=ke_load_mask, other=1.0)

    tau_pad = W_ef_pad * mass_pad * ke_pad
    tau_sum = tl.sum(tau_pad, axis=1)
    t_full = tl.exp(-tau_sum * inv_mu)
    one_minus_t = 1.0 - t_full

    x_d = tau_pad * inv_mu[:, None]
    x_d = tl.where(c16[None, :] == 8, mu[:, None], x_d)

    # ---- Right path only: 4 dots through all layers ----
    w0_R = tl.load(W0_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    b0_R = tl.load(B0_ptr + c16 + 16)
    h = tl.dot(x_d, w0_R, input_precision="ieee") + b0_R[None, :]
    h = tl.maximum(h, 0.0)

    wh0_R = tl.load(WH0_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    bh0_R = tl.load(BH0_ptr + c16 + 16)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh0_R, out_dtype=tl.float32) + bh0_R[None, :]
    h = tl.maximum(h, 0.0)

    wh1_R = tl.load(WH1_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    bh1_R = tl.load(BH1_ptr + c16 + 16)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh1_R, out_dtype=tl.float32) + bh1_R[None, :]
    h = tl.maximum(h, 0.0)

    wo_R = tl.load(WO_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    bo_R = tl.load(BO_ptr + c16 + 16)
    out_R = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo_R, out_dtype=tl.float32) + bo_R[None, :]

    # Softmax over 4 groups of 4
    out_gR = tl.reshape(out_R, [BLOCK_B * 4, 4])
    m = tl.max(out_gR, axis=1)[:, None]
    out_gR = tl.exp(out_gR - m)
    s = tl.sum(out_gR, axis=1)[:, None]
    out_gR = out_gR / s
    out_basis_R = tl.reshape(out_gR, [BLOCK_B, 4, 4])

    # Right partial wsel combination (groups 4-7)
    wsel_R = tl.load(Wsel_ptr + c_idx[:, None] * 8 + (c4[None, :] + 4),
                     mask=mask[:, None], other=0.0)
    partial_R = tl.sum(out_basis_R * wsel_R[:, :, None], axis=1)

    # Load left partial and merge
    partial_L = tl.load(Partial_ptr + offs[:, None] * 4 + c4[None, :],
                        mask=mask[:, None], other=0.0)
    combined = partial_L + partial_R

    # Final 3-way softmax (T, R, A)
    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    tl.store(Et_ptr + offs, e_t_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Er_ptr + offs, e_r_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Ea_ptr + offs, e_a_e * inv_s2 * one_minus_t, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 128}, num_warps=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
        triton.Config({'BLOCK_B': 512}, num_warps=8),
    ],
    key=['B_total'],
)
@triton.jit
def _optical_diffuse_T4_bd_v2_left_kernel(
    Mass_ptr, Ke_ptr, WeffFilter_ptr,
    W0_BD_ptr, B0_ptr,
    WH0_BD_ptr, BH0_ptr,
    WH1_BD_ptr, BH1_ptr,
    WO_BD_ptr, BO_ptr,
    Wsel_ptr,
    Tfull_ptr, Partial_ptr,
    inv_mu_diffuse,
    B_total,
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels
    c_idx = offs %  n_channels

    c4  = tl.arange(0, 4)
    c16 = tl.arange(0, 16)

    mass_pad = tl.load(Mass_ptr + b_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    W_ef_pad = tl.load(WeffFilter_ptr + c_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    ke_offsets = tl.where(c16 >= 2, c16 - 2, 0)
    ke_offsets = tl.where(c16 < 8, ke_offsets, 0)
    ke_load_mask = mask[:, None] & (c16[None, :] >= 2) & (c16[None, :] < 8)
    ke_pad = tl.load(Ke_ptr + b_idx[:, None] * 6 + ke_offsets[None, :],
                     mask=ke_load_mask, other=1.0)

    tau_pad = W_ef_pad * mass_pad * ke_pad
    tau_sum = tl.sum(tau_pad, axis=1)
    t_full = tl.exp(-tau_sum * inv_mu_diffuse)

    x_d = tau_pad

    # ---- Left path only ----
    w0_L = tl.load(W0_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    b0_L = tl.load(B0_ptr + c16)
    h = tl.dot(x_d, w0_L, input_precision="ieee") + b0_L[None, :]
    h = tl.maximum(h, 0.0)

    wh0_L = tl.load(WH0_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    bh0_L = tl.load(BH0_ptr + c16)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh0_L, out_dtype=tl.float32) + bh0_L[None, :]
    h = tl.maximum(h, 0.0)

    wh1_L = tl.load(WH1_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    bh1_L = tl.load(BH1_ptr + c16)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh1_L, out_dtype=tl.float32) + bh1_L[None, :]
    h = tl.maximum(h, 0.0)

    wo_L = tl.load(WO_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    bo_L = tl.load(BO_ptr + c16)
    out_L = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo_L, out_dtype=tl.float32) + bo_L[None, :]

    out_gL = tl.reshape(out_L, [BLOCK_B * 4, 4])
    m = tl.max(out_gL, axis=1)[:, None]
    out_gL = tl.exp(out_gL - m)
    s = tl.sum(out_gL, axis=1)[:, None]
    out_gL = out_gL / s
    out_basis_L = tl.reshape(out_gL, [BLOCK_B, 4, 4])

    wsel_L = tl.load(Wsel_ptr + c_idx[:, None] * 8 + c4[None, :],
                     mask=mask[:, None], other=0.0)
    partial_L = tl.sum(out_basis_L * wsel_L[:, :, None], axis=1)

    tl.store(Tfull_ptr + offs, t_full, mask=mask)
    tl.store(Partial_ptr + offs[:, None] * 4 + c4[None, :], partial_L, mask=mask[:, None])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 128}, num_warps=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
        triton.Config({'BLOCK_B': 512}, num_warps=8),
    ],
    key=['B_total'],
)
@triton.jit
def _optical_diffuse_T4_bd_v2_right_kernel(
    Mass_ptr, Ke_ptr, WeffFilter_ptr,
    W0_BD_ptr, B0_ptr,
    WH0_BD_ptr, BH0_ptr,
    WH1_BD_ptr, BH1_ptr,
    WO_BD_ptr, BO_ptr,
    Wsel_ptr,
    Partial_ptr,
    Et_ptr, Er_ptr, Ea_ptr,
    inv_mu_diffuse,
    B_total,
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels
    c_idx = offs %  n_channels

    c4  = tl.arange(0, 4)
    c16 = tl.arange(0, 16)

    mass_pad = tl.load(Mass_ptr + b_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    W_ef_pad = tl.load(WeffFilter_ptr + c_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    ke_offsets = tl.where(c16 >= 2, c16 - 2, 0)
    ke_offsets = tl.where(c16 < 8, ke_offsets, 0)
    ke_load_mask = mask[:, None] & (c16[None, :] >= 2) & (c16[None, :] < 8)
    ke_pad = tl.load(Ke_ptr + b_idx[:, None] * 6 + ke_offsets[None, :],
                     mask=ke_load_mask, other=1.0)

    tau_pad = W_ef_pad * mass_pad * ke_pad
    tau_sum = tl.sum(tau_pad, axis=1)
    t_full = tl.exp(-tau_sum * inv_mu_diffuse)
    one_minus_t = 1.0 - t_full

    x_d = tau_pad

    # ---- Right path only ----
    w0_R = tl.load(W0_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    b0_R = tl.load(B0_ptr + c16 + 16)
    h = tl.dot(x_d, w0_R, input_precision="ieee") + b0_R[None, :]
    h = tl.maximum(h, 0.0)

    wh0_R = tl.load(WH0_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    bh0_R = tl.load(BH0_ptr + c16 + 16)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh0_R, out_dtype=tl.float32) + bh0_R[None, :]
    h = tl.maximum(h, 0.0)

    wh1_R = tl.load(WH1_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    bh1_R = tl.load(BH1_ptr + c16 + 16)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh1_R, out_dtype=tl.float32) + bh1_R[None, :]
    h = tl.maximum(h, 0.0)

    wo_R = tl.load(WO_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    bo_R = tl.load(BO_ptr + c16 + 16)
    out_R = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo_R, out_dtype=tl.float32) + bo_R[None, :]

    out_gR = tl.reshape(out_R, [BLOCK_B * 4, 4])
    m = tl.max(out_gR, axis=1)[:, None]
    out_gR = tl.exp(out_gR - m)
    s = tl.sum(out_gR, axis=1)[:, None]
    out_gR = out_gR / s
    out_basis_R = tl.reshape(out_gR, [BLOCK_B, 4, 4])

    wsel_R = tl.load(Wsel_ptr + c_idx[:, None] * 8 + (c4[None, :] + 4),
                     mask=mask[:, None], other=0.0)
    partial_R = tl.sum(out_basis_R * wsel_R[:, :, None], axis=1)

    # Load left partial and merge
    partial_L = tl.load(Partial_ptr + offs[:, None] * 4 + c4[None, :],
                        mask=mask[:, None], other=0.0)
    combined = partial_L + partial_R

    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    tl.store(Et_ptr + offs, e_t_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Er_ptr + offs, e_r_e * inv_s2 * one_minus_t, mask=mask)
    tl.store(Ea_ptr + offs, e_a_e * inv_s2 * one_minus_t, mask=mask)


# ---------------------------------------------------------------------------
# T4-optimized combined kernel: both MLPs in one launch, pre-stored FP16
# weights for hidden/output layers.  Reads mass/ke/W_eff/mu ONCE (halves
# DRAM traffic vs split kernels) and loads FP16 weights directly (no
# in-kernel conversion → less shared memory → better occupancy).
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 128}, num_warps=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
        triton.Config({'BLOCK_B': 512}, num_warps=8),
        # maxnreg-capped variants (may lose to spills, autotune will decide)
        triton.Config({'BLOCK_B': 128}, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_B': 256}, num_warps=4, maxnreg=128),
        # num_stages=2 variants (less smem pressure)
        triton.Config({'BLOCK_B': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4, num_stages=2),
    ],
    key=['B_total'],
)
@triton.jit
def _optical_combined_T4_kernel(
    Mass_ptr, Ke_ptr, WeffFilter_ptr, Mu_ptr,
    # Direct MLP: first layer FP32, hidden/output FP16
    W0d_ptr, B0d_ptr,
    WH0d_f16_ptr, BH0d_ptr, WH1d_f16_ptr, BH1d_ptr, WOd_f16_ptr, BOd_ptr,
    Wsel_dir_ptr,
    # Diffuse MLP: first layer FP32, hidden/output FP16
    W0f_ptr, B0f_ptr,
    WH0f_f16_ptr, BH0f_ptr, WH1f_f16_ptr, BH1f_ptr, WOf_f16_ptr, BOf_ptr,
    Wsel_dif_ptr,
    # Outputs
    Tfull_dir_ptr, Et_dir_ptr, Er_dir_ptr, Ea_dir_ptr,
    Tfull_dif_ptr, Et_dif_ptr, Er_dif_ptr, Ea_dif_ptr,
    inv_mu_diffuse,
    B_total,
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels
    c_idx = offs %  n_channels

    c8  = tl.arange(0, 8)
    c16 = tl.arange(0, 16)
    c32 = tl.arange(0, 32)

    # ---- Load inputs ONCE (shared between direct and diffuse) ----
    mu = tl.load(Mu_ptr + b_idx, mask=mask, other=1.0)
    inv_mu = 1.0 / (mu + 1.0e-7)

    mass_pad = tl.load(Mass_ptr + b_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    W_ef_pad = tl.load(WeffFilter_ptr + c_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    ke_offsets = tl.where(c16 >= 2, c16 - 2, 0)
    ke_offsets = tl.where(c16 < 8, ke_offsets, 0)
    ke_load_mask = mask[:, None] & (c16[None, :] >= 2) & (c16[None, :] < 8)
    ke_pad = tl.load(Ke_ptr + b_idx[:, None] * 6 + ke_offsets[None, :],
                     mask=ke_load_mask, other=1.0)

    tau_pad = W_ef_pad * mass_pad * ke_pad
    tau_sum = tl.sum(tau_pad, axis=1)

    t_full_direct  = tl.exp(-tau_sum * inv_mu)
    t_full_diffuse = tl.exp(-tau_sum * inv_mu_diffuse)
    one_minus_t_dir = 1.0 - t_full_direct
    one_minus_t_dif = 1.0 - t_full_diffuse

    # =====================================================================
    # ---------------------- DIRECT branch --------------------------------
    # =====================================================================
    x_d = tau_pad * inv_mu[:, None]
    x_d = tl.where(c16[None, :] == 8, mu[:, None], x_d)

    # Layer 0 (FP32)
    w0 = tl.load(W0d_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0d_ptr + c32)
    h = tl.dot(x_d, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    # Hidden 0 (FP16 TC, pre-stored FP16 weights)
    wh0 = tl.load(WH0d_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bh0 = tl.load(BH0d_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh0, out_dtype=tl.float32) + bh0[None, :]
    h = tl.maximum(h, 0.0)

    # Hidden 1 (FP16 TC)
    wh1 = tl.load(WH1d_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bh1 = tl.load(BH1d_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh1, out_dtype=tl.float32) + bh1[None, :]
    h = tl.maximum(h, 0.0)

    # Output (FP16 TC)
    wo = tl.load(WOd_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BOd_ptr + c32)
    out = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo, out_dtype=tl.float32) + bo[None, :]

    # Per-group softmax (8 groups of 4)
    out_g = tl.reshape(out, [BLOCK_B * 8, 4])
    m = tl.max(out_g, axis=1)[:, None]
    out_g = tl.exp(out_g - m)
    s = tl.sum(out_g, axis=1)[:, None]
    out_g = out_g / s
    out_basis = tl.reshape(out_g, [BLOCK_B, 8, 4])

    wsel = tl.load(Wsel_dir_ptr + c_idx[:, None] * 8 + c8[None, :],
                   mask=mask[:, None], other=0.0)
    combined = tl.sum(out_basis * wsel[:, :, None], axis=1)

    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    tl.store(Tfull_dir_ptr + offs, t_full_direct, mask=mask)
    tl.store(Et_dir_ptr + offs, e_t_e * inv_s2 * one_minus_t_dir, mask=mask)
    tl.store(Er_dir_ptr + offs, e_r_e * inv_s2 * one_minus_t_dir, mask=mask)
    tl.store(Ea_dir_ptr + offs, e_a_e * inv_s2 * one_minus_t_dir, mask=mask)

    # =====================================================================
    # ---------------------- DIFFUSE branch --------------------------------
    # =====================================================================
    w0 = tl.load(W0f_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0f_ptr + c32)
    h = tl.dot(tau_pad, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    wh0 = tl.load(WH0f_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bh0 = tl.load(BH0f_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh0, out_dtype=tl.float32) + bh0[None, :]
    h = tl.maximum(h, 0.0)

    wh1 = tl.load(WH1f_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bh1 = tl.load(BH1f_ptr + c32)
    h = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wh1, out_dtype=tl.float32) + bh1[None, :]
    h = tl.maximum(h, 0.0)

    wo = tl.load(WOf_f16_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BOf_ptr + c32)
    out = tl.dot(tl.minimum(h, 65504.0).to(tl.float16), wo, out_dtype=tl.float32) + bo[None, :]

    out_g = tl.reshape(out, [BLOCK_B * 8, 4])
    m = tl.max(out_g, axis=1)[:, None]
    out_g = tl.exp(out_g - m)
    s = tl.sum(out_g, axis=1)[:, None]
    out_g = out_g / s
    out_basis = tl.reshape(out_g, [BLOCK_B, 8, 4])

    wsel = tl.load(Wsel_dif_ptr + c_idx[:, None] * 8 + c8[None, :],
                   mask=mask[:, None], other=0.0)
    combined = tl.sum(out_basis * wsel[:, :, None], axis=1)

    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    tl.store(Tfull_dif_ptr + offs, t_full_diffuse, mask=mask)
    tl.store(Et_dif_ptr + offs, e_t_e * inv_s2 * one_minus_t_dif, mask=mask)
    tl.store(Er_dif_ptr + offs, e_r_e * inv_s2 * one_minus_t_dif, mask=mask)
    tl.store(Ea_dif_ptr + offs, e_a_e * inv_s2 * one_minus_t_dif, mask=mask)


# ---------------------------------------------------------------------------
# T4-optimized combined kernel with BLOCK-DIAGONAL v2 (split-path) weights.
#
# Single launch: both direct and diffuse MLPs.  Input data (mass/ke/w_eff/mu)
# loaded ONCE and shared.  Each MLP uses the BD v2 split-path approach
# (two independent [BLOCK_B, 16] streams, zero reshape/trans/split/join).
# tau_pad kept live across both MLPs to avoid reloading.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 128}, num_warps=2),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
        triton.Config({'BLOCK_B': 512}, num_warps=8),
        # maxnreg-capped variants (register pressure is higher here)
        triton.Config({'BLOCK_B': 128}, num_warps=4, maxnreg=128),
        triton.Config({'BLOCK_B': 256}, num_warps=4, maxnreg=128),
    ],
    key=['B_total'],
)
@triton.jit
def _optical_combined_T4_bd_v2_kernel(
    Mass_ptr, Ke_ptr, WeffFilter_ptr, Mu_ptr,
    # Direct MLP weights (all BD-packed [2,16,16])
    W0d_BD_ptr, B0d_ptr,
    WH0d_BD_ptr, BH0d_ptr, WH1d_BD_ptr, BH1d_ptr, WOd_BD_ptr, BOd_ptr,
    Wsel_dir_ptr,
    # Diffuse MLP weights (all BD-packed [2,16,16])
    W0f_BD_ptr, B0f_ptr,
    WH0f_BD_ptr, BH0f_ptr, WH1f_BD_ptr, BH1f_ptr, WOf_BD_ptr, BOf_ptr,
    Wsel_dif_ptr,
    # Outputs
    Tfull_dir_ptr, Et_dir_ptr, Er_dir_ptr, Ea_dir_ptr,
    Tfull_dif_ptr, Et_dif_ptr, Er_dif_ptr, Ea_dif_ptr,
    inv_mu_diffuse,
    B_total,
    n_channels: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask = offs < B_total

    b_idx = offs // n_channels
    c_idx = offs %  n_channels

    c4  = tl.arange(0, 4)
    c16 = tl.arange(0, 16)

    # ---- Load inputs ONCE (shared between direct and diffuse) ----
    mu = tl.load(Mu_ptr + b_idx, mask=mask, other=1.0)
    inv_mu = 1.0 / (mu + 1.0e-7)

    mass_pad = tl.load(Mass_ptr + b_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    W_ef_pad = tl.load(WeffFilter_ptr + c_idx[:, None] * 8 + c16[None, :],
                       mask=mask[:, None] & (c16[None, :] < 8), other=0.0)
    ke_offsets = tl.where(c16 >= 2, c16 - 2, 0)
    ke_offsets = tl.where(c16 < 8, ke_offsets, 0)
    ke_load_mask = mask[:, None] & (c16[None, :] >= 2) & (c16[None, :] < 8)
    ke_pad = tl.load(Ke_ptr + b_idx[:, None] * 6 + ke_offsets[None, :],
                     mask=ke_load_mask, other=1.0)

    tau_pad = W_ef_pad * mass_pad * ke_pad
    tau_sum = tl.sum(tau_pad, axis=1)

    t_full_direct  = tl.exp(-tau_sum * inv_mu)
    t_full_diffuse = tl.exp(-tau_sum * inv_mu_diffuse)
    one_minus_t_dir = 1.0 - t_full_direct
    one_minus_t_dif = 1.0 - t_full_diffuse

    # =====================================================================
    # ---------------------- DIRECT branch --------------------------------
    # =====================================================================
    x_d = tau_pad * inv_mu[:, None]
    x_d = tl.where(c16[None, :] == 8, mu[:, None], x_d)

    # Layer 0 (FP32, split into L/R streams)
    w0_L = tl.load(W0d_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    w0_R = tl.load(W0d_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    b0_L = tl.load(B0d_ptr + c16)
    b0_R = tl.load(B0d_ptr + c16 + 16)
    h_L = tl.dot(x_d, w0_L, input_precision="ieee") + b0_L[None, :]
    h_R = tl.dot(x_d, w0_R, input_precision="ieee") + b0_R[None, :]
    h_L = tl.maximum(h_L, 0.0)
    h_R = tl.maximum(h_R, 0.0)

    # Hidden 0 (FP16 TC, block-diagonal)
    wh0_L = tl.load(WH0d_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    wh0_R = tl.load(WH0d_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    bh0_L = tl.load(BH0d_ptr + c16)
    bh0_R = tl.load(BH0d_ptr + c16 + 16)
    h_L = tl.dot(tl.minimum(h_L, 65504.0).to(tl.float16), wh0_L, out_dtype=tl.float32) + bh0_L[None, :]
    h_R = tl.dot(tl.minimum(h_R, 65504.0).to(tl.float16), wh0_R, out_dtype=tl.float32) + bh0_R[None, :]
    h_L = tl.maximum(h_L, 0.0)
    h_R = tl.maximum(h_R, 0.0)

    # Hidden 1 (FP16 TC, block-diagonal)
    wh1_L = tl.load(WH1d_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    wh1_R = tl.load(WH1d_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    bh1_L = tl.load(BH1d_ptr + c16)
    bh1_R = tl.load(BH1d_ptr + c16 + 16)
    h_L = tl.dot(tl.minimum(h_L, 65504.0).to(tl.float16), wh1_L, out_dtype=tl.float32) + bh1_L[None, :]
    h_R = tl.dot(tl.minimum(h_R, 65504.0).to(tl.float16), wh1_R, out_dtype=tl.float32) + bh1_R[None, :]
    h_L = tl.maximum(h_L, 0.0)
    h_R = tl.maximum(h_R, 0.0)

    # Output (FP16 TC, block-diagonal)
    wo_L = tl.load(WOd_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    wo_R = tl.load(WOd_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    bo_L = tl.load(BOd_ptr + c16)
    bo_R = tl.load(BOd_ptr + c16 + 16)
    out_L = tl.dot(tl.minimum(h_L, 65504.0).to(tl.float16), wo_L, out_dtype=tl.float32) + bo_L[None, :]
    out_R = tl.dot(tl.minimum(h_R, 65504.0).to(tl.float16), wo_R, out_dtype=tl.float32) + bo_R[None, :]

    # Per-group softmax (4 groups of 4 per half)
    out_gL = tl.reshape(out_L, [BLOCK_B * 4, 4])
    m = tl.max(out_gL, axis=1)[:, None]
    out_gL = tl.exp(out_gL - m)
    s = tl.sum(out_gL, axis=1)[:, None]
    out_gL = out_gL / s
    out_basis_L = tl.reshape(out_gL, [BLOCK_B, 4, 4])

    out_gR = tl.reshape(out_R, [BLOCK_B * 4, 4])
    m = tl.max(out_gR, axis=1)[:, None]
    out_gR = tl.exp(out_gR - m)
    s = tl.sum(out_gR, axis=1)[:, None]
    out_gR = out_gR / s
    out_basis_R = tl.reshape(out_gR, [BLOCK_B, 4, 4])

    # wsel combination: merge left + right
    wsel_L = tl.load(Wsel_dir_ptr + c_idx[:, None] * 8 + c4[None, :],
                     mask=mask[:, None], other=0.0)
    wsel_R = tl.load(Wsel_dir_ptr + c_idx[:, None] * 8 + (c4[None, :] + 4),
                     mask=mask[:, None], other=0.0)
    combined = (tl.sum(out_basis_L * wsel_L[:, :, None], axis=1) +
                tl.sum(out_basis_R * wsel_R[:, :, None], axis=1))

    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    tl.store(Tfull_dir_ptr + offs, t_full_direct, mask=mask)
    tl.store(Et_dir_ptr + offs, e_t_e * inv_s2 * one_minus_t_dir, mask=mask)
    tl.store(Er_dir_ptr + offs, e_r_e * inv_s2 * one_minus_t_dir, mask=mask)
    tl.store(Ea_dir_ptr + offs, e_a_e * inv_s2 * one_minus_t_dir, mask=mask)

    # =====================================================================
    # ---------------------- DIFFUSE branch --------------------------------
    # =====================================================================
    # tau_pad is reused from above (no mu scaling, no mu column for diffuse)
    w0_L = tl.load(W0f_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    w0_R = tl.load(W0f_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    b0_L = tl.load(B0f_ptr + c16)
    b0_R = tl.load(B0f_ptr + c16 + 16)
    h_L = tl.dot(tau_pad, w0_L, input_precision="ieee") + b0_L[None, :]
    h_R = tl.dot(tau_pad, w0_R, input_precision="ieee") + b0_R[None, :]
    h_L = tl.maximum(h_L, 0.0)
    h_R = tl.maximum(h_R, 0.0)

    wh0_L = tl.load(WH0f_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    wh0_R = tl.load(WH0f_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    bh0_L = tl.load(BH0f_ptr + c16)
    bh0_R = tl.load(BH0f_ptr + c16 + 16)
    h_L = tl.dot(tl.minimum(h_L, 65504.0).to(tl.float16), wh0_L, out_dtype=tl.float32) + bh0_L[None, :]
    h_R = tl.dot(tl.minimum(h_R, 65504.0).to(tl.float16), wh0_R, out_dtype=tl.float32) + bh0_R[None, :]
    h_L = tl.maximum(h_L, 0.0)
    h_R = tl.maximum(h_R, 0.0)

    wh1_L = tl.load(WH1f_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    wh1_R = tl.load(WH1f_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    bh1_L = tl.load(BH1f_ptr + c16)
    bh1_R = tl.load(BH1f_ptr + c16 + 16)
    h_L = tl.dot(tl.minimum(h_L, 65504.0).to(tl.float16), wh1_L, out_dtype=tl.float32) + bh1_L[None, :]
    h_R = tl.dot(tl.minimum(h_R, 65504.0).to(tl.float16), wh1_R, out_dtype=tl.float32) + bh1_R[None, :]
    h_L = tl.maximum(h_L, 0.0)
    h_R = tl.maximum(h_R, 0.0)

    wo_L = tl.load(WOf_BD_ptr + 0 * 256 + c16[:, None] * 16 + c16[None, :])
    wo_R = tl.load(WOf_BD_ptr + 1 * 256 + c16[:, None] * 16 + c16[None, :])
    bo_L = tl.load(BOf_ptr + c16)
    bo_R = tl.load(BOf_ptr + c16 + 16)
    out_L = tl.dot(tl.minimum(h_L, 65504.0).to(tl.float16), wo_L, out_dtype=tl.float32) + bo_L[None, :]
    out_R = tl.dot(tl.minimum(h_R, 65504.0).to(tl.float16), wo_R, out_dtype=tl.float32) + bo_R[None, :]

    out_gL = tl.reshape(out_L, [BLOCK_B * 4, 4])
    m = tl.max(out_gL, axis=1)[:, None]
    out_gL = tl.exp(out_gL - m)
    s = tl.sum(out_gL, axis=1)[:, None]
    out_gL = out_gL / s
    out_basis_L = tl.reshape(out_gL, [BLOCK_B, 4, 4])

    out_gR = tl.reshape(out_R, [BLOCK_B * 4, 4])
    m = tl.max(out_gR, axis=1)[:, None]
    out_gR = tl.exp(out_gR - m)
    s = tl.sum(out_gR, axis=1)[:, None]
    out_gR = out_gR / s
    out_basis_R = tl.reshape(out_gR, [BLOCK_B, 4, 4])

    wsel_L = tl.load(Wsel_dif_ptr + c_idx[:, None] * 8 + c4[None, :],
                     mask=mask[:, None], other=0.0)
    wsel_R = tl.load(Wsel_dif_ptr + c_idx[:, None] * 8 + (c4[None, :] + 4),
                     mask=mask[:, None], other=0.0)
    combined = (tl.sum(out_basis_L * wsel_L[:, :, None], axis=1) +
                tl.sum(out_basis_R * wsel_R[:, :, None], axis=1))

    combined_22 = tl.reshape(combined, [BLOCK_B, 2, 2])
    T_even, T_odd = tl.split(combined_22)
    v_t, v_a = tl.split(T_even)
    v_r, _v_dead = tl.split(T_odd)

    mx = tl.maximum(tl.maximum(v_t, v_r), v_a)
    e_t_e = tl.exp(v_t - mx)
    e_r_e = tl.exp(v_r - mx)
    e_a_e = tl.exp(v_a - mx)
    inv_s2 = 1.0 / (e_t_e + e_r_e + e_a_e)

    tl.store(Tfull_dif_ptr + offs, t_full_diffuse, mask=mask)
    tl.store(Et_dif_ptr + offs, e_t_e * inv_s2 * one_minus_t_dif, mask=mask)
    tl.store(Er_dif_ptr + offs, e_r_e * inv_s2 * one_minus_t_dif, mask=mask)
    tl.store(Ea_dif_ptr + offs, e_a_e * inv_s2 * one_minus_t_dif, mask=mask)


# ---------------------------------------------------------------------------
# Python launchers
# ---------------------------------------------------------------------------

def launch_direct(tau_flat, mu_dir_flat, mlp_module, wsel_direct,
                  t_full, e_t, e_r, e_a, n_channels):
    """
    tau_flat:       [B, n_channels, 8] contiguous, where B = n_samples*n_layers
    mu_dir_flat:    [B] contiguous (per sample-layer cosine zenith angle)
    mlp_module:     MultipleMLPs_I9_Triton (already reconfigured)
    wsel_direct:    [n_channels, 8] contiguous flat selection weights
    Outputs (preallocated, [B, n_channels] contiguous):
        t_full, e_t, e_r, e_a
    """
    B_total = tau_flat.shape[0] * n_channels
    grid = lambda meta: (triton.cdiv(B_total, meta['BLOCK_B']),)
    _fused_scattering_direct_kernel[grid](
        tau_flat, mu_dir_flat,
        mlp_module.w0, mlp_module.b0,
        mlp_module.wh0, mlp_module.bh0,
        mlp_module.wh1, mlp_module.bh1,
        mlp_module.wo,  mlp_module.bo,
        wsel_direct,
        t_full, e_t, e_r, e_a,
        B_total,
        n_channels=n_channels,
    )


def launch_diffuse(tau_flat, inv_mu_diffuse_scalar, mlp_module, wsel_diffuse,
                   t_full, e_t, e_r, e_a, n_channels):
    """
    inv_mu_diffuse_scalar: Python float = 1 / (mu_diffuse + eps)
    """
    B_total = tau_flat.shape[0] * n_channels
    grid = lambda meta: (triton.cdiv(B_total, meta['BLOCK_B']),)
    _fused_scattering_diffuse_kernel[grid](
        tau_flat,
        mlp_module.w0, mlp_module.b0,
        mlp_module.wh0, mlp_module.bh0,
        mlp_module.wh1, mlp_module.bh1,
        mlp_module.wo,  mlp_module.bo,
        wsel_diffuse,
        t_full, e_t, e_r, e_a,
        inv_mu_diffuse_scalar,
        B_total,
        n_channels=n_channels,
    )


def launch_ke_mlps(tp_flat, ke_mlps_list, ke_output):
    """Launch the fused ke MLPs kernel.

    tp_flat        : [B, 2] contiguous (temperature, pressure)
    ke_mlps_list   : list of 6 nn.Module objects (net_ke_h2o, net_ke_o3, ...)
    ke_output      : [B, 6] contiguous output tensor (pre-allocated)
    """
    B = tp_flat.shape[0]
    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']),)

    # Extract weight and bias pointers from the 6 MLPs.
    # Each MLP has 4 layers: 2->6, 6->4, 4->4, 4->1
    def _get_w_b(mlp):
        return (mlp.hidden[0].weight, mlp.hidden[0].bias,
                mlp.hidden[1].weight, mlp.hidden[1].bias,
                mlp.hidden[2].weight, mlp.hidden[2].bias,
                mlp.output.weight, mlp.output.bias)

    w_b_tuples = [_get_w_b(mlp) for mlp in ke_mlps_list]
    flat_w_b = []
    for tup in w_b_tuples:
        flat_w_b.extend(tup)

    _fused_ke_mlps_kernel[grid](
        tp_flat, ke_output,
        *flat_w_b,
        B,
    )

# Fused optical depth and scattering in single kernel
# Does not take advantage of block diagonal structure
def launch_optical_scattering_fke(
        mass_flat, tp_flat, w_eff_filter, mu_dir_flat, inv_mu_diffuse_scalar,
        ke_weights,
        direct_mlp, diffuse_mlp,
        wsel_direct, wsel_diffuse,
        t_full_direct, e_t_direct, e_r_direct, e_a_direct,
        t_full_diffuse, e_t_diffuse, e_r_diffuse, e_a_diffuse,
        n_channels):
    """T4-optimized: fused ke+scattering.

    Computes ke MLPs inline from temperature/pressure (no separate ke
    kernel launch, no ke DRAM round-trip).  Two kernel launches
    (direct + diffuse) with pre-stored FP16 scattering weights.
    """
    B = mass_flat.shape[0]
    B_total = B * n_channels
    grid = lambda meta: (triton.cdiv(B_total, meta['BLOCK_B']),)

    _optical_split_direct_T4_fke_kernel[grid](
        mass_flat, tp_flat, w_eff_filter, mu_dir_flat,
        ke_weights,
        direct_mlp.w0,      direct_mlp.b0,
        direct_mlp.wh0_f16, direct_mlp.bh0,
        direct_mlp.wh1_f16, direct_mlp.bh1,
        direct_mlp.wo_f16,  direct_mlp.bo,
        wsel_direct,
        t_full_direct, e_t_direct, e_r_direct, e_a_direct,
        B_total,
        n_channels=n_channels,
    )

    _optical_split_diffuse_T4_fke_kernel[grid](
        mass_flat, tp_flat, w_eff_filter,
        ke_weights,
        diffuse_mlp.w0,      diffuse_mlp.b0,
        diffuse_mlp.wh0_f16, diffuse_mlp.bh0,
        diffuse_mlp.wh1_f16, diffuse_mlp.bh1,
        diffuse_mlp.wo_f16,  diffuse_mlp.bo,
        wsel_diffuse,
        t_full_diffuse, e_t_diffuse, e_r_diffuse, e_a_diffuse,
        inv_mu_diffuse_scalar,
        B_total,
        n_channels=n_channels,
    )

# Does not take advantage of block diagonal matrix multiply
def launch_optical_scattering(
        mass_flat, ke_flat, w_eff_filter, mu_dir_flat, inv_mu_diffuse_scalar,
        direct_mlp, diffuse_mlp,
        wsel_direct, wsel_diffuse,
        t_full_direct, e_t_direct, e_r_direct, e_a_direct,
        t_full_diffuse, e_t_diffuse, e_r_diffuse, e_a_diffuse,
        n_channels):
    """T4-optimized: split kernels with pre-stored FP16 weights.

    Two kernel launches (direct + diffuse), each computing tau inline.
    Hidden/output weights are pre-stored as FP16 (no in-kernel conversion).
    """
    B = mass_flat.shape[0]
    B_total = B * n_channels
    grid = lambda meta: (triton.cdiv(B_total, meta['BLOCK_B']),)

    _optical_split_direct_T4_kernel[grid](
        mass_flat, ke_flat, w_eff_filter, mu_dir_flat,
        direct_mlp.w0,      direct_mlp.b0,
        direct_mlp.wh0_f16, direct_mlp.bh0,
        direct_mlp.wh1_f16, direct_mlp.bh1,
        direct_mlp.wo_f16,  direct_mlp.bo,
        wsel_direct,
        t_full_direct, e_t_direct, e_r_direct, e_a_direct,
        B_total,
        n_channels=n_channels,
    )

    _optical_split_diffuse_T4_kernel[grid](
        mass_flat, ke_flat, w_eff_filter,
        diffuse_mlp.w0,      diffuse_mlp.b0,
        diffuse_mlp.wh0_f16, diffuse_mlp.bh0,
        diffuse_mlp.wh1_f16, diffuse_mlp.bh1,
        diffuse_mlp.wo_f16,  diffuse_mlp.bo,
        wsel_diffuse,
        t_full_diffuse, e_t_diffuse, e_r_diffuse, e_a_diffuse,
        inv_mu_diffuse_scalar,
        B_total,
        n_channels=n_channels,
    )


def launch_optical_scattering_combined(
        mass_flat, ke_flat, w_eff_filter, mu_dir_flat, inv_mu_diffuse_scalar,
        direct_mlp, diffuse_mlp,
        wsel_direct, wsel_diffuse,
        t_full_direct, e_t_direct, e_r_direct, e_a_direct,
        t_full_diffuse, e_t_diffuse, e_r_diffuse, e_a_diffuse,
        n_channels):
    """T4-optimized combined direct+diffuse kernel.

    Single kernel launch.  mass/ke/w_eff/mu are read once per row and
    tau_sum is computed once, reused for both t_full_direct and
    t_full_diffuse.  Both MLPs run in the same block while the shared
    inputs are L1-hot.  Pre-stored FP16 hidden/output weights (no
    in-kernel conversion).
    """
    B = mass_flat.shape[0]
    B_total = B * n_channels
    grid = lambda meta: (triton.cdiv(B_total, meta['BLOCK_B']),)

    _optical_combined_T4_kernel[grid](
        mass_flat, ke_flat, w_eff_filter, mu_dir_flat,
        direct_mlp.w0,       direct_mlp.b0,
        direct_mlp.wh0_f16,  direct_mlp.bh0,
        direct_mlp.wh1_f16,  direct_mlp.bh1,
        direct_mlp.wo_f16,   direct_mlp.bo,
        wsel_direct,
        diffuse_mlp.w0,      diffuse_mlp.b0,
        diffuse_mlp.wh0_f16, diffuse_mlp.bh0,
        diffuse_mlp.wh1_f16, diffuse_mlp.bh1,
        diffuse_mlp.wo_f16,  diffuse_mlp.bo,
        wsel_diffuse,
        t_full_direct,  e_t_direct,  e_r_direct,  e_a_direct,
        t_full_diffuse, e_t_diffuse, e_r_diffuse, e_a_diffuse,
        inv_mu_diffuse_scalar,
        B_total,
        n_channels=n_channels,
    )

# Does unnecessary split, transpose, and join operations in handling
# block diagonal matrix
def launch_optical_scattering_bd(
        mass_flat, ke_flat, w_eff_filter, mu_dir_flat, inv_mu_diffuse_scalar,
        direct_mlp, diffuse_mlp,
        wsel_direct, wsel_diffuse,
        t_full_direct, e_t_direct, e_r_direct, e_a_direct,
        t_full_diffuse, e_t_diffuse, e_r_diffuse, e_a_diffuse,
        n_channels):
    """Split kernels with BLOCK-DIAGONAL hidden-layer weights.

    Hidden weight matrices are pre-packed as [4, 8, 8] FP16.  Output layer
    remains dense (FP16).  Expects the MLPs to have `wh0_f16_bd`,
    `wh1_f16_bd` buffers registered (see ScatteringFused.__init__).
    """
    B = mass_flat.shape[0]
    B_total = B * n_channels
    grid = lambda meta: (triton.cdiv(B_total, meta['BLOCK_B']),)

    _optical_split_direct_T4_bd_kernel[grid](
        mass_flat, ke_flat, w_eff_filter, mu_dir_flat,
        direct_mlp.w0,         direct_mlp.b0,
        direct_mlp.wh0_f16_bd, direct_mlp.bh0,
        direct_mlp.wh1_f16_bd, direct_mlp.bh1,
        direct_mlp.wo_f16,     direct_mlp.bo,
        wsel_direct,
        t_full_direct, e_t_direct, e_r_direct, e_a_direct,
        B_total,
        n_channels=n_channels,
    )

    _optical_split_diffuse_T4_bd_kernel[grid](
        mass_flat, ke_flat, w_eff_filter,
        diffuse_mlp.w0,         diffuse_mlp.b0,
        diffuse_mlp.wh0_f16_bd, diffuse_mlp.bh0,
        diffuse_mlp.wh1_f16_bd, diffuse_mlp.bh1,
        diffuse_mlp.wo_f16,     diffuse_mlp.bo,
        wsel_diffuse,
        t_full_diffuse, e_t_diffuse, e_r_diffuse, e_a_diffuse,
        inv_mu_diffuse_scalar,
        B_total,
        n_channels=n_channels,
    )

# Splits direct and diffuse in separate kernels
def launch_optical_scattering_bd_v2(
        mass_flat, ke_flat, w_eff_filter, mu_dir_flat, inv_mu_diffuse_scalar,
        direct_mlp, diffuse_mlp,
        wsel_direct, wsel_diffuse,
        t_full_direct, e_t_direct, e_r_direct, e_a_direct,
        t_full_diffuse, e_t_diffuse, e_r_diffuse, e_a_diffuse,
        n_channels):
    """Split-path block-diagonal kernels (v2).

    All four weight layers use pre-packed [2, 16, 16] buffers.  Two
    independent [BLOCK_B, 16] streams flow from input to output with
    zero reshape/trans/split/join overhead.  Expects the MLPs to have
    `w0_bd`, `wh0_f16_bd`, `wh1_f16_bd`, `wo_f16_bd` buffers registered.
    """
    B = mass_flat.shape[0]
    B_total = B * n_channels
    grid = lambda meta: (triton.cdiv(B_total, meta['BLOCK_B']),)

    _optical_split_direct_T4_bd_v2_kernel[grid](
        mass_flat, ke_flat, w_eff_filter, mu_dir_flat,
        direct_mlp.w0_bd,      direct_mlp.b0,
        direct_mlp.wh0_f16_bd, direct_mlp.bh0,
        direct_mlp.wh1_f16_bd, direct_mlp.bh1,
        direct_mlp.wo_f16_bd,  direct_mlp.bo,
        wsel_direct,
        t_full_direct, e_t_direct, e_r_direct, e_a_direct,
        B_total,
        n_channels=n_channels,
    )

    _optical_split_diffuse_T4_bd_v2_kernel[grid](
        mass_flat, ke_flat, w_eff_filter,
        diffuse_mlp.w0_bd,      diffuse_mlp.b0,
        diffuse_mlp.wh0_f16_bd, diffuse_mlp.bh0,
        diffuse_mlp.wh1_f16_bd, diffuse_mlp.bh1,
        diffuse_mlp.wo_f16_bd,  diffuse_mlp.bo,
        wsel_diffuse,
        t_full_diffuse, e_t_diffuse, e_r_diffuse, e_a_diffuse,
        inv_mu_diffuse_scalar,
        B_total,
        n_channels=n_channels,
    )

# In addition to direct and diffuse split, splits each into separate
# kernels for the blocks in the block diagonal matrix.
# Gives a total for 4 kernels
def launch_optical_scattering_bd_v2_split(
        mass_flat, ke_flat, w_eff_filter, mu_dir_flat, inv_mu_diffuse_scalar,
        direct_mlp, diffuse_mlp,
        wsel_direct, wsel_diffuse,
        t_full_direct, e_t_direct, e_r_direct, e_a_direct,
        t_full_diffuse, e_t_diffuse, e_r_diffuse, e_a_diffuse,
        n_channels):
    """Split-serial block-diagonal v2: left then right, per direction.

    Each sub-kernel carries only one [BLOCK_B, 16] stream (half the
    register pressure).  The left kernel stores t_full and a partial
    wsel combination [B_total, 4]; the right kernel reads the partial,
    adds its own, and writes final Et/Er/Ea.  Input data is loaded in
    both sub-kernels (duplicated).
    """
    B = mass_flat.shape[0]
    B_total = B * n_channels
    grid = lambda meta: (triton.cdiv(B_total, meta['BLOCK_B']),)

    # Temporary buffer for partial wsel combination [B_total, 4]
    partial_buf = torch.empty(B_total, 4, device=mass_flat.device,
                              dtype=torch.float32)

    # ---- Direct: left then right ----
    _optical_direct_T4_bd_v2_left_kernel[grid](
        mass_flat, ke_flat, w_eff_filter, mu_dir_flat,
        direct_mlp.w0_bd,      direct_mlp.b0,
        direct_mlp.wh0_f16_bd, direct_mlp.bh0,
        direct_mlp.wh1_f16_bd, direct_mlp.bh1,
        direct_mlp.wo_f16_bd,  direct_mlp.bo,
        wsel_direct,
        t_full_direct, partial_buf,
        B_total, n_channels=n_channels,
    )
    _optical_direct_T4_bd_v2_right_kernel[grid](
        mass_flat, ke_flat, w_eff_filter, mu_dir_flat,
        direct_mlp.w0_bd,      direct_mlp.b0,
        direct_mlp.wh0_f16_bd, direct_mlp.bh0,
        direct_mlp.wh1_f16_bd, direct_mlp.bh1,
        direct_mlp.wo_f16_bd,  direct_mlp.bo,
        wsel_direct,
        partial_buf,
        e_t_direct, e_r_direct, e_a_direct,
        B_total, n_channels=n_channels,
    )

    # ---- Diffuse: left then right (reuse partial_buf) ----
    _optical_diffuse_T4_bd_v2_left_kernel[grid](
        mass_flat, ke_flat, w_eff_filter,
        diffuse_mlp.w0_bd,      diffuse_mlp.b0,
        diffuse_mlp.wh0_f16_bd, diffuse_mlp.bh0,
        diffuse_mlp.wh1_f16_bd, diffuse_mlp.bh1,
        diffuse_mlp.wo_f16_bd,  diffuse_mlp.bo,
        wsel_diffuse,
        t_full_diffuse, partial_buf,
        inv_mu_diffuse_scalar,
        B_total, n_channels=n_channels,
    )
    _optical_diffuse_T4_bd_v2_right_kernel[grid](
        mass_flat, ke_flat, w_eff_filter,
        diffuse_mlp.w0_bd,      diffuse_mlp.b0,
        diffuse_mlp.wh0_f16_bd, diffuse_mlp.bh0,
        diffuse_mlp.wh1_f16_bd, diffuse_mlp.bh1,
        diffuse_mlp.wo_f16_bd,  diffuse_mlp.bo,
        wsel_diffuse,
        partial_buf,
        e_t_diffuse, e_r_diffuse, e_a_diffuse,
        inv_mu_diffuse_scalar,
        B_total, n_channels=n_channels,
    )

# Puts all computation (direct and diffuse, left and right blocks)
# together in a single kernels. Avoids some duplicated computation.
# However, isn't faster for T4 GPU probably because of register
# spilling and oversubscribed L1 Cache. Needs testing on other
# GPUs
def launch_optical_scattering_combined_bd_v2(
        mass_flat, ke_flat, w_eff_filter, mu_dir_flat, inv_mu_diffuse_scalar,
        direct_mlp, diffuse_mlp,
        wsel_direct, wsel_diffuse,
        t_full_direct, e_t_direct, e_r_direct, e_a_direct,
        t_full_diffuse, e_t_diffuse, e_r_diffuse, e_a_diffuse,
        n_channels):
    """Combined direct+diffuse BD v2 kernel — single launch.

    Reads mass/ke/w_eff/mu ONCE, runs both MLPs using BD v2 split-path
    (two [BLOCK_B, 16] streams per MLP), writes all 8 output tensors.
    """
    B = mass_flat.shape[0]
    B_total = B * n_channels
    grid = lambda meta: (triton.cdiv(B_total, meta['BLOCK_B']),)

    _optical_combined_T4_bd_v2_kernel[grid](
        mass_flat, ke_flat, w_eff_filter, mu_dir_flat,
        # Direct weights
        direct_mlp.w0_bd,      direct_mlp.b0,
        direct_mlp.wh0_f16_bd, direct_mlp.bh0,
        direct_mlp.wh1_f16_bd, direct_mlp.bh1,
        direct_mlp.wo_f16_bd,  direct_mlp.bo,
        wsel_direct,
        # Diffuse weights
        diffuse_mlp.w0_bd,      diffuse_mlp.b0,
        diffuse_mlp.wh0_f16_bd, diffuse_mlp.bh0,
        diffuse_mlp.wh1_f16_bd, diffuse_mlp.bh1,
        diffuse_mlp.wo_f16_bd,  diffuse_mlp.bo,
        wsel_diffuse,
        # Outputs
        t_full_direct,  e_t_direct,  e_r_direct,  e_a_direct,
        t_full_diffuse, e_t_diffuse, e_r_diffuse, e_a_diffuse,
        inv_mu_diffuse_scalar,
        B_total,
        n_channels=n_channels,
    )


def launch_both(tau_flat, mu_dir_flat, inv_mu_diffuse_scalar,
                direct_mlp, diffuse_mlp,
                wsel_direct, wsel_diffuse,
                t_full_direct, e_t_direct, e_r_direct, e_a_direct,
                t_full_diffuse, e_t_diffuse, e_r_diffuse, e_a_diffuse,
                n_channels):
    """Single-launch combined direct + diffuse fused scattering kernel.

    Reads `tau_flat` exactly once per row and writes all 8 output
    tensors.  Both MLPs share `tau_sum` for their respective
    `t_full = exp(-tau_sum / mu)` calculations.
    """
    B_total = tau_flat.shape[0] * n_channels
    grid = lambda meta: (triton.cdiv(B_total, meta['BLOCK_B']),)
    _fused_scattering_both_kernel[grid](
        tau_flat, mu_dir_flat,
        # Direct MLP weights
        direct_mlp.w0,  direct_mlp.b0,
        direct_mlp.wh0, direct_mlp.bh0,
        direct_mlp.wh1, direct_mlp.bh1,
        direct_mlp.wo,  direct_mlp.bo,
        wsel_direct,
        # Diffuse MLP weights
        diffuse_mlp.w0,  diffuse_mlp.b0,
        diffuse_mlp.wh0, diffuse_mlp.bh0,
        diffuse_mlp.wh1, diffuse_mlp.bh1,
        diffuse_mlp.wo,  diffuse_mlp.bo,
        wsel_diffuse,
        # Direct outputs
        t_full_direct, e_t_direct, e_r_direct, e_a_direct,
        # Diffuse outputs
        t_full_diffuse, e_t_diffuse, e_r_diffuse, e_a_diffuse,
        inv_mu_diffuse_scalar,
        B_total,
        n_channels=n_channels,
    )
