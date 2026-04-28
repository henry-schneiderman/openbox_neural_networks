
import numpy as np
import sys
#from numba import jit, cuda
import torch
from torch import nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function

#import torch_xla
#import torch_xla.core.xla_model as xm
import time
from typing import List
import os
import triton
import triton.language as tl

import evaluate_network_fast

os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
#os.environ["TORCH_LOGS"] = "+dynamic"
#print(f"torch version: {torch.__version__}")
#print(f"torch_xla version: {torch_xla.__version__}")

print(f"Python Version: {sys.version}", flush=True)


@triton.jit
def _mul_rn(a, b):
    """Multiply with explicit round-to-nearest, preventing FMA fusion.
    PTX mul.rn.f32 cannot be fused into FMA by the assembler."""
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=r,r,r", args=[a, b], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def _rcp_rn(b):
    """IEEE-compliant reciprocal (div.full.f32 1.0, b)."""
    return tl.inline_asm_elementwise(
        "rcp.rn.f32 $0, $1;",
        "=r,r", args=[b], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def _div_rn(a, b):
    """IEEE-compliant division (div.rn.f32)."""
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=r,r,r", args=[a, b], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def _multireflection_kernel(
    # 8 input layer tensors, each (n_sample, n_layers, n_channels), contiguous
    td_ptr, tf_ptr, etd_ptr, erd_ptr, ead_ptr, etf_ptr, erf_ptr, eaf_ptr,
    # surface tensor (n_sample, 2), contiguous
    surf_ptr,
    # 6 output tensors, each (n_sample, n_layers, n_channels)
    o_tmd_ptr, o_tmf_ptr, o_rsmd_ptr, o_rsmf_ptr, o_almd_ptr, o_almf_ptr,
    # TOA reflection output (n_sample, n_channels)
    o_toa_ptr,
    n_layers: tl.constexpr,
    n_channels: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fused adding-doubling kernel: one program per sample, loops over all layers."""
    sid = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    cmask = c < n_channels

    lc = n_layers * n_channels
    sbase = sid * lc

    # Load surface albedo and broadcast to channel vector
    r_s = tl.load(surf_ptr + sid * 2 + 1).to(tl.float32)
    zeros = tl.zeros((BLOCK_C,), dtype=tl.float32)
    rs_dir = zeros + r_s
    rs_dif = zeros + r_s
    as_dir = zeros + (1.0 - r_s)
    as_dif = zeros + (1.0 - r_s)

    rlm_dir = zeros  # set every iteration; init for compiler

    # Process layers from surface (bottom) to TOA (top)
    for j in range(n_layers):
        i = n_layers - 1 - j
        off = sbase + i * n_channels + c

        # Load 8 layer radiative properties
        t_dir = tl.load(td_ptr + off, mask=cmask, other=0.0)
        t_dif = tl.load(tf_ptr + off, mask=cmask, other=0.0)
        et_dir = tl.load(etd_ptr + off, mask=cmask, other=0.0)
        er_dir = tl.load(erd_ptr + off, mask=cmask, other=0.0)
        ea_dir = tl.load(ead_ptr + off, mask=cmask, other=0.0)
        et_dif = tl.load(etf_ptr + off, mask=cmask, other=0.0)
        er_dif = tl.load(erf_ptr + off, mask=cmask, other=0.0)
        ea_dif = tl.load(eaf_ptr + off, mask=cmask, other=0.0)

        # Adding-doubling denominator
        # Triton's compiler fuses a*b+c into FMA, computing a*b at
        # extended precision before the add — a differently-rounded
        # result than PyTorch's separate multiply-then-add. These
        # ~1 ULP differences compound through the carry-forward
        # state across 60 iterations, amplified by d=1/denom. Fix:
        # use _mul_rn (PTX mul.rn.f32) for the last multiply before
        # every addition so both add operands are opaque to the
        # compiler's FMA fuser.
        denom = 1.0 - _mul_rn(er_dif, rs_dif) + 1.0e-06
        d = _rcp_rn(denom)

        # Direct beam
        tm_dir = _div_rn(_mul_rn(t_dir * rs_dir, er_dif) + et_dir, denom)
        rsm_dir = _mul_rn(t_dir * rs_dir, d) + _mul_rn(et_dir * rs_dif, d)
        alm_dir = ea_dir + _mul_rn(rsm_dir, ea_dif)
        rlm_dir = er_dir + _mul_rn(rsm_dir, t_dif + et_dif)
        asm_dir = _mul_rn(t_dir, as_dir) + _mul_rn(tm_dir, as_dif)

        # Diffuse beam
        tm_dif = _mul_rn(t_dif * rs_dif * er_dif, d) + _mul_rn(et_dif, d)
        rsm_dif = _mul_rn(t_dif * rs_dif, d) + _mul_rn(et_dif * rs_dif, d)
        alm_dif = ea_dif + _mul_rn(rsm_dif, ea_dif)
        rlm_dif = er_dif + _mul_rn(rsm_dif, t_dif + et_dif)
        asm_dif = _mul_rn(t_dif, as_dif) + _mul_rn(tm_dif, as_dif)

        # Store results directly at layer index i (top-to-bottom order, no flip needed)
        tl.store(o_tmd_ptr + off, tm_dir, mask=cmask)
        tl.store(o_tmf_ptr + off, tm_dif, mask=cmask)
        tl.store(o_rsmd_ptr + off, rsm_dir, mask=cmask)
        tl.store(o_rsmf_ptr + off, rsm_dif, mask=cmask)
        tl.store(o_almd_ptr + off, alm_dir, mask=cmask)
        tl.store(o_almf_ptr + off, alm_dif, mask=cmask)

        # Merge layer and virtual surface for next iteration
        rs_dir = rlm_dir
        rs_dif = rlm_dif
        as_dir = alm_dir + asm_dir
        as_dif = alm_dif + asm_dif

    # TOA upward reflection = r_layer_multi_direct of top layer (last iteration)
    toa_off = sid * n_channels + c
    tl.store(o_toa_ptr + toa_off, rlm_dir, mask=cmask)


class MultiReflectionFast(nn.Module):
    """
    Same as MultiReflection in training_network except contains
    some internal timing code, and assumes inputs are normalized
    such that
        e_t_direct + e_r_direct + e_a_direct + t_direct = 1.0
        e_t_diffuse + e_r_diffuse + e_a_diffuse + t_diffuse = 1.0

    Whereas the version in training_network.py assumes:
        e_r_direct + e_a_direct + t_direct = 1.0
        e_r_diffuse + e_a_diffuse + t_diffuse = 1.0

    ----------------------------------------------------
    Computes each layer's "multi-reflection coefficients" by accounting
    for multireflection with all other layers using the
    Adding-Doubling method (no learning).
    """

    def __init__(self, device):
        super(MultiReflectionFast, self).__init__()
        self.device = device

    def forward(self, x):
        """
        Traverses the atmospheric layers from the surface to the
        top of the atmosphere. At each layer computes "multi-reflection"
        coefficients modeling the effects of inter-reflection among
        the layers.

        Uses a fused Triton kernel that processes all layers in a single
        GPU kernel launch, eliminating per-layer kernel launch overhead
        and temporary tensor allocations.
        """

        radiative_layers, x_surface = x

        t_direct, t_diffuse, e_t_direct, e_r_direct, e_a_direct, \
                  e_t_diffuse, e_r_diffuse, e_a_diffuse = radiative_layers

        n_sample, n_layers, n_channels = t_direct.shape

        # Pre-allocate all output tensors
        t_multi_direct = torch.empty_like(t_direct)
        t_multi_diffuse = torch.empty_like(t_direct)
        r_surface_multi_direct = torch.empty_like(t_direct)
        r_surface_multi_diffuse = torch.empty_like(t_direct)
        a_layer_multi_direct = torch.empty_like(t_direct)
        a_layer_multi_diffuse = torch.empty_like(t_direct)
        upward_reflection_toa = torch.empty(
            (n_sample, n_channels), device=t_direct.device, dtype=torch.float32)

        BLOCK_C = triton.next_power_of_2(n_channels)

        _multireflection_kernel[(n_sample,)](
            t_direct, t_diffuse, e_t_direct, e_r_direct, e_a_direct,
            e_t_diffuse, e_r_diffuse, e_a_diffuse,
            x_surface,
            t_multi_direct, t_multi_diffuse,
            r_surface_multi_direct, r_surface_multi_diffuse,
            a_layer_multi_direct, a_layer_multi_diffuse,
            upward_reflection_toa,
            n_layers, n_channels, BLOCK_C=BLOCK_C,
            num_warps=2,
        )

        multireflected_layers = [t_direct, t_diffuse,
                                 t_multi_direct, t_multi_diffuse,
                                 r_surface_multi_direct, r_surface_multi_diffuse,
                                 a_layer_multi_direct, a_layer_multi_diffuse]

        return (multireflected_layers, upward_reflection_toa)


def evaluate_multireflection_mlp():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    n_channels = 42
    n_layers = 60
    n_sample = 4096
    n_batch = 50
    t_delta_1 = 0.0
    t_delta_2 = 0.0
    n_elements = 8


    original_mlp = evaluate_network_fast.MultiReflectionTiming(device=device)
    
    fast_mlp = MultiReflectionFast(device=device)

    print("Round 1")
    with torch.inference_mode():
        for _ in range(n_batch):
            x_surface = torch.rand((n_sample, 2), device=device, dtype=torch.float32)
            radiative_layers = [torch.rand((n_sample, n_layers, n_channels), device=device, dtype=torch.float32) for _ in range(n_elements)]
            torch.cuda.synchronize()
            t_start = time.time()
            output_original = original_mlp([ radiative_layers, x_surface])
            torch.cuda.synchronize()
            t_delta_1 += time.time() - t_start
            
            torch.cuda.synchronize()
            t_start = time.time()
            output_original = fast_mlp([ radiative_layers, x_surface])
            torch.cuda.synchronize()
            t_delta_2 += time.time() - t_start


    print(f" n_batch={n_batch}, N={n_sample * n_layers}, Total_N={n_batch * n_sample * n_layers}")
    print(f" Original Time = {t_delta_1:.4f} s")
    print(f" Fast Time = {t_delta_2:.4f} s")

    print("Round 2")
    t_delta_1 = 0.0
    t_delta_2 = 0.0
    with torch.inference_mode():
        for _ in range(n_batch):
            x_surface = torch.rand((n_sample, 2), device=device, dtype=torch.float32)
            radiative_layers = [torch.rand((n_sample, n_layers, n_channels), device=device, dtype=torch.float32) for _ in range(n_elements)]
            torch.cuda.synchronize()
            t_start = time.time()
            output_original = original_mlp([ radiative_layers, x_surface])
            torch.cuda.synchronize()
            t_delta_1 += time.time() - t_start
            torch.cuda.synchronize()
            t_start = time.time()
            output_fast = fast_mlp([ radiative_layers, x_surface])
            torch.cuda.synchronize()
            t_delta_2 += time.time() - t_start


    print(f" n_batch={n_batch}, N={n_sample * n_layers}, Total_N={n_batch * n_sample * n_layers}")
    print(f" Original Time = {t_delta_1:.4f} s")
    print(f" Fast Time = {t_delta_2:.4f} s")
    
    # Compare nested outputs for floating point accuracy
    original_layers, original_toa = output_original
    fast_layers, fast_toa = output_fast

    print("\nAccuracy comparison (last batch):")
    for i, (t_orig, t_fast) in enumerate(zip(original_layers, fast_layers)):
        max_err = (t_orig - t_fast).abs().max().item()
        print(f"  Layer tensor {i}: max_abs_err={max_err:.2e}")
        assert torch.allclose(t_orig, t_fast, rtol=1e-2, atol=1e-2), \
                        "Outputs differ!"

    toa_err = (original_toa - fast_toa).abs().max().item()
    print(f"  TOA reflection:  max_abs_err={toa_err:.2e}")
    
    assert torch.allclose(original_toa, fast_toa, rtol=1e-2, atol=1e-2), \
                        "Outputs differ!"


if __name__ == "__main__":
    evaluate_multireflection_mlp()

