"""
Fused Triton kernel for the shortwave Propagation module.

The original PyTorch implementation walks the 60 atmospheric layers in
a Python for-loop, launching ~10 small CUDA kernels per layer (broadcast
multiplies, adds and final stack/sum reductions).  Each launch is small
relative to the L4's launch latency, so the loop wastes most of its
time on launch overhead.

This kernel does all of the following in a single launch (one program
per atmospheric column):

  * Loop over all n_layers layers in registers, carrying flux_direct
    and flux_diffuse from one layer to the next without touching DRAM.
  * Per layer, compute the four physical quantities
        flux_absorbed
        flux_down_direct (post-layer)
        flux_down_diffuse (post-layer)
        flux_up_diffuse  (post-layer)
    each as a [n_channels] vector kept in registers.
  * Reduce each quantity across channels with tl.sum, producing a
    single scalar per layer per quantity.
  * Write the per-layer scalar directly to the output tensors, which
    are already in the [n_examples, n_layers(+1)] shape that the loss
    functions expect.  No intermediate per-channel tensors are written
    to global memory.

This eliminates ~600 small kernel launches per batch and avoids
allocating four [n_examples, n_layers+1, n_channels] intermediate
tensors that would otherwise need to be summed across channels.

Author: Henry Schneiderman, henry@pittdata.com
"""

import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _fused_propagation_kernel(
    # Inputs (per-channel, all [n_samples, n_channels] or [n_samples, n_layers, n_channels])
    Fdir_in_ptr, Fdif_in_ptr,            # [n_samples, n_channels]
    Toa_ptr,                             # [n_samples, n_channels]
    Td_ptr, Tdf_ptr,                     # [n_samples, n_layers, n_channels]
    Tmd_ptr, Tmf_ptr,
    Rsmd_ptr, Rsmf_ptr,
    Almd_ptr, Almf_ptr,
    # Outputs ([n_samples, n_layers+1] or [n_samples, n_layers])
    OutFdd_ptr,    # flux_down_direct  : [n_samples, n_layers+1]
    OutFddf_ptr,   # flux_down_diffuse : [n_samples, n_layers+1]
    OutFud_ptr,    # flux_up_diffuse   : [n_samples, n_layers+1]
    OutAbs_ptr,    # flux_absorbed     : [n_samples, n_layers]
    n_layers: tl.constexpr,
    n_channels: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """One program per atmospheric column."""
    sid = tl.program_id(0)

    c = tl.arange(0, BLOCK_C)
    cmask = c < n_channels

    # Per-channel state
    base = sid * n_channels + c
    flux_direct = tl.load(Fdir_in_ptr + base, mask=cmask, other=0.0)
    flux_diffuse = tl.load(Fdif_in_ptr + base, mask=cmask, other=0.0)
    toa = tl.load(Toa_ptr + base, mask=cmask, other=0.0)

    # Output base offsets (per-sample)
    out_base_p1 = sid * (n_layers + 1)   # for flux_down_*, flux_up_*
    out_base    = sid * n_layers         # for flux_absorbed

    # Layer 0 (top of atmosphere): assign initial fluxes summed across channels
    fdd0 = tl.sum(tl.where(cmask, flux_direct, 0.0))
    fdfd0 = tl.sum(tl.where(cmask, flux_diffuse, 0.0))
    fud0 = tl.sum(tl.where(cmask, flux_direct * toa, 0.0))
    tl.store(OutFdd_ptr  + out_base_p1, fdd0)
    tl.store(OutFddf_ptr + out_base_p1, fdfd0)
    tl.store(OutFud_ptr  + out_base_p1, fud0)

    sl_base = sid * n_layers * n_channels
    for i in range(n_layers):
        loff = sl_base + i * n_channels + c

        td   = tl.load(Td_ptr   + loff, mask=cmask, other=0.0)
        tdf  = tl.load(Tdf_ptr  + loff, mask=cmask, other=0.0)
        tmd  = tl.load(Tmd_ptr  + loff, mask=cmask, other=0.0)
        tmf  = tl.load(Tmf_ptr  + loff, mask=cmask, other=0.0)
        rsmd = tl.load(Rsmd_ptr + loff, mask=cmask, other=0.0)
        rsmf = tl.load(Rsmf_ptr + loff, mask=cmask, other=0.0)
        almd = tl.load(Almd_ptr + loff, mask=cmask, other=0.0)
        almf = tl.load(Almf_ptr + loff, mask=cmask, other=0.0)

        # Compute new flux state for this layer
        absorbed     = flux_direct * almd + flux_diffuse * almf
        new_fdd      = flux_direct * td
        new_fddf     = flux_direct * tmd  + flux_diffuse * (tdf + tmf)
        new_fud      = flux_direct * rsmd + flux_diffuse * rsmf

        # Channel-wise reductions for this layer
        absorbed_sum = tl.sum(tl.where(cmask, absorbed, 0.0))
        fdd_sum      = tl.sum(tl.where(cmask, new_fdd,  0.0))
        fddf_sum     = tl.sum(tl.where(cmask, new_fddf, 0.0))
        fud_sum      = tl.sum(tl.where(cmask, new_fud,  0.0))

        tl.store(OutAbs_ptr  + out_base    + i,     absorbed_sum)
        tl.store(OutFdd_ptr  + out_base_p1 + i + 1, fdd_sum)
        tl.store(OutFddf_ptr + out_base_p1 + i + 1, fddf_sum)
        tl.store(OutFud_ptr  + out_base_p1 + i + 1, fud_sum)

        # Carry state to next layer
        flux_direct = new_fdd
        flux_diffuse = new_fddf


class PropagationFused(nn.Module):
    """Fused replacement for train_network.Propagation.

    Output shape matches the original Propagation:
        flux_down_direct  : [n_samples, n_layers+1]
        flux_down_diffuse : [n_samples, n_layers+1]
        flux_up_diffuse   : [n_samples, n_layers+1]
        flux_absorbed     : [n_samples, n_layers]
    """

    def __init__(self, n_channel):
        super().__init__()
        self.n_channel = n_channel

    def forward(self, x):
        multireflected_layers, upward_reflection_toa, input_flux = x

        (t_direct, t_diffuse,
         t_multi_direct, t_multi_diffuse,
         r_surface_multi_direct, r_surface_multi_diffuse,
         a_layer_multi_direct, a_layer_multi_diffuse) = multireflected_layers

        flux_direct, flux_diffuse = input_flux

        n_samples, n_layers, n_channels = t_direct.shape
        device = t_direct.device

        flux_down_direct  = torch.empty((n_samples, n_layers + 1),
                                        device=device, dtype=torch.float32)
        flux_down_diffuse = torch.empty((n_samples, n_layers + 1),
                                        device=device, dtype=torch.float32)
        flux_up_diffuse   = torch.empty((n_samples, n_layers + 1),
                                        device=device, dtype=torch.float32)
        flux_absorbed     = torch.empty((n_samples, n_layers),
                                        device=device, dtype=torch.float32)

        BLOCK_C = triton.next_power_of_2(n_channels)

        _fused_propagation_kernel[(n_samples,)](
            flux_direct, flux_diffuse,
            upward_reflection_toa,
            t_direct, t_diffuse,
            t_multi_direct, t_multi_diffuse,
            r_surface_multi_direct, r_surface_multi_diffuse,
            a_layer_multi_direct, a_layer_multi_diffuse,
            flux_down_direct, flux_down_diffuse, flux_up_diffuse,
            flux_absorbed,
            n_layers=n_layers,
            n_channels=n_channels,
            BLOCK_C=BLOCK_C,
            num_warps=2,
        )

        return [flux_down_direct, flux_down_diffuse, flux_up_diffuse,
                flux_absorbed]
