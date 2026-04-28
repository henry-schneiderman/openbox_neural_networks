
import sys
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.profiler as profiler
#import torch_xla
#import torch_xla.core.xla_model as xm
from netCDF4 import Dataset
import time

# Specify network to evaluate by importing it as "tn"
#import train_network_3 as tn
import train_network as tn
import data_generation
import network_losses as nl

import triton
import triton.language as tl

default_float_type = torch.float32

os.environ['TRITON_PRINT_AUTOTUNING'] = '1'


# ---- Triton fused forward kernel ----
# Fuses the entire MLP forward pass (input + 2 hidden + output) into one
# kernel using 2D tl.dot operations.  All intermediate activations live in
# registers — zero global memory traffic between layers.
# Weights are pre-padded to multiples of 16 for tensor-core compatibility:
#   input  [16, 32]  (padded from [8, 32])
#   hidden [32, 32]  (original block-diagonal, zeros included)
#   output [32, 32]  (padded from [32, 24])
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
        triton.Config({'BLOCK_B': 512}, num_warps=8),
    ],
    key=['B'],
)

@triton.jit
def _fused_mlp_i8_kernel(
    X_ptr,   Out_ptr,
    W0_ptr,  B0_ptr,          # input  [16, 32]  [32]
    WH0_ptr, BH0_ptr,         # hidden [32, 32]  [32]
    WH1_ptr, BH1_ptr,         # hidden [32, 32]  [32]
    WO_ptr,  BO_ptr,          # output [32, 32]  [32]
    B,                        # batch size
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B

    c16 = tl.arange(0, 16)
    c32 = tl.arange(0, 32)

    # ---- load input [BLOCK_B, 16] (padded from 8 with mask) ----
    x = tl.load(X_ptr + offs_b[:, None] * 8 + c16[None, :],
                mask=mask_b[:, None] & (c16[None, :] < 8), other=0.0)

    # ---- input layer: [BLOCK_B,16] @ [16,32] + bias -> relu ----
    w0 = tl.load(W0_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0_ptr + c32)
    h = tl.dot(x, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    # ---- hidden layer 0: [BLOCK_B,32] @ [32,32] + bias -> relu ----
    wh0 = tl.load(WH0_ptr + c32[:, None] * 32 + c32[None, :])
    bh0 = tl.load(BH0_ptr + c32)
    h = tl.dot(h, wh0, input_precision="ieee") + bh0[None, :]
    h = tl.maximum(h, 0.0)

    # ---- hidden layer 1: [BLOCK_B,32] @ [32,32] + bias -> relu ----
    wh1 = tl.load(WH1_ptr + c32[:, None] * 32 + c32[None, :])
    bh1 = tl.load(BH1_ptr + c32)
    h = tl.dot(h, wh1, input_precision="ieee") + bh1[None, :]
    h = tl.maximum(h, 0.0)

    # ---- output layer: [BLOCK_B,32] @ [32,32] + bias (no relu) ----
    # Output weights are arranged in groups of 4 columns: 3 real outputs
    # followed by 1 padding column whose bias is -inf, so exp(-inf) = 0
    # and the padding does not affect the softmax result.
    wo = tl.load(WO_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BO_ptr + c32)
    out = tl.dot(h, wo, input_precision="ieee") + bo[None, :]

    # ---- fused softmax over groups of 4 (3 real + 1 padding at -inf) ----
    out_r = tl.reshape(out, [BLOCK_B * 8, 4])
    m = tl.max(out_r, axis=1)[:, None]
    out_r = tl.exp(out_r - m)
    s = tl.sum(out_r, axis=1)[:, None]
    out_r = out_r / s
    out = tl.reshape(out_r, [BLOCK_B, 32])

    # ---- store 24 real columns to [B, 24] packed output ----
    is_real = (c32 % 4) < 3
    out_col = (c32 // 4) * 3 + (c32 % 4)
    tl.store(Out_ptr + offs_b[:, None] * 24 + out_col[None, :],
             out, mask=mask_b[:, None] & is_real[None, :])

#best config selected: BLOCK_B: 256, num_warps: 8, num_ctas: 1, num_stages: 3, maxnreg: None;
#@triton.jit
def _fused_mlp_i8_kernel_old(
    X_ptr,   Out_ptr,
    W0_ptr,  B0_ptr,          # input  [16, 32]  [32]
    WH0_ptr, BH0_ptr,         # hidden [32, 32]  [32]
    WH1_ptr, BH1_ptr,         # hidden [32, 32]  [32]
    WO_ptr,  BO_ptr,          # output [32, 32]  [32]
    B,                        # batch size
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B

    c16 = tl.arange(0, 16)
    c32 = tl.arange(0, 32)

    # ---- load input [BLOCK_B, 16] (padded from 8 with mask) ----
    x = tl.load(X_ptr + offs_b[:, None] * 8 + c16[None, :],
                mask=mask_b[:, None] & (c16[None, :] < 8), other=0.0)

    # ---- input layer: [BLOCK_B,16] @ [16,32] + bias -> relu ----
    w0 = tl.load(W0_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0_ptr + c32)
    h = tl.dot(x, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    # ---- hidden layer 0: [BLOCK_B,32] @ [32,32] + bias -> relu ----
    wh0 = tl.load(WH0_ptr + c32[:, None] * 32 + c32[None, :])
    bh0 = tl.load(BH0_ptr + c32)
    h = tl.dot(h, wh0, input_precision="ieee") + bh0[None, :]
    h = tl.maximum(h, 0.0)

    # ---- hidden layer 1: [BLOCK_B,32] @ [32,32] + bias -> relu ----
    wh1 = tl.load(WH1_ptr + c32[:, None] * 32 + c32[None, :])
    bh1 = tl.load(BH1_ptr + c32)
    h = tl.dot(h, wh1, input_precision="ieee") + bh1[None, :]
    h = tl.maximum(h, 0.0)

    # ---- output layer: [BLOCK_B,32] @ [32,32] + bias (no relu) ----
    wo = tl.load(WO_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BO_ptr + c32)
    out = tl.dot(h, wo, input_precision="ieee") + bo[None, :]

    # ---- store first 24 columns to [B, 32] padded output ----
    tl.store(Out_ptr + offs_b[:, None] * 32 + c32[None, :],
             out, mask=mask_b[:, None])



#best config selected: BLOCK_B: 256, num_warps: 8, num_ctas: 1, num_stages: 3, maxnreg: None;
class MultipleMLPs_I8_Triton(nn.Module):
    """
    Triton-fused version of MultipleMLPs_I8.

    Fuses the entire forward pass (input + 2 hidden + output layers) into a
    single Triton kernel using 2D tl.dot operations.  All intermediate
    activations stay in GPU registers — no global memory traffic between
    layers.  Weights are pre-padded to multiples of 16 for tensor-core use.

    Requires exactly N_LAYERS=4 (1 input + 2 hidden + 1 output).
    """

    N_INPUTS = 8
    N_HIDDEN_NODES = 32
    N_OUTPUT_NODES = 24
    N_LAYERS = 4
    
    # Original init()
    def __init__(
            self, n_inputs, n_hidden_layers, n_channels, dropout_p, device, bias=False, requires_grad=True):

        super(MultipleMLPs_I8_Triton, self).__init__()
        
        assert n_inputs == self.N_INPUTS, f"Expected n_inputs={self.N_INPUTS}, got {n_inputs}"
        n_hidden_nodes = 32
        n_output_nodes = 24
        self.n_channels = n_channels
        self.n_hidden_layers = n_hidden_layers
        self.dropout_p = dropout_p
        self.device = device

        weight_values = torch.rand(
            (n_inputs, n_hidden_nodes), requires_grad=requires_grad,
            device=device, dtype=default_float_type)

        self.input_weight = nn.parameter.Parameter(weight_values,
                                                   requires_grad=requires_grad)

        self.bias = bias
        if bias:
            bias_values = torch.rand(
                (n_hidden_nodes,), requires_grad=requires_grad, device=device,
                dtype=default_float_type,
            )
            self.input_bias = nn.parameter.Parameter(
                bias_values, requires_grad=requires_grad)
            biases = []

        template = torch.ones((8, 8), device=device, dtype=default_float_type)
        self.filter = torch.block_diag(template, template, template, template)
        weights = []

        n_last = n_hidden_nodes
        for i in range(n_hidden_layers-1):
            weights.append(
                torch.rand(
                    (n_last, n_hidden_nodes), requires_grad=requires_grad,
                    device=device, dtype=default_float_type))
            if bias:
                biases.append(
                    torch.rand(
                        (n_hidden_nodes,), requires_grad=requires_grad,
                        device=device, dtype=default_float_type))
            n_last = n_hidden_nodes

        self.weights = torch.nn.ParameterList(weights)
        tmp_weights = torch.rand(
            (n_last, n_output_nodes), requires_grad=requires_grad, device=device,
            dtype=default_float_type)

        self.output_weights = nn.parameter.Parameter(
            tmp_weights, requires_grad=requires_grad)

        if bias:
            self.biases = torch.nn.ParameterList(biases)
            weight_values = torch.rand(
                (n_output_nodes,), requires_grad=requires_grad, device=device,
                dtype=default_float_type)

            self.output_bias = nn.parameter.Parameter(
                weight_values, requires_grad=requires_grad)

        template = torch.ones(
            (4, 3), device=device, dtype=default_float_type)
        self.output_filter = torch.block_diag(
            template, template, template, template,
            template, template, template, template)

    def reconfigure(self, device, use_original_structure=False):
        assert device == self.device, f"Expected n_inputs={self.device}, got {device}"
        # All weights are stored pre-padded for the kernel.
        # Input layer: [8, 32] padded to [16, 32]
        self.register_buffer('w0', torch.zeros(16, 32, device=self.device, dtype=default_float_type))
        self.register_buffer('b0', torch.zeros(32, device=self.device, dtype=default_float_type))
        # Hidden layers: [32, 32] (block-diagonal with zeros)
        self.register_buffer('wh0', torch.zeros(32, 32, device=self.device, dtype=default_float_type))
        self.register_buffer('bh0', torch.zeros(32, device=self.device, dtype=default_float_type))
        self.register_buffer('wh1', torch.zeros(32, 32, device=self.device, dtype=default_float_type))
        self.register_buffer('bh1', torch.zeros(32, device=self.device, dtype=default_float_type))
        # Output layer: [32, 24] padded to [32, 32]
        self.register_buffer('wo', torch.zeros(32, 32, device=self.device, dtype=default_float_type))
        self.register_buffer('bo', torch.zeros(32, device=self.device, dtype=default_float_type))
        
        """Copy weights from an original MultipleMLPs_I8 instance."""
        with torch.no_grad():
            # Input layer: [8, 32] -> top-left of [16, 32]
            self.w0[:8, :].copy_(self.input_weight)
            self.b0.copy_(self.input_bias)
            # Hidden layers: already [32, 32] with block-diagonal structure
            self.wh0.copy_(self.weights[0] * self.filter)
            self.bh0.copy_(self.biases[0])
            self.wh1.copy_(self.weights[1] * self.filter)
            self.bh1.copy_(self.biases[1])
            if use_original_structure:
                # Output layer: [32, 24] -> left of [32, 32]
                self.wo[:, :24].copy_(self.output_weights * self.output_filter)
                self.bo[:24].copy_(self.output_bias)
            else:
                # Output layer: rearrange [32, 24] → [32, 32] in groups of 4.
                # Original col 3*g+p maps to new col 4*g+p (p=0,1,2).
                # The 4th column in each group is padding (zero weight, -inf bias)
                # so that exp(-inf)=0 and it doesn't affect the fused softmax.
                for g in range(8):
                    self.wo[:, 4*g:4*g+3].copy_(self.output_weights[:, 3*g:3*g+3] * self.output_filter[:, 3*g:3*g+3])
                    self.bo[4*g:4*g+3].copy_(self.output_bias[3*g:3*g+3])
                    self.bo[4*g+3] = float('-inf')


    def reset_dropout(self, dropout_p):
        self.dropout_p = dropout_p

    def forward(self, x):
        B = x.shape[0]
        x = x.contiguous()
        # Allocate padded output [B, 32]; kernel writes all 32 cols
        #out_padded = torch.empty((B, 32), device=x.device, dtype=torch.float32)
        # Kernel fuses MLP + softmax; writes 24 packed columns directly
        out_padded = torch.empty((B, 24), device=x.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']),)
        _fused_mlp_i8_kernel[grid](
            x, out_padded,
            self.w0, self.b0,
            self.wh0, self.bh0,
            self.wh1, self.bh1,
            self.wo, self.bo,
            B,
        )
        #return out_padded[:, :24]
        return out_padded.reshape(B // self.n_channels, self.n_channels, 8, 3)


# ---- Triton fused forward kernel for I9 (9 inputs) ----
# Same architecture as I8 kernel but input is 9 elements instead of 8.
# Input [9, 32] padded to [16, 32] for tensor-core compatibility.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 64},  num_warps=2),
        triton.Config({'BLOCK_B': 128}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=4),
        triton.Config({'BLOCK_B': 256}, num_warps=8),
        triton.Config({'BLOCK_B': 512}, num_warps=8),
    ],
    key=['B'],
)
@triton.jit
def _fused_mlp_i9_kernel(
    X_ptr,   Out_ptr,
    W0_ptr,  B0_ptr,          # input  [16, 32]  [32]
    WH0_ptr, BH0_ptr,         # hidden [32, 32]  [32]
    WH1_ptr, BH1_ptr,         # hidden [32, 32]  [32]
    WO_ptr,  BO_ptr,          # output [32, 32]  [32]
    B,                        # batch size
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B

    c16 = tl.arange(0, 16)
    c32 = tl.arange(0, 32)

    # ---- load input [BLOCK_B, 16] (padded from 9 with mask) ----
    x = tl.load(X_ptr + offs_b[:, None] * 9 + c16[None, :],
                mask=mask_b[:, None] & (c16[None, :] < 9), other=0.0)

    # ---- input layer: [BLOCK_B,16] @ [16,32] + bias -> relu ----
    w0 = tl.load(W0_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0_ptr + c32)
    h = tl.dot(x, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    # ---- hidden layer 0: [BLOCK_B,32] @ [32,32] + bias -> relu ----
    wh0 = tl.load(WH0_ptr + c32[:, None] * 32 + c32[None, :])
    bh0 = tl.load(BH0_ptr + c32)
    h = tl.dot(h, wh0, input_precision="ieee") + bh0[None, :]
    h = tl.maximum(h, 0.0)

    # ---- hidden layer 1: [BLOCK_B,32] @ [32,32] + bias -> relu ----
    wh1 = tl.load(WH1_ptr + c32[:, None] * 32 + c32[None, :])
    bh1 = tl.load(BH1_ptr + c32)
    h = tl.dot(h, wh1, input_precision="ieee") + bh1[None, :]
    h = tl.maximum(h, 0.0)

    # ---- output layer: [BLOCK_B,32] @ [32,32] + bias (no relu) ----
    # Output weights are arranged in groups of 4 columns: 3 real outputs
    # followed by 1 padding column whose bias is -inf, so exp(-inf) = 0
    # and the padding does not affect the softmax result.
    wo = tl.load(WO_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BO_ptr + c32)
    out = tl.dot(h, wo, input_precision="ieee") + bo[None, :]

    # ---- fused softmax over groups of 4 (3 real + 1 padding at -inf) ----
    out_r = tl.reshape(out, [BLOCK_B * 8, 4])
    m = tl.max(out_r, axis=1)[:, None]
    out_r = tl.exp(out_r - m)
    s = tl.sum(out_r, axis=1)[:, None]
    out_r = out_r / s
    out = tl.reshape(out_r, [BLOCK_B, 32])

    # ---- store 24 real columns to [B, 24] packed output ----
    is_real = (c32 % 4) < 3
    out_col = (c32 // 4) * 3 + (c32 % 4)
    tl.store(Out_ptr + offs_b[:, None] * 24 + out_col[None, :],
             out, mask=mask_b[:, None] & is_real[None, :])

#best config selected: BLOCK_B: 512, num_warps: 8, num_ctas: 1, num_stages: 3, maxnreg: None;
#@triton.jit
def _fused_mlp_i9_kernel_old(
    X_ptr,   Out_ptr,
    W0_ptr,  B0_ptr,          # input  [16, 32]  [32]
    WH0_ptr, BH0_ptr,         # hidden [32, 32]  [32]
    WH1_ptr, BH1_ptr,         # hidden [32, 32]  [32]
    WO_ptr,  BO_ptr,          # output [32, 32]  [32]
    B,                        # batch size
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B

    c16 = tl.arange(0, 16)
    c32 = tl.arange(0, 32)

    # ---- load input [BLOCK_B, 16] (padded from 9 with mask) ----
    x = tl.load(X_ptr + offs_b[:, None] * 9 + c16[None, :],
                mask=mask_b[:, None] & (c16[None, :] < 9), other=0.0)

    # ---- input layer: [BLOCK_B,16] @ [16,32] + bias -> relu ----
    w0 = tl.load(W0_ptr + c16[:, None] * 32 + c32[None, :])
    b0 = tl.load(B0_ptr + c32)
    h = tl.dot(x, w0, input_precision="ieee") + b0[None, :]
    h = tl.maximum(h, 0.0)

    # ---- hidden layer 0: [BLOCK_B,32] @ [32,32] + bias -> relu ----
    wh0 = tl.load(WH0_ptr + c32[:, None] * 32 + c32[None, :])
    bh0 = tl.load(BH0_ptr + c32)
    h = tl.dot(h, wh0, input_precision="ieee") + bh0[None, :]
    h = tl.maximum(h, 0.0)

    # ---- hidden layer 1: [BLOCK_B,32] @ [32,32] + bias -> relu ----
    wh1 = tl.load(WH1_ptr + c32[:, None] * 32 + c32[None, :])
    bh1 = tl.load(BH1_ptr + c32)
    h = tl.dot(h, wh1, input_precision="ieee") + bh1[None, :]
    h = tl.maximum(h, 0.0)

    # ---- output layer: [BLOCK_B,32] @ [32,32] + bias (no relu) ----
    wo = tl.load(WO_ptr + c32[:, None] * 32 + c32[None, :])
    bo = tl.load(BO_ptr + c32)
    out = tl.dot(h, wo, input_precision="ieee") + bo[None, :]

    # ---- store to [B, 32] padded output ----
    tl.store(Out_ptr + offs_b[:, None] * 32 + c32[None, :],
             out, mask=mask_b[:, None])



#best config selected: BLOCK_B: 512, num_warps: 8, num_ctas: 1, num_stages: 3, maxnreg: None;
class MultipleMLPs_I9_Triton(nn.Module):
    """
    Triton-fused version of MultipleMLPs_I9.

    Identical to MultipleMLPs_I8_Triton except the input layer is 9 -> 32
    instead of 8 -> 32.  Input weights are padded from [9, 32] to [16, 32].

    Requires exactly N_LAYERS=4 (1 input + 2 hidden + 1 output).
    """

    N_INPUTS = 9
    N_HIDDEN_NODES = 32
    N_OUTPUT_NODES = 24
    N_LAYERS = 4
    
    def __init__(
            self, n_inputs, n_hidden_layers, n_channels, dropout_p, device, bias=False, requires_grad=True):

        super(MultipleMLPs_I9_Triton, self).__init__()
        
        assert n_inputs == self.N_INPUTS, f"Expected n_inputs={self.N_INPUTS}, got {n_inputs}"
        n_hidden_nodes = 32
        n_output_nodes = 24
        self.n_channels = n_channels
        self.n_hidden_layers = n_hidden_layers
        self.dropout_p = dropout_p
        self.device = device

        weight_values = torch.rand(
            (n_inputs, n_hidden_nodes), requires_grad=requires_grad,
            device=device, dtype=default_float_type)

        self.input_weight = nn.parameter.Parameter(weight_values,
                                                   requires_grad=requires_grad)

        self.bias = bias
        if bias:
            bias_values = torch.rand(
                (n_hidden_nodes,), requires_grad=requires_grad, device=device,
                dtype=default_float_type,
            )
            self.input_bias = nn.parameter.Parameter(
                bias_values, requires_grad=requires_grad)
            biases = []

        template = torch.ones((8, 8), device=device, dtype=default_float_type)
        self.filter = torch.block_diag(template, template, template, template)
        weights = []

        n_last = n_hidden_nodes
        for i in range(n_hidden_layers-1):
            weights.append(
                torch.rand(
                    (n_last, n_hidden_nodes), requires_grad=requires_grad,
                    device=device, dtype=default_float_type))
            if bias:
                biases.append(
                    torch.rand(
                        (n_hidden_nodes,), requires_grad=requires_grad,
                        device=device, dtype=default_float_type))
            n_last = n_hidden_nodes

        self.weights = torch.nn.ParameterList(weights)
        tmp_weights = torch.rand(
            (n_last, n_output_nodes), requires_grad=requires_grad, device=device,
            dtype=default_float_type)

        self.output_weights = nn.parameter.Parameter(
            tmp_weights, requires_grad=requires_grad)

        if bias:
            self.biases = torch.nn.ParameterList(biases)
            weight_values = torch.rand(
                (n_output_nodes,), requires_grad=requires_grad, device=device,
                dtype=default_float_type)

            self.output_bias = nn.parameter.Parameter(
                weight_values, requires_grad=requires_grad)

        template = torch.ones(
            (4, 3), device=device, dtype=default_float_type)
        self.output_filter = torch.block_diag(
            template, template, template, template,
            template, template, template, template)

    def reconfigure(self,device, use_original_structure=False):
        assert device == self.device, f"Expected n_inputs={self.device}, got {device}"
        # All weights are stored pre-padded for the kernel.
        # Input layer: [9, 32] padded to [16, 32]
        self.register_buffer('w0', torch.zeros(16, 32, device=self.device, dtype=default_float_type))
        self.register_buffer('b0', torch.zeros(32, device=self.device, dtype=default_float_type))
        # Hidden layers: [32, 32] (block-diagonal with zeros)
        self.register_buffer('wh0', torch.zeros(32, 32, device=self.device, dtype=default_float_type))
        self.register_buffer('bh0', torch.zeros(32, device=self.device, dtype=default_float_type))
        self.register_buffer('wh1', torch.zeros(32, 32, device=self.device, dtype=default_float_type))
        self.register_buffer('bh1', torch.zeros(32, device=self.device, dtype=default_float_type))
        # Output layer: [32, 24] padded to [32, 32]
        self.register_buffer('wo', torch.zeros(32, 32, device=self.device, dtype=default_float_type))
        self.register_buffer('bo', torch.zeros(32, device=self.device, dtype=default_float_type))
        
        """Copy weights from an original MultipleMLPs_I9 instance."""
        with torch.no_grad():
            # Input layer: [9, 32] -> top-left of [16, 32]
            self.w0[:9, :].copy_(self.input_weight)
            self.b0.copy_(self.input_bias)
            # Hidden layers: already [32, 32] with block-diagonal structure
            self.wh0.copy_(self.weights[0] * self.filter)
            self.bh0.copy_(self.biases[0])
            self.wh1.copy_(self.weights[1] * self.filter)
            self.bh1.copy_(self.biases[1])
            # Output layer: [32, 24] -> left of [32, 32]
            if use_original_structure:
                self.wo[:, :24].copy_(self.output_weights * self.output_filter)
                self.bo[:24].copy_(self.output_bias)
            else:
                # Output layer: rearrange [32, 24] → [32, 32] in groups of 4.
                # Original col 3*g+p maps to new col 4*g+p (p=0,1,2).
                # The 4th column in each group is padding (zero weight, -inf bias)
                # so that exp(-inf)=0 and it doesn't affect the fused softmax.
                for g in range(8):
                    self.wo[:, 4*g:4*g+3].copy_(self.output_weights[:, 3*g:3*g+3] * self.output_filter[:, 3*g:3*g+3])
                    self.bo[4*g:4*g+3].copy_(self.output_bias[3*g:3*g+3])
                    self.bo[4*g+3] = float('-inf')
    def reset_dropout(self, dropout_p):
        self.dropout_p = dropout_p

    def forward(self, x):
        B = x.shape[0]
        x = x.contiguous()
        # Allocate padded output [B, 32]; kernel writes all 32 cols
        #out_padded = torch.empty((B, 32), device=x.device, dtype=torch.float32)
        # Kernel fuses MLP + softmax; writes 24 packed columns directly to output.
        out_padded = torch.empty((B, 24), device=x.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_B']),)
        _fused_mlp_i9_kernel[grid](
            x, out_padded,
            self.w0, self.b0,
            self.wh0, self.bh0,
            self.wh1, self.bh1,
            self.wo, self.bo,
            B,
        )
        #return out_padded[:, :24]
        return out_padded.reshape(B // self.n_channels, self.n_channels, 8, 3)
