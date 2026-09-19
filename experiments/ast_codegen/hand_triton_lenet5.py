#!/usr/bin/env python3
"""Handwritten naive Triton LeNet-5 for comparison with the generated
(lenet5_triton.py) backend output.  Same op sequence and fp32 semantics;
kernels are written idiomatically by hand (one program per (channel,
output-row) for conv/pool) instead of the generated flat grid.

Reports: max abs diff vs the NumPy reference, and kernel-chain time
(CUDA events, min/median of 50) next to the generated chain.
"""

import numpy as np
import torch
import triton
import triton.language as tl

from run_triton_lenet5 import (  # reuse staged data + reference
    ref_output, input_, kc1, bc1, kc2, bc2, fc1_w, fc1_b,
    fc2_w, fc2_b, fc3_w, fc3_b, bench, t, dev,
)

ITERS = 50


@triton.jit
def conv_hand(x_ptr, w_ptr, b_ptr, out_ptr,
              CI: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
              HO: tl.constexpr, WO: tl.constexpr, K: tl.constexpr,
              BLOCK: tl.constexpr):
    # one program per (co, ho); vectorised across wo
    co = tl.program_id(0)
    ho = tl.program_id(1)
    wo = tl.arange(0, BLOCK)
    mask = wo < WO
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for ci in range(CI):
        for di in range(K):
            for dj in range(K):
                xv = tl.load(x_ptr + ci * H * W + (ho + di) * W + (wo + dj),
                             mask=mask, other=0.0)
                wv = tl.load(w_ptr + (co * CI + ci) * K * K + di * K + dj)
                acc += xv * wv
    acc += tl.load(b_ptr + co)
    tl.store(out_ptr + co * HO * WO + ho * WO + wo, acc, mask=mask)


@triton.jit
def relu_hand(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(out_ptr + offs,
             tl.maximum(tl.load(x_ptr + offs, mask=mask, other=0.0), 0.0),
             mask=mask)


@triton.jit
def pool_hand(x_ptr, out_ptr,
              C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
              HO: tl.constexpr, WO: tl.constexpr, BLOCK: tl.constexpr):
    c = tl.program_id(0)
    ho = tl.program_id(1)
    wo = tl.arange(0, BLOCK)
    mask = wo < WO
    base = x_ptr + c * H * W + 2 * ho * W
    x00 = tl.load(base + 2 * wo, mask=mask, other=0.0)
    x01 = tl.load(base + 2 * wo + 1, mask=mask, other=0.0)
    x10 = tl.load(base + W + 2 * wo, mask=mask, other=0.0)
    x11 = tl.load(base + W + 2 * wo + 1, mask=mask, other=0.0)
    m = tl.maximum(tl.maximum(x00, x01), tl.maximum(x10, x11))
    tl.store(out_ptr + c * HO * WO + ho * WO + wo, m, mask=mask)


@triton.jit
def linear_hand(x_ptr, w_ptr, b_ptr, out_ptr,
                IN: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr):
    f = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = f < F
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(IN):
        acc += tl.load(x_ptr + k) * tl.load(w_ptr + f * IN + k, mask=mask, other=0.0)
    acc += tl.load(b_ptr + f, mask=mask, other=0.0)
    tl.store(out_ptr + f, acc, mask=mask)


@triton.jit
def softmax_hand(x_ptr, out_ptr, F: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < F
    xv = tl.load(x_ptr + offs, mask=mask, other=float("-inf"))
    ev = tl.exp(xv)
    tl.store(out_ptr + offs, ev / tl.sum(ev, axis=0), mask=mask)


g_in, g_kc1, g_bc1 = t(input_), t(kc1), t(bc1)
g_kc2, g_bc2 = t(kc2), t(bc2)
g_fc1w, g_fc1b = t(fc1_w), t(fc1_b)
g_fc2w, g_fc2b = t(fc2_w), t(fc2_b)
g_fc3w, g_fc3b = t(fc3_w), t(fc3_b)


def hand_forward():
    c1 = torch.empty(6, 28, 28, device=dev)
    conv_hand[(6, 28)](g_in, g_kc1, g_bc1, c1, 1, 32, 32, 28, 28, 5, BLOCK=32)
    r1 = torch.empty_like(c1)
    relu_hand[(triton.cdiv(4704, 256),)](c1, r1, 4704, BLOCK=256)
    p1 = torch.empty(6, 14, 14, device=dev)
    pool_hand[(6, 14)](r1, p1, 6, 28, 28, 14, 14, BLOCK=16)
    c2 = torch.empty(16, 10, 10, device=dev)
    conv_hand[(16, 10)](p1, g_kc2, g_bc2, c2, 6, 14, 14, 10, 10, 5, BLOCK=16)
    r2 = torch.empty_like(c2)
    relu_hand[(triton.cdiv(1600, 256),)](c2, r2, 1600, BLOCK=256)
    p2 = torch.empty(16, 5, 5, device=dev)
    pool_hand[(16, 5)](r2, p2, 16, 10, 10, 5, 5, BLOCK=8)
    flat = p2.reshape(-1)
    fc1 = torch.empty(120, device=dev)
    linear_hand[(triton.cdiv(120, 128),)](flat, g_fc1w, g_fc1b, fc1, 400, 120, BLOCK=128)
    fc1r = torch.empty_like(fc1)
    relu_hand[(1,)](fc1, fc1r, 120, BLOCK=128)
    fc2 = torch.empty(84, device=dev)
    linear_hand[(1,)](fc1r, g_fc2w, g_fc2b, fc2, 120, 84, BLOCK=128)
    fc2r = torch.empty_like(fc2)
    relu_hand[(1,)](fc2, fc2r, 84, BLOCK=128)
    fc3 = torch.empty(10, device=dev)
    linear_hand[(1,)](fc2r, g_fc3w, g_fc3b, fc3, 84, 10, BLOCK=16)
    out = torch.empty(10, device=dev)
    softmax_hand[(1,)](fc3, out, 10, BLOCK=16)
    return out


h_out = hand_forward().detach().cpu().numpy()
herr = np.max(np.abs(h_out - ref_output))
print(f"max |hand triton - ref| = {herr:.3e}")
assert np.allclose(ref_output, h_out, atol=1e-5, rtol=1e-5)

import lenet5_triton as G


def gen_forward():
    return G.forward(g_in, g_kc1, g_bc1, g_kc2, g_bc2,
                     g_fc1w, g_fc1b, g_fc2w, g_fc2b, g_fc3w, g_fc3b)


g_out = gen_forward().detach().cpu().numpy()
print(f"max |hand - generated|  = {np.max(np.abs(h_out - g_out)):.3e}")

hmin, hmed = bench(hand_forward, ITERS)
print(f"handwritten Triton chain: min {hmin*1000:8.1f} us   median {hmed*1000:8.1f} us")
gmin, gmed = bench(gen_forward, ITERS)
print(f"generated  Triton chain: min {gmin*1000:8.1f} us   median {gmed*1000:8.1f} us")
