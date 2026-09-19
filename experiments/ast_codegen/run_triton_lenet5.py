#!/usr/bin/env python3
"""Host harness for the Triton LeNet-5 backend (G2 phase 1).

- Loads the same seed-7 data as validate_lenet5.py (NumPy reference).
- Runs the generated Triton kernel chain (lenet5_triton.py, produced by
  comm_to_triton.ml from the Coq-extracted AST) on GPU.
- Numerical check: max abs diff vs the NumPy reference.
- Performance: CUDA events around the kernel chain only (no H2D/D2H
  transfers), min/median of 50 runs; same measurement for a PyTorch eager
  LeNet-5 with identical weights.
"""

import numpy as np
import torch
import torch.nn.functional as Fn

import lenet5_triton as G

ITERS = 50

# ---- data: identical to validate_lenet5.py -------------------------------
np.random.seed(7)
scale = 1e-3
input_ = (scale * np.random.randn(1, 32, 32)).astype(np.float32)
kc1    = (scale * np.random.randn(6, 1, 5, 5)).astype(np.float32)
bc1    = (scale * np.random.randn(6)).astype(np.float32)
kc2    = (scale * np.random.randn(16, 6, 5, 5)).astype(np.float32)
bc2    = (scale * np.random.randn(16)).astype(np.float32)
fc1_w  = (scale * np.random.randn(120, 400)).astype(np.float32)
fc1_b  = (scale * np.random.randn(120)).astype(np.float32)
fc2_w  = (scale * np.random.randn(84, 120)).astype(np.float32)
fc2_b  = (scale * np.random.randn(84)).astype(np.float32)
fc3_w  = (scale * np.random.randn(10, 84)).astype(np.float32)
fc3_b  = (scale * np.random.randn(10)).astype(np.float32)


def relu(x):
    return np.maximum(x, 0.0)


def conv3d_no_pad(x, w, b):
    N, Cin, H, W = x.shape
    Cout, _, K, _ = w.shape
    Ho, Wo = H - K + 1, W - K + 1
    out = np.zeros((N, Cout, Ho, Wo), dtype=np.float32)
    for n in range(N):
        for co in range(Cout):
            for i in range(Ho):
                for j in range(Wo):
                    s = 0.0
                    for ci in range(Cin):
                        for di in range(K):
                            for dj in range(K):
                                s += x[n, ci, i + di, j + dj] * w[co, ci, di, dj]
                    out[n, co, i, j] = s + b[co]
    return out


def maxpool2d_k2(x):
    N, C, H, W = x.shape
    out = np.zeros((N, C, H // 2, W // 2), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            for i in range(H // 2):
                for j in range(W // 2):
                    out[n, c, i, j] = np.max(x[n, c, 2 * i:2 * i + 2, 2 * j:2 * j + 2])
    return out


def softmax1d(x):
    # matches the AST semantics (no max subtraction), same as validate_lenet5
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def ref_forward():
    x = input_.reshape(1, 1, 32, 32)
    x = conv3d_no_pad(x, kc1, bc1)
    x = relu(x)
    x = maxpool2d_k2(x)
    x = conv3d_no_pad(x, kc2, bc2)
    x = relu(x)
    x = maxpool2d_k2(x)
    x = x.reshape(-1)
    x = relu(x @ fc1_w.T + fc1_b)
    x = relu(x @ fc2_w.T + fc2_b)
    x = x @ fc3_w.T + fc3_b
    return softmax1d(x)


ref_output = ref_forward()

# ---- Triton forward ------------------------------------------------------
dev = "cuda"
t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)

g_in, g_kc1, g_bc1 = t(input_), t(kc1), t(bc1)
g_kc2, g_bc2 = t(kc2), t(bc2)
g_fc1w, g_fc1b, g_fc2w, g_fc2b = t(fc1_w), t(fc1_b), t(fc2_w), t(fc2_b)
g_fc3w, g_fc3b = t(fc3_w), t(fc3_b)


def triton_forward():
    return G.forward(g_in, g_kc1, g_bc1, g_kc2, g_bc2,
                     g_fc1w, g_fc1b, g_fc2w, g_fc2b, g_fc3w, g_fc3b)


out = triton_forward().detach().cpu().numpy()
err = np.max(np.abs(out - ref_output))
print("Triton output:", out)
print("Ref output:   ", ref_output)
print(f"max |triton - ref| = {err:.3e}")
ok = np.allclose(ref_output, out, atol=1e-5, rtol=1e-5)
print("Match (atol=rtol=1e-5):", ok)
assert ok

# ---- PyTorch eager baseline (same weights) -------------------------------
def torch_forward(x):
    x = Fn.relu(Fn.conv2d(x, g_kc1, g_bc1))
    x = Fn.max_pool2d(x, 2)
    x = Fn.relu(Fn.conv2d(x, g_kc2, g_bc2))
    x = Fn.max_pool2d(x, 2)
    x = x.reshape(1, -1)
    x = Fn.relu(x @ g_fc1w.T + g_fc1b)
    x = Fn.relu(x @ g_fc2w.T + g_fc2b)
    x = x @ g_fc3w.T + g_fc3b
    return Fn.softmax(x, dim=-1)


t_in4 = g_in.reshape(1, 1, 32, 32)
t_out = torch_forward(t_in4).detach().cpu().numpy().reshape(-1)
terr = np.max(np.abs(t_out - ref_output))
print(f"max |torch eager - ref| = {terr:.3e}")


# ---- benchmark: CUDA events, kernel-chain only, min/median of 50 ---------
def bench(fn, iters=ITERS):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times = np.array(times)
    return times.min(), np.median(times)


tmin, tmed = bench(triton_forward)
print(f"Triton kernel chain:  min {tmin*1000:8.1f} us   median {tmed*1000:8.1f} us  (n={ITERS})")

pmin, pmed = bench(lambda: torch_forward(t_in4))
print(f"PyTorch eager (GPU):  min {pmin*1000:8.1f} us   median {pmed*1000:8.1f} us  (n={ITERS})")
