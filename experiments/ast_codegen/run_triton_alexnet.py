#!/usr/bin/env python3
"""Host harness for the Triton AlexNet backend (G2 phase 2).

- Loads the same seed-123 data as validate_cuda_alexnet.py (NumPy reference).
- Runs the generated Triton kernel chain (alexnet_triton.py, produced by
  comm_to_triton.ml from the Coq-extracted AST) on GPU — the two convs use
  the new masked-load padding path.
- Numerical check: max abs diff vs the NumPy reference.
- Performance: CUDA events around the kernel chain only (no H2D/D2H
  transfers), min/median of 50 runs; same measurement for a PyTorch eager
  AlexNet (repo mini version) with identical weights.
"""

import numpy as np
import torch
import torch.nn.functional as Fn

import alexnet_triton as G

ITERS = 50

# ---- data: identical to validate_cuda_alexnet.py -------------------------
np.random.seed(123)
scale = 1e-3

INPUT   = (scale * np.random.randn(1, 3, 8, 8)).astype(np.float32)
CONV1_W = (scale * np.random.randn(8, 3, 3, 3)).astype(np.float32)
CONV1_B = (scale * np.random.randn(8)).astype(np.float32)
CONV2_W = (scale * np.random.randn(16, 8, 3, 3)).astype(np.float32)
CONV2_B = (scale * np.random.randn(16)).astype(np.float32)
FC1_W   = (scale * np.random.randn(16, 64)).astype(np.float32)
FC1_B   = (scale * np.random.randn(16)).astype(np.float32)
FC2_W   = (scale * np.random.randn(10, 16)).astype(np.float32)
FC2_B   = (scale * np.random.randn(10)).astype(np.float32)


def relu(x):
    return np.maximum(x, np.float32(0.0))


def conv2d_pad1_k3(x, w, b):
    """Same accumulation order as the Coq/kernel reduction: c, k1, k2."""
    N, Cin, H, W = x.shape
    Cout = w.shape[0]
    out = np.zeros((N, Cout, H, W), dtype=np.float32)
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    for n in range(N):
        for co in range(Cout):
            for i in range(H):
                for j in range(W):
                    s = np.float32(0.0)
                    for ci in range(Cin):
                        for k1 in range(3):
                            for k2 in range(3):
                                s += padded[n, ci, i + k1, j + k2] * w[co, ci, k1, k2]
                    out[n, co, i, j] = s + b[co]
    return out


def maxpool2d_k2(x):
    N, C, H, W = x.shape
    out = np.zeros((N, C, H // 2, W // 2), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            for i in range(H // 2):
                for j in range(W // 2):
                    out[n, c, i, j] = np.max(x[n, c, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2])
    return out


def linear(x, w, b):
    return x @ w.T + b


def softmax(x):
    # No max-subtraction: matches the Coq/kernel semantics exactly.
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def numpy_reference():
    x = relu(conv2d_pad1_k3(INPUT, CONV1_W, CONV1_B))
    x = maxpool2d_k2(x)
    x = relu(conv2d_pad1_k3(x, CONV2_W, CONV2_B))
    x = maxpool2d_k2(x)
    x = x.reshape(1, -1)
    x = linear(x, FC1_W, FC1_B)
    x = linear(x, FC2_W, FC2_B)
    return softmax(x)


ref_output = numpy_reference()

# ---- Triton forward ------------------------------------------------------
dev = "cuda"
t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)

g = {
    "input": t(INPUT),
    "conv1_w": t(CONV1_W), "conv1_b": t(CONV1_B),
    "conv2_w": t(CONV2_W), "conv2_b": t(CONV2_B),
    "fc1_w": t(FC1_W), "fc1_b": t(FC1_B),
    "fc2_w": t(FC2_W), "fc2_b": t(FC2_B),
}


def triton_forward():
    return G.forward(g["input"], g["conv1_w"], g["conv1_b"],
                     g["conv2_w"], g["conv2_b"],
                     g["fc1_w"], g["fc1_b"], g["fc2_w"], g["fc2_b"])


out = triton_forward().detach().cpu().numpy().reshape(1, 10)
err = np.max(np.abs(out - ref_output))
print("Triton output:", out[0])
print("Ref output:   ", ref_output[0])
print(f"max |triton - ref| = {err:.3e}")
ok = np.allclose(ref_output, out, atol=1e-5, rtol=1e-5)
print("Match (atol=rtol=1e-5):", ok)
assert ok

# ---- PyTorch eager baseline (same weights) -------------------------------
def torch_forward(x):
    x = Fn.relu(Fn.conv2d(x, g["conv1_w"], g["conv1_b"], padding=1))
    x = Fn.max_pool2d(x, 2)
    x = Fn.relu(Fn.conv2d(x, g["conv2_w"], g["conv2_b"], padding=1))
    x = Fn.max_pool2d(x, 2)
    x = x.reshape(1, -1)
    x = x @ g["fc1_w"].T + g["fc1_b"]
    x = x @ g["fc2_w"].T + g["fc2_b"]
    return Fn.softmax(x, dim=-1)


t_out = torch_forward(g["input"]).detach().cpu().numpy()
terr = np.max(np.abs(t_out - ref_output))
print(f"max |torch eager - ref| = {terr:.3e}")


# ---- benchmark: CUDA events, kernel-chain only, min/median of 50 ---------
def bench(fn, iters=ITERS, warmup=20):
    for _ in range(warmup):
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

pmin, pmed = bench(lambda: torch_forward(g["input"]))
print(f"PyTorch eager (GPU):  min {pmin*1000:8.1f} us   median {pmed*1000:8.1f} us  (n={ITERS})")
