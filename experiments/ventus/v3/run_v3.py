#!/usr/bin/env python3
"""Ionia Ventus V3 validation driver: the full LeNet-5 network.

Uses exactly the same RNG as experiments/ast_codegen/validate_lenet5_par.py
(seed=7, scale=1e-3, same sampling order) to generate inputs and weights,
written out as .bin files; v3_host then launches the 13 kernels on spike in
manifest order; the host reads back each layer's output buffer for a
per-layer comparison against the NumPy reference (fp64).
Final check: softmax output max_abs_diff <= 1e-5.
"""
import os
import subprocess
import sys
import time

import numpy as np

V3 = os.path.dirname(os.path.abspath(__file__))
HOST = os.path.join(V3, "build", "v3_host")
KERNELS = os.path.join(V3, "kernels")
DATA = os.path.join(V3, "data")

# ---- inputs/weights: identical to validate_lenet5_par.py (M7) ----
np.random.seed(7)
scale = 1e-3
input_  = (scale * np.random.randn(1, 32, 32)).astype(np.float32)
kc1     = (scale * np.random.randn(6, 1, 5, 5)).astype(np.float32)
bc1     = (scale * np.random.randn(6)).astype(np.float32)
kc2     = (scale * np.random.randn(16, 6, 5, 5)).astype(np.float32)
bc2     = (scale * np.random.randn(16)).astype(np.float32)
fc1_w   = (scale * np.random.randn(120, 400)).astype(np.float32)
fc1_b   = (scale * np.random.randn(120)).astype(np.float32)
fc2_w   = (scale * np.random.randn(84, 120)).astype(np.float32)
fc2_b   = (scale * np.random.randn(84)).astype(np.float32)
fc3_w   = (scale * np.random.randn(10, 84)).astype(np.float32)
fc3_b   = (scale * np.random.randn(10)).astype(np.float32)

ARRAYS = {
    "input": input_, "kc1": kc1, "bc1": bc1, "kc2": kc2, "bc2": bc2,
    "fc1_w": fc1_w, "fc1_b": fc1_b, "fc2_w": fc2_w, "fc2_b": fc2_b,
    "fc3_w": fc3_w, "fc3_b": fc3_b,
}


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
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def ref_layers():
    """Return ordered [(kernel_out_name, reference_array)] per layer."""
    x = input_.reshape(1, 1, 32, 32)
    c1 = conv3d_no_pad(x, kc1, bc1)
    r1 = relu(c1)
    p1 = maxpool2d_k2(r1)
    c2 = conv3d_no_pad(p1, kc2, bc2)
    r2 = relu(c2)
    p2 = maxpool2d_k2(r2)
    flat = p2.reshape(-1)
    fc1o = flat @ fc1_w.T + fc1_b
    fc1r = relu(fc1o)
    fc2o = fc1r @ fc2_w.T + fc2_b
    fc2r = relu(fc2o)
    fc3o = fc2r @ fc3_w.T + fc3_b
    output = softmax1d(fc3o)
    return [
        ("conv1", "c1", c1.reshape(-1)),
        ("relu1", "r1", r1.reshape(-1)),
        ("pool1", "p1", p1.reshape(-1)),
        ("conv2", "c2", c2.reshape(-1)),
        ("relu2", "r2", r2.reshape(-1)),
        ("pool2", "p2", p2.reshape(-1)),
        ("flatten", "flat", flat),
        ("fc1", "fc1o", fc1o),
        ("relu3", "fc1r", fc1r),
        ("fc2", "fc2o", fc2o),
        ("relu4", "fc2r", fc2r),
        ("fc3", "fc3o", fc3o),
        ("softmax", "output", output),
    ]


def main():
    os.makedirs(DATA, exist_ok=True)
    for name, arr in ARRAYS.items():
        arr.astype("<f4").tofile(os.path.join(DATA, name + ".bin"))

    t0 = time.time()
    r = subprocess.run([HOST, KERNELS, DATA], capture_output=True, text=True)
    dt = time.time() - t0
    print(r.stdout)
    if r.returncode != 0:
        print(f"v3_host FAIL rc={r.returncode}\n{r.stderr}")
        return 1
    print(f"chain wall time: {dt:.2f}s")

    npass = 0
    layers = ref_layers()
    for kname, pname, ref in layers:
        got = np.fromfile(os.path.join(DATA, f"v3out_{kname}_{pname}.bin"),
                          dtype="<f4")
        if got.shape != ref.shape:
            print(f"[{kname}] FAIL shape {got.shape} vs ref {ref.shape}")
            continue
        diff = np.abs(got.astype(np.float64) - ref.astype(np.float64))
        max_diff = float(diff.max()) if diff.size else 0.0
        ok = bool(np.isfinite(got).all()) and max_diff <= 1e-5
        npass += ok
        print(f"[{kname:8s}] {'PASS' if ok else 'FAIL'} n={got.size:5d} "
              f"max_abs_diff={max_diff:.3e}")
    final_diff = np.abs(
        np.fromfile(os.path.join(DATA, "v3out_softmax_output.bin"), dtype="<f4")
        - layers[-1][2]).max()
    print(f"V3 LeNet-5: {npass}/{len(layers)} layers PASS; "
          f"final softmax max_abs_diff={final_diff:.3e} (threshold 1e-5)")
    return 0 if npass == len(layers) else 1


if __name__ == "__main__":
    sys.exit(main())
