#!/usr/bin/env python3
"""Ionia Ventus V4 validation driver: the full AlexNet (reduced sizes).

Same RNG protocol as experiments/ast_codegen/validate_alexnet_par.py
(seed=42, scale=1e-3, same sampling order); only the shapes are scaled down:
  input 3x48x48, channels /4 (16/48/96/64/64), fc 1024/1024/256.
Layer sizes: c1=11x11(s4p2k11), p1=5x5, c2=5x5, p2=2x2, c3=c4=c5=2x2, p5=1x1.
The reference mirrors the Coq semantics (Conv4d zero padding + bias added
last; Maxpool4d_pad zero padding + maxl zero init; Softmax without max
subtraction; flatten in (c,i,j) row-major order), computed in fp64.
Final check: softmax output max_abs_diff <= 1e-5.
"""
import os
import subprocess
import sys
import time

import numpy as np

V4 = os.path.dirname(os.path.abspath(__file__))
HOST = os.path.join(V4, "build", "v4_host")
KERNELS = os.path.join(V4, "kernels_alexnet")
DATA = os.path.join(V4, "data_alexnet")

# ---- inputs/weights: same RNG protocol as validate_alexnet_par.py (M9) ----
np.random.seed(42)
scale = 1e-3


def rand(*shape):
    return (scale * np.random.randn(*shape)).astype(np.float32)


input_ = rand(1, 3, 48, 48)
c1_w = rand(16, 3, 11, 11)
c1_b = rand(16)
c2_w = rand(48, 16, 5, 5)
c2_b = rand(48)
c3_w = rand(96, 48, 3, 3)
c3_b = rand(96)
c4_w = rand(64, 96, 3, 3)
c4_b = rand(64)
c5_w = rand(64, 64, 3, 3)
c5_b = rand(64)
f1_w = rand(1024, 64)
f1_b = rand(1024)
f2_w = rand(1024, 1024)
f2_b = rand(1024)
f3_w = rand(256, 1024)
f3_b = rand(256)

ARRAYS = {
    "input": input_,
    "c1_w": c1_w, "c1_b": c1_b, "c2_w": c2_w, "c2_b": c2_b,
    "c3_w": c3_w, "c3_b": c3_b, "c4_w": c4_w, "c4_b": c4_b,
    "c5_w": c5_w, "c5_b": c5_b,
    "f1_w": f1_w, "f1_b": f1_b, "f2_w": f2_w, "f2_b": f2_b,
    "f3_w": f3_w, "f3_b": f3_b,
}


def relu(x):
    return np.maximum(x, 0.0)


def conv4d(x, w, b, p, s):
    """fp64 reference; zero padding p, stride s, bias added at the end."""
    from numpy.lib.stride_tricks import sliding_window_view
    N, Cin, H, W = x.shape
    Cout, _, K1, K2 = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
    win = sliding_window_view(xp, (K1, K2), axis=(2, 3))
    win = win[:, :, ::s, ::s]
    out = np.einsum("ncijkl,ockl->noij",
                    win.astype(np.float64), w.astype(np.float64))
    return out + b.astype(np.float64).reshape(1, Cout, 1, 1)


def maxpool4d(x, k, s):
    """Window max; Coq Maxpool4d_pad semantics: safe_idx out-of-range reads
    as zero and maxl starts at 0 -> max(0, window over zero-extended input).
    ho = (h - k)//s + 1 (nat division), so the last window may hang past the
    input edge; extend with zeros to (ho-1)*s + k."""
    from numpy.lib.stride_tricks import sliding_window_view
    N, C, H, W = x.shape
    # nat-style truncating division (Coq/OCaml), not Python floor division
    ho = int(np.trunc((H - k) / s)) + 1
    wo = int(np.trunc((W - k) / s)) + 1
    xe = np.zeros((N, C, max((ho - 1) * s + k, H), max((wo - 1) * s + k, W)),
                  dtype=x.dtype)
    xe[:, :, :H, :W] = x
    win = sliding_window_view(xe, (k, k), axis=(2, 3))[:, :, ::s, ::s]
    return np.maximum(win.max(axis=(4, 5)), 0.0)


def softmax(x):
    # No max-subtraction: matches the Coq semantics exactly.
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def ref_layers():
    """Return ordered [(kernel_out_name, reference_array)] per layer."""
    c1 = conv4d(input_, c1_w, c1_b, 2, 4)
    r1 = relu(c1)
    p1 = maxpool4d(r1, 3, 2)
    c2 = conv4d(p1, c2_w, c2_b, 2, 1)
    r2 = relu(c2)
    p2 = maxpool4d(r2, 3, 2)
    c3 = conv4d(p2, c3_w, c3_b, 1, 1)
    r3 = relu(c3)
    c4 = conv4d(r3, c4_w, c4_b, 1, 1)
    r4 = relu(c4)
    c5 = conv4d(r4, c5_w, c5_b, 1, 1)
    r5 = relu(c5)
    p5 = maxpool4d(r5, 3, 2)
    flat = p5.reshape(-1)
    fc1o = flat @ f1_w.astype(np.float64).T + f1_b
    fc1r = relu(fc1o)
    fc2o = fc1r @ f2_w.astype(np.float64).T + f2_b
    fc2r = relu(fc2o)
    fc3o = fc2r @ f3_w.astype(np.float64).T + f3_b
    output = softmax(fc3o)
    return [
        ("conv1", "c1", c1.reshape(-1)),
        ("relu1", "r1", r1.reshape(-1)),
        ("pool1", "p1", p1.reshape(-1)),
        ("conv2", "c2", c2.reshape(-1)),
        ("relu2", "r2", r2.reshape(-1)),
        ("pool2", "p2", p2.reshape(-1)),
        ("conv3", "c3", c3.reshape(-1)),
        ("relu3", "r3", r3.reshape(-1)),
        ("conv4", "c4", c4.reshape(-1)),
        ("relu4", "r4", r4.reshape(-1)),
        ("conv5", "c5", c5.reshape(-1)),
        ("relu5", "r5", r5.reshape(-1)),
        ("pool5", "p5", p5.reshape(-1)),
        ("flatten", "flat", flat),
        ("fc1", "fc1o", fc1o.reshape(-1)),
        ("relu6", "fc1r", fc1r.reshape(-1)),
        ("fc2", "fc2o", fc2o.reshape(-1)),
        ("relu7", "fc2r", fc2r.reshape(-1)),
        ("fc3", "fc3o", fc3o.reshape(-1)),
        ("softmax", "output", output.reshape(-1)),
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
        print(f"v4_host FAIL rc={r.returncode}\n{r.stderr}")
        return 1
    print(f"chain wall time: {dt:.2f}s")

    npass = 0
    layers = ref_layers()
    for kname, pname, ref in layers:
        got = np.fromfile(os.path.join(DATA, f"v4out_{kname}_{pname}.bin"),
                          dtype="<f4")
        if got.shape != ref.shape:
            print(f"[{kname}] FAIL shape {got.shape} vs ref {ref.shape}")
            continue
        diff = np.abs(got.astype(np.float64) - ref)
        max_diff = float(diff.max()) if diff.size else 0.0
        ok = bool(np.isfinite(got).all()) and max_diff <= 1e-5
        npass += ok
        print(f"[{kname:8s}] {'PASS' if ok else 'FAIL'} n={got.size:5d} "
              f"max_abs_diff={max_diff:.3e}")
    final_diff = np.abs(
        np.fromfile(os.path.join(DATA, "v4out_softmax_output.bin"),
                    dtype="<f4") - layers[-1][2]).max()
    print(f"V4 AlexNet: {npass}/{len(layers)} layers PASS; "
          f"final softmax max_abs_diff={final_diff:.3e} (threshold 1e-5)")
    return 0 if npass == len(layers) else 1


if __name__ == "__main__":
    sys.exit(main())
