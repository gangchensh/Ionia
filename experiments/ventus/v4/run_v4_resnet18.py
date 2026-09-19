#!/usr/bin/env python3
"""Ionia Ventus V4 validation driver: the full ResNet-18 (reduced sizes).

Same RNG protocol as experiments/ast_codegen/validate_resnet18.py
(seed=18, scale=1e-2, eps=1e-5, same sampling order); only the shapes are
scaled down:
  input 3x40x40, channels /4 (16/32/64/128), fc 256.
Layer sizes: stem conv k7p3s2 -> 20x20; pool k3s2p1 -> 10x10;
  l1 10x10, l2 5x5, l3 3x3, l4 2x2; GAP -> 128; fc 256.
BN uses the given-mean/variance parameterized semantics (matching
validate_resnet18.py); Coq Maxpool4d_pad zero padding + maxl zero init
(the pool input is ReLU output, equivalent to an ordinary max with -inf
padding).
All references computed in fp64. Final check: softmax max_abs_diff <= 1e-5.
"""
import os
import subprocess
import sys
import time

import numpy as np

V4 = os.path.dirname(os.path.abspath(__file__))
HOST = os.path.join(V4, "build", "v4_host")
KERNELS = os.path.join(V4, "kernels_resnet18")
DATA = os.path.join(V4, "data_resnet18")

# ---- inputs/weights: same RNG protocol as validate_resnet18.py (M10) ----
np.random.seed(18)
scale = 1e-2
eps = 1e-5


def rand(*shape):
    return (scale * np.random.randn(*shape)).astype(np.float32)


input_ = rand(1, 3, 40, 40)

conv1_w = rand(16, 3, 7, 7)
conv1_b = rand(16)
bn1_mean = rand(16)
bn1_var = rand(16) ** 2 + 1e-4
bn1_gamma = rand(16)
bn1_beta = rand(16)

# (prefix, inch, outch, m_out, proj)
blocks = [
    ("l1b1", 16, 16, 10, False),
    ("l1b2", 16, 16, 10, False),
    ("l2b1", 16, 32, 5, True),
    ("l2b2", 32, 32, 5, False),
    ("l3b1", 32, 64, 3, True),
    ("l3b2", 64, 64, 3, False),
    ("l4b1", 64, 128, 2, True),
    ("l4b2", 128, 128, 2, False),
]

params = {}
for prefix, inch, outch, m, proj in blocks:
    params[prefix] = {
        "inch": inch, "outch": outch, "m": m, "proj": proj,
        "w1": rand(outch, inch, 3, 3), "b1": rand(outch),
        "w2": rand(outch, outch, 3, 3), "b2": rand(outch),
        "bn1": (rand(outch), rand(outch) ** 2 + 1e-4, rand(outch), rand(outch)),
        "bn2": (rand(outch), rand(outch) ** 2 + 1e-4, rand(outch), rand(outch)),
    }
    if proj:
        params[prefix]["shortcut_w"] = rand(outch, inch, 1, 1)
        params[prefix]["shortcut_b"] = rand(outch)
        params[prefix]["shortcut_bn"] = (
            rand(outch), rand(outch) ** 2 + 1e-4, rand(outch), rand(outch))

fc_w = rand(256, 128)
fc_b = rand(256)


def stage_arrays():
    """Map python-side tensors to the flat buffer names of the manifest."""
    a = {"input": input_, "sc1_w": conv1_w, "sc1_b": conv1_b,
         "s_bnm": bn1_mean, "s_bnv": bn1_var,
         "s_bng": bn1_gamma, "s_bnb": bn1_beta}
    for prefix, inch, outch, m, proj in blocks:
        p = params[prefix]
        a[prefix + "_w1"] = p["w1"]
        a[prefix + "_b1"] = p["b1"]
        a[prefix + "_w2"] = p["w2"]
        a[prefix + "_b2"] = p["b2"]
        for i, s in enumerate(["m", "v", "g", "b"]):
            a[prefix + "_bn1" + s] = p["bn1"][i]
            a[prefix + "_bn2" + s] = p["bn2"][i]
        if proj:
            a[prefix + "_scw"] = p["shortcut_w"]
            a[prefix + "_scb"] = p["shortcut_b"]
            for i, s in enumerate(["m", "v", "g", "b"]):
                a[prefix + "_sbn" + s] = p["shortcut_bn"][i]
    a["fc_w"] = fc_w
    a["fc_b"] = fc_b
    return a


# ---- fp64 reference forward (mirrors the Coq/Kernel semantics) ----
def relu(x):
    return np.maximum(x, 0.0)


def batch_norm(x, mean, var, gamma, beta):
    return ((x - mean.reshape(1, -1, 1, 1))
            / np.sqrt(var.reshape(1, -1, 1, 1) + eps)
            * gamma.reshape(1, -1, 1, 1) + beta.reshape(1, -1, 1, 1))


def conv2d(x, w, b, padding, stride):
    N, Cin, H, W = x.shape
    Cout, _, K, _ = w.shape
    p, s = padding, stride
    Ho = (H + 2 * p - K) // s + 1
    Wo = (W + 2 * p - K) // s + 1
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
    patches = np.lib.stride_tricks.as_strided(
        xp,
        shape=(N, Cin, Ho, Wo, K, K),
        strides=(xp.strides[0], xp.strides[1], xp.strides[2] * s,
                 xp.strides[3] * s, xp.strides[2], xp.strides[3]),
        writeable=False)
    out = np.tensordot(patches.astype(np.float64), w.astype(np.float64),
                       axes=([1, 4, 5], [1, 2, 3]))
    out = np.transpose(out, (0, 3, 1, 2))
    return out + b.astype(np.float64).reshape(1, Cout, 1, 1)


def maxpool2d(x, kernel=3, padding=1, stride=2):
    """Zero padding + max with 0 init (Coq Maxpool4d_pad / maxl)."""
    N, C, H, W = x.shape
    Ho = (H + 2 * padding - kernel) // stride + 1
    Wo = (W + 2 * padding - kernel) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                mode="constant")
    patches = np.lib.stride_tricks.as_strided(
        xp,
        shape=(N, C, Ho, Wo, kernel, kernel),
        strides=(xp.strides[0], xp.strides[1], xp.strides[2] * stride,
                 xp.strides[3] * stride, xp.strides[2], xp.strides[3]),
        writeable=False)
    return np.maximum(patches.max(axis=(4, 5)), 0.0)


def ref_forward():
    """Return (ordered per-kernel [(kname, pname, ref)], final)."""
    layers = []

    def bn_params(pfx, bn):
        return dict(zip(["m", "v", "g", "b"], bn))

    x = conv2d(input_, conv1_w, conv1_b, padding=3, stride=2)
    layers.append(("s_conv", "s_c1o", x))
    x = batch_norm(x, bn1_mean, bn1_var, bn1_gamma, bn1_beta)
    layers.append(("s_bn", "s_bn1o", x))
    x = relu(x)
    layers.append(("s_relu", "s_r1o", x))
    x = maxpool2d(x)
    layers.append(("s_pool", "spool", x))
    for prefix, inch, outch, m, proj in blocks:
        p = params[prefix]
        identity = x
        out = conv2d(x, p["w1"], p["b1"], padding=1,
                     stride=(2 if proj else 1))
        layers.append((prefix + "_c1", prefix + "_c1o", out))
        out = batch_norm(out, *p["bn1"])
        layers.append((prefix + "_bn1", prefix + "_bn1o", out))
        out = relu(out)
        layers.append((prefix + "_r1", prefix + "_r1o", out))
        out = conv2d(out, p["w2"], p["b2"], padding=1, stride=1)
        layers.append((prefix + "_c2", prefix + "_c2o", out))
        out = batch_norm(out, *p["bn2"])
        layers.append((prefix + "_bn2", prefix + "_bn2o", out))
        if proj:
            identity = conv2d(x, p["shortcut_w"], p["shortcut_b"],
                              padding=0, stride=2)
            layers.append((prefix + "_sc", prefix + "_sco", identity))
            identity = batch_norm(identity, *p["shortcut_bn"])
            layers.append((prefix + "_sbn", prefix + "_sbno", identity))
        out = out + identity
        layers.append((prefix + "_add", prefix + "_addo", out))
        out = relu(out)
        layers.append((prefix + "_r2", prefix + "_r2o", out))
        x = out
    gap = x.mean(axis=(2, 3)).reshape(-1)
    layers.append(("gap", "gapo", gap))
    fco = gap @ fc_w.astype(np.float64).T + fc_b.astype(np.float64)
    layers.append(("fc", "fco", fco))
    e = np.exp(fco)  # no max-subtraction: matches the Coq semantics
    output = e / e.sum()
    layers.append(("softmax", "output", output))
    return layers


def main():
    os.makedirs(DATA, exist_ok=True)
    for name, arr in stage_arrays().items():
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
    layers = ref_forward()
    for kname, pname, ref in layers:
        ref = np.asarray(ref, dtype=np.float64).reshape(-1)
        got = np.fromfile(os.path.join(DATA, f"v4out_{kname}_{pname}.bin"),
                          dtype="<f4")
        if got.shape != ref.shape:
            print(f"[{kname}] FAIL shape {got.shape} vs ref {ref.shape}")
            continue
        diff = np.abs(got.astype(np.float64) - ref)
        max_diff = float(diff.max()) if diff.size else 0.0
        ok = bool(np.isfinite(got).all()) and max_diff <= 1e-5
        npass += ok
        print(f"[{kname:9s}] {'PASS' if ok else 'FAIL'} n={got.size:5d} "
              f"max_abs_diff={max_diff:.3e}")
    final_diff = np.abs(
        np.fromfile(os.path.join(DATA, "v4out_softmax_output.bin"),
                    dtype="<f4")
        - np.asarray(layers[-1][2], dtype=np.float64).reshape(-1)).max()
    print(f"V4 ResNet-18: {npass}/{len(layers)} layers PASS; "
          f"final softmax max_abs_diff={final_diff:.3e} (threshold 1e-5)")
    return 0 if npass == len(layers) else 1


if __name__ == "__main__":
    sys.exit(main())
