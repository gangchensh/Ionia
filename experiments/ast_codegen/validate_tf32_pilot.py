#!/usr/bin/env python3
"""TF32 pilot: data generation + FP64 reference + error-vs-bound analysis.

Data regime mirrors validate_cuda_resnet18.py: np.random.seed(18), weight
scale 1e-2, standard normal input.  Writes input/weight .bin for
tf32_conv_pilot.cu, reads back fp32/tf32 kernel outputs, and reports per
shape (batch=1, 3x3 stride 1 pad 1 conv):

- max abs / rel error of both kernels vs the float64 reference
- theoretical reduction bound  |err| <= (2*u_tf32 + n*u_fp32) * sum|a_k b_k|
  (u_tf32 = 2^-11 coefficient rounding, u_fp32 = 2^-24 accumulation,
  n = K = CI*9), evaluated per element; reports measured/bound ratio.

Usage: python validate_tf32_pilot.py [--gen-only]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DIR = ROOT / "tf32_pilot"
SHAPES = [
    ("l1_64ch_56x56", 64, 64, 56),
    ("l2_128ch_28x28", 128, 128, 28),
    ("l4_512ch_7x7", 512, 512, 7),
]

U_TF32 = 2.0**-11  # tf32 coefficient rounding (10 explicit mantissa bits)
U_FP32 = 2.0**-24


def gen():
    DIR.mkdir(exist_ok=True)
    rng = np.random.RandomState(18)
    for name, ci, co, hw in SHAPES:
        x = rng.randn(ci, hw, hw).astype(np.float32)
        w = (rng.randn(co, ci, 3, 3) * 1e-2).astype(np.float32)
        x.tofile(DIR / f"{name}_input.bin")
        w.tofile(DIR / f"{name}_weight.bin")
        print(f"gen {name}: x{x.shape} w{w.shape}")


def conv64(x, w):
    """float64 reference via stride-trick im2col + matmul. x:(CI,H,W) w:(CO,CI,3,3)."""
    ci, h, wd = x.shape
    xp = np.pad(x.astype(np.float64), ((0, 0), (1, 1), (1, 1)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(1, 2))
    # win: (CI, H, W, 3, 3) -> cols (CI*9, H*W) with k = ci*9 + kh*3 + kw
    cols = win.transpose(0, 3, 4, 1, 2).reshape(ci * 9, h * wd)
    wm = w.astype(np.float64).reshape(w.shape[0], -1)
    return (wm @ cols).reshape(w.shape[0], h, wd), wm, cols


def analyze(name, ci, co, hw):
    x = np.fromfile(DIR / f"{name}_input.bin", np.float32).reshape(ci, hw, hw)
    w = np.fromfile(DIR / f"{name}_weight.bin", np.float32).reshape(co, ci, 3, 3)
    ref, wm, cols = conv64(x, w)
    n = ci * 9
    # per-element reduction bound scale S = |W| @ |Xcol|
    S = (np.abs(wm) @ np.abs(cols)).reshape(co, hw, hw)
    refn = np.abs(ref)
    refnorm = np.linalg.norm(ref.ravel())
    for impl, u_coef in (("fp32", 0.0), ("tf32", U_TF32)):
        out = np.fromfile(DIR / f"{name}_out_{impl}.bin", np.float32).reshape(
            co, hw, hw
        )
        err = np.abs(out.astype(np.float64) - ref)
        bound = (2.0 * u_coef + n * U_FP32) * S + U_FP32 * refn
        rel = err / np.maximum(refn, 1e-30)
        print(
            f"{name} [{impl:5s}] maxabs={err.max():.3e} "
            f"normrel={np.linalg.norm(err.ravel()) / refnorm:.3e} "
            f"maxrel_elem={rel.max():.3e} "
            f"bound_max={bound.max():.3e} "
            f"err<=bound: {bool((err <= bound).all())} "
            f"max(err/bound)={(err / np.maximum(bound, 1e-300)).max():.3e}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-only", action="store_true")
    a = ap.parse_args()
    gen()
    if a.gen_only:
        sys.exit(0)
    for name, ci, co, hw in SHAPES:
        analyze(name, ci, co, hw)
