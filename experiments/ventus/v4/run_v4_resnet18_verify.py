#!/usr/bin/env python3
"""V4 ResNet-18 offline re-verification: the spike dumps are complete (two
segments), so spike is not re-run.
Segment 1 (48 kernels, s_conv..l3b2_bn2) outputs are data_resnet18/v4out_*;
segment 2 (21 kernels, l3b2_add..softmax) outputs are
data_resnet18_b/v4out_*; the chain continues across segments through the
actual spike-produced l3b2_bn2o / l3b1_r2o buffers (confirmed
byte-identical with cmp).
The reference and RNG protocol are fully reused from run_v4_resnet18.py
(importing it rebuilds the same parameters).
"""
import os
import sys

import numpy as np

import run_v4_resnet18 as m

V4 = os.path.dirname(os.path.abspath(__file__))
DIRS = [os.path.join(V4, "data_resnet18"), os.path.join(V4, "data_resnet18_b")]


def find_dump(kname, pname):
    fn = f"v4out_{kname}_{pname}.bin"
    for d in DIRS:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    return None


def main():
    npass = 0
    layers = m.ref_forward()
    for kname, pname, ref in layers:
        ref = np.asarray(ref, dtype=np.float64).reshape(-1)
        p = find_dump(kname, pname)
        if p is None:
            print(f"[{kname:9s}] FAIL missing dump")
            continue
        got = np.fromfile(p, dtype="<f4")
        if got.shape != ref.shape:
            print(f"[{kname:9s}] FAIL shape {got.shape} vs ref {ref.shape}")
            continue
        max_diff = float(np.abs(got.astype(np.float64) - ref).max())
        ok = bool(np.isfinite(got).all()) and max_diff <= 1e-5
        npass += ok
        seg = "b" if "data_resnet18_b" in p else "a"
        print(f"[{kname:9s}] {'PASS' if ok else 'FAIL'} seg={seg} "
              f"n={got.size:5d} max_abs_diff={max_diff:.3e}")
    final_diff = float(np.abs(
        np.fromfile(find_dump("softmax", "output"), dtype="<f4")
        - np.asarray(layers[-1][2], dtype=np.float64).reshape(-1)).max())
    print(f"V4 ResNet-18 (offline re-verify): {npass}/{len(layers)} layers PASS; "
          f"final softmax max_abs_diff={final_diff:.3e} (threshold 1e-5)")
    return 0 if npass == len(layers) else 1


if __name__ == "__main__":
    sys.exit(main())
