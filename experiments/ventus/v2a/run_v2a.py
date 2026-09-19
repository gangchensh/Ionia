#!/usr/bin/env python3
"""Ionia Ventus V2a validation driver: for each operator in
   kernels/manifest.txt, generate deterministic inputs, run the generated
   OpenCL kernel on spike via v2a_host, and compare elementwise against a
   NumPy reference.

   Tolerances: relu/add/max are exact operations (bit-exact check);
   sum/conv use atol against an fp64 reference, since the tree-reduction /
   accumulation order differs from NumPy.
"""
import os
import subprocess
import sys
import time

import numpy as np

V2A = os.path.dirname(os.path.abspath(__file__))
HOST = os.path.join(V2A, "build", "v2a_host")
KERNELS = os.path.join(V2A, "kernels")
DATA = os.path.join(V2A, "data")


def gen_inputs(name, ins):
    rng = np.random.default_rng(20260831)
    arrays = {}
    for pname, count in ins:
        if name == "conv2d" and pname == "b":
            arr = rng.standard_normal(count).astype(np.float32) * 0.5
        else:
            arr = rng.standard_normal(count).astype(np.float32)
        arrays[pname] = arr
    return arrays


def reference(name, a):
    if name == "relu1d":
        return np.maximum(a["x"], np.float32(0))
    if name == "add1d":
        return (a["x"].astype(np.float64) + a["y"].astype(np.float64)).astype(np.float32)
    if name == "sum256":
        return np.array([a["x"].astype(np.float64).sum()], dtype=np.float32)
    if name == "sum4096":
        return a["x"].reshape(16, 256).astype(np.float64).sum(axis=1).astype(np.float32)
    if name == "max256":
        return np.array([a["x"].max()], dtype=np.float32)
    if name == "conv2d":
        F, C, H, W, K, P = 4, 2, 8, 8, 3, 1
        x = a["x"].reshape(C, H, W).astype(np.float64)
        w = a["w"].reshape(F, C, K, K).astype(np.float64)
        b = a["b"].astype(np.float64)
        xp = np.pad(x, ((0, 0), (P, P), (P, P)))
        out = np.zeros((F, H, W), dtype=np.float64)
        for f in range(F):
            for i in range(H):
                for j in range(W):
                    acc = b[f]
                    for c in range(C):
                        acc += (xp[c, i:i + K, j:j + K] * w[f, c]).sum()
                    out[f, i, j] = acc
        return out.reshape(-1).astype(np.float32)
    raise ValueError(name)


EXACT = {"relu1d", "add1d", "max256"}
ATOL = {"sum256": 1e-3, "sum4096": 1e-2, "conv2d": 1e-3}


def run_op(spec):
    name, gws, lws, ins, outs = spec
    arrays = gen_inputs(name, ins)
    args = [HOST, os.path.join(KERNELS, name + ".cl"), name, str(gws), str(lws)]
    out_name, out_count = outs[0]
    out_bin = os.path.join(DATA, f"{name}_{out_name}.bin")
    args += [out_bin, str(out_count)]
    for pname, count in ins:
        p = os.path.join(DATA, f"{name}_{pname}.bin")
        arrays[pname].astype("<f4").tofile(p)
        args += [p, str(count)]
    t0 = time.time()
    r = subprocess.run(args, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"[{name}] HOST FAIL rc={r.returncode}\n{r.stdout}\n{r.stderr}")
        return False
    got = np.fromfile(out_bin, dtype="<f4")
    ref = reference(name, arrays)
    if got.shape != ref.shape:
        print(f"[{name}] FAIL shape {got.shape} vs ref {ref.shape}")
        return False
    diff = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    max_diff = float(diff.max()) if diff.size else 0.0
    if name in EXACT:
        ok = np.array_equal(got, ref)
        crit = "bit-exact"
    else:
        ok = bool(np.isfinite(got).all()) and max_diff <= ATOL[name]
        crit = f"atol={ATOL[name]}"
    status = "PASS" if ok else "FAIL"
    print(f"[{name}] {status} ({crit}) n={got.size} max_abs_diff={max_diff:.3e} "
          f"wall={dt:.2f}s")
    return ok


def main():
    specs = []
    with open(os.path.join(KERNELS, "manifest.txt")) as f:
        for line in f:
            parts = line.split()
            name, gws, lws = parts[0], int(parts[1]), int(parts[2])
            ins, outs = [], []
            for tok in parts[3:]:
                io, pname, cnt = tok.split(":")
                (ins if io == "in" else outs).append((pname, int(cnt)))
            specs.append((name, gws, lws, ins, outs))
    os.makedirs(DATA, exist_ok=True)
    results = [run_op(s) for s in specs]
    npass = sum(1 for r in results if r)
    print(f"V2a total: {npass}/{len(results)} PASS")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
