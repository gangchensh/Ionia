#!/usr/bin/env python3
"""Demo 1: elementwise ReLU on a 1x2x4x4 tensor.

Functional form (Ionia, Ops_Elementwise.ReLU4d):
    mkvSeq (fun b => ... if leExp x|[b]|[c]|[i]|[j] zero then zero else x|[...])
Rewriting rules behind the comm: AmkvSeq x4 (one per dimension), Aif1, A_num.
Proofs: Ops_Elementwise.A_ReLU4d (route-A Leibniz equality),
CNN_Examples_Base_Ops.ReLU4d_equiv_sem (the CG-world variant used here).

Fixed seed: 202608271.
"""

import numpy as np

from util import wrap_compile_run, read_output, report

NETWORK = "relu_demo"
SHAPE = (1, 2, 4, 4)


def main():
    rng = np.random.default_rng(202608271)
    x = rng.standard_normal(SHAPE).astype(np.float32)

    wrap_compile_run(NETWORK, [("input", x)])
    got = read_output(NETWORK, SHAPE)

    expected = np.maximum(x, np.float32(0.0))
    print("input[0,0] :", x[0, 0])
    print("NumPy ref  :", expected[0, 0])
    print("CUDA output:", got[0, 0])
    report(NETWORK, got, expected)


if __name__ == "__main__":
    main()
