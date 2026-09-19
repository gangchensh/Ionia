#!/usr/bin/env python3
"""Demo 2: small Conv2d (Conv4d p=1, k=3, s=1), 1x1x8x8 -> 1x2x8x8.

Functional form (Ionia, CNN_OP.Conv4d): each output cell is
    add (sum (mkvSeq (fun c => sum (mkvSeq (fun k1 => sum (mkvSeq (fun k2 =>
        mul (padded_input ...) w|[nu]|[c]|[k1]|[k2]))))))) bias|[nu]
i.e. a 1x3x3 reduction per output element.  Rewriting rules: AmkvSeq x4,
A_add_post, C_sum_post_acc (A_sum family).
Proofs: Ops_Conv.A_Conv4d (general Conv4d <-> Conv4d_comm),
Bridge_AlexNet.Conv4d_sem_single (Conv4d <-> Conv4d_ast_comm).  The
materialised-padding variant Conv4d_p1_k3_comm used by the pipeline is the
same computation; see the honesty note in demo/README.md.

Fixed seed: 202608272.  NumPy reference uses the kernel's accumulation
order (c -> k1 -> k2, sequential float32).
"""

import numpy as np

from util import wrap_compile_run, read_output, report

NETWORK = "conv_demo"


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
                                s = np.float32(s + np.float32(
                                    padded[n, ci, i + k1, j + k2]
                                    * w[co, ci, k1, k2]))
                    out[n, co, i, j] = np.float32(s + b[co])
    return out


def main():
    rng = np.random.default_rng(202608272)
    x = rng.standard_normal((1, 1, 8, 8)).astype(np.float32)
    w = rng.standard_normal((2, 1, 3, 3)).astype(np.float32)
    b = rng.standard_normal((2,)).astype(np.float32)

    wrap_compile_run(NETWORK, [("input", x), ("conv_w", w), ("conv_b", b)])
    got = read_output(NETWORK, (1, 2, 8, 8))

    expected = conv2d_pad1_k3(x, w, b)
    print("NumPy ref [0,:, :2, :2]  :\n", expected[0, :, :2, :2])
    print("CUDA output[0,:, :2, :2] :\n", got[0, :, :2, :2])
    report(NETWORK, got, expected)


if __name__ == "__main__":
    main()
