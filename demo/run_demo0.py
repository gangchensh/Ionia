#!/usr/bin/env python3
"""Demo 0: dot product  y = dot(x, w) + b  (Linear2d 1x8 -> 1x1).

Functional form (Ionia, CNN_OP.Linear2d specialised to one output unit):
    out = add (sum (mkvSeq (fun i => mul x|[i] w|[i]))) b
Rewriting rules behind the comm: AmkvSeq (A10) expands mkvSeq into loops,
A_sum/C_sum_post_acc (A13 family) turns the sum into a sequential
accumulator loop, A_mul (A4) handles the pointwise multiply.
Proofs: Operators_Sem_Equiv.Linear2d_sem, Bridge_AlexNet.Linear2d_sem_single.

Fixed seed: 202608270.
"""

import numpy as np

from util import wrap_compile_run, read_output, report

NETWORK = "dot_demo"


def main():
    rng = np.random.default_rng(202608270)
    x = rng.standard_normal((1, 8)).astype(np.float32)
    w = rng.standard_normal((1, 8)).astype(np.float32)
    b = rng.standard_normal((1,)).astype(np.float32)

    wrap_compile_run(NETWORK, [("x", x), ("w", w), ("b", b)])
    got = read_output(NETWORK, (1, 1))

    # NumPy reference in the kernel's accumulation order (i = 0..7, then +b),
    # mirroring the Coq functional semantics.
    acc = np.float32(0.0)
    for i in range(8):
        acc = np.float32(acc + np.float32(x[0, i] * w[0, i]))
    expected = np.float32(acc + b[0]).reshape(1, 1)

    print("x          :", x[0])
    print("w          :", w[0])
    print("b          :", b)
    print(f"np.dot(x,w)+b: {np.dot(x[0], w[0]) + b[0]:.8e}   (NumPy, pairwise summation)")
    print(f"sequential : {expected[0, 0]:.8e}   (kernel accumulation order)")
    print(f"CUDA output: {got[0, 0]:.8e}")
    report(NETWORK, got, expected)


if __name__ == "__main__":
    main()
