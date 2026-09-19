#!/usr/bin/env python3
"""Demo 3: end-to-end network (small CIFAR-style AlexNet) on GPU.

Reuses the existing pipeline in experiments/ast_codegen/ unchanged:
  alexnet_comm (CNN_Examples_AlexNet.v, extracted alexnet_comm.ml)
  -> transform_to_symbolic.py -> alexnet_sym.ml
  -> comm_to_cuda_alexnet.ml (pattern matchers + Conv/ReLU fusion)
  -> wrap_generated_cuda.py -> nvcc -> run -> NumPy compare.
This script just builds the OCaml printer and runs
experiments/ast_codegen/validate_cuda_alexnet.py (seed 123, scale 1e-3,
float32) with NVCC resolved.

Network: input 1x3x8x8
  -> Conv4d(p1,k3) 3->8 + ReLU -> MaxPool(k2)     1x8x4x4
  -> Conv4d(p1,k3) 8->16 + ReLU -> MaxPool(k2)    1x16x2x2
  -> Flatten 1x64 -> Linear 64->16 -> Linear 16->10 -> Softmax 1x10
End-to-end theorem (full-size AlexNet): AlexNet_DPIA.A_AlexNet;
per-operator bridge theorems: Bridge_AlexNet.v (bridge_ReLU4d_buf,
Conv4d_sem_single, Linear2d_sem_single, ...).
"""

import os
import subprocess
import sys
from pathlib import Path

AST = Path(__file__).resolve().parent.parent / "experiments" / "ast_codegen"


def find_nvcc():
    for cand in (os.environ.get("NVCC"), None, "/tmp/cuda-12.9/bin/nvcc"):
        if cand and Path(cand).exists():
            return cand
    import shutil
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    sys.exit("nvcc not found; set NVCC (see demo/README.md §2).")


def main():
    subprocess.run(["make", "comm_to_cuda_alexnet"], cwd=AST, check=True)
    env = dict(os.environ, NVCC=find_nvcc())
    subprocess.run([sys.executable, "validate_cuda_alexnet.py"],
                   cwd=AST, env=env, check=True)


if __name__ == "__main__":
    main()
