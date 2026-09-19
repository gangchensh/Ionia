#!/usr/bin/env python3
"""End-to-end numerical validation of the generated AlexNet CUDA code.

Pipeline:
  1. comm_to_cuda_alexnet (OCaml) prints alexnet_test_cuda_output.cu from the
     extracted alexnet_comm AST (pattern matchers + Conv/ReLU fusion).
  2. wrap_generated_cuda.py wraps the body into a compilable alexnet_test.cu.
  3. nvcc compiles it; random float32 inputs/weights are staged as .bin files.
  4. The executable is run; its output is compared element-wise against a
     NumPy reference that mirrors the Coq operator semantics (same reduction
     order, no softmax max-subtraction).
  5. GPU kernel time (TIME), staging+H2D time (MEMTIME_H2D) and D2H time
     (MEMTIME_D2H) are parsed from the executable output.
  6. The sequential C version (comm_to_c_alexnet) is built and timed for a
     CPU-vs-GPU comparison.

Set NVCC to the nvcc binary if it is not on PATH.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
NVCC = os.environ.get("NVCC", "nvcc")

# Same data regime as validate_alexnet.py (CPU C validation).
np.random.seed(123)
scale = 1e-3

INPUT   = (scale * np.random.randn(1, 3, 8, 8)).astype(np.float32)
CONV1_W = (scale * np.random.randn(8, 3, 3, 3)).astype(np.float32)
CONV1_B = (scale * np.random.randn(8)).astype(np.float32)
CONV2_W = (scale * np.random.randn(16, 8, 3, 3)).astype(np.float32)
CONV2_B = (scale * np.random.randn(16)).astype(np.float32)
FC1_W   = (scale * np.random.randn(16, 64)).astype(np.float32)
FC1_B   = (scale * np.random.randn(16)).astype(np.float32)
FC2_W   = (scale * np.random.randn(10, 16)).astype(np.float32)
FC2_B   = (scale * np.random.randn(10)).astype(np.float32)

TENSORS = {
    "input": INPUT,
    "conv1_w": CONV1_W,
    "conv1_b": CONV1_B,
    "conv2_w": CONV2_W,
    "conv2_b": CONV2_B,
    "fc1_w": FC1_W,
    "fc1_b": FC1_B,
    "fc2_w": FC2_W,
    "fc2_b": FC2_B,
}


def relu(x):
    return np.maximum(x, np.float32(0.0))


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
                                s += padded[n, ci, i + k1, j + k2] * w[co, ci, k1, k2]
                    out[n, co, i, j] = s + b[co]
    return out


def maxpool2d_k2(x):
    N, C, H, W = x.shape
    out = np.zeros((N, C, H // 2, W // 2), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            for i in range(H // 2):
                for j in range(W // 2):
                    out[n, c, i, j] = np.max(x[n, c, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2])
    return out


def linear(x, w, b):
    return x @ w.T + b


def softmax(x):
    # No max-subtraction: matches the Coq/kernel semantics exactly.
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def numpy_reference():
    x = relu(conv2d_pad1_k3(INPUT, CONV1_W, CONV1_B))
    x = maxpool2d_k2(x)
    x = relu(conv2d_pad1_k3(x, CONV2_W, CONV2_B))
    x = maxpool2d_k2(x)
    x = x.reshape(1, -1)
    x = linear(x, FC1_W, FC1_B)
    x = linear(x, FC2_W, FC2_B)
    return softmax(x)


def run(cmd, **kw):
    subprocess.run(cmd, cwd=ROOT, check=True, **kw)


def main():
    name = "alexnet_test"

    # 1-2. Regenerate the CUDA body and wrap it.
    run(["./comm_to_cuda_alexnet"])
    run([sys.executable, "wrap_generated_cuda.py", name,
         f"{name}_cuda_output.cu", f"{name}.cu"])

    # 3. Compile.
    nvcc_flags = os.environ.get("NVCC_FLAGS", "-O2").split()
    run([NVCC, *nvcc_flags, "-o", name, f"{name}.cu"])

    # 4. Stage inputs.
    for var_name, arr in TENSORS.items():
        (ROOT / f"{name}_{var_name}.bin").write_bytes(arr.tobytes())

    # 5. Run.
    result = subprocess.run([f"./{name}"], cwd=ROOT, check=True,
                            capture_output=True, text=True)
    print(result.stdout, end="")

    output = np.frombuffer((ROOT / f"{name}_output.bin").read_bytes(),
                           dtype=np.float32).reshape(1, 10)
    expected = numpy_reference()

    max_abs = float(np.max(np.abs(output - expected)))
    max_rel = float(np.max(np.abs(output - expected) / np.maximum(np.abs(expected), 1e-20)))
    print("NumPy reference:", expected[0])
    print("CUDA output:    ", output[0])
    print(f"max abs error: {max_abs:.3e}")
    print(f"max rel error: {max_rel:.3e}")
    ok = np.allclose(output, expected, atol=1e-5, rtol=1e-5)
    print(f"{name}: {'PASSED' if ok else 'FAILED'} (atol=1e-5, rtol=1e-5)")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
