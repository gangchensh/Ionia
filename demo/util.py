#!/usr/bin/env python3
"""Shared helpers for the Ionia tutorial demos (demo/README.md).

Pipeline per demo: wrap the generated CUDA body with
experiments/ast_codegen/wrap_generated_cuda.py, compile with nvcc, run the
executable in demo/build/, and compare the output against a NumPy reference.

nvcc resolution order: $NVCC, PATH, /tmp/cuda-12.9/bin/nvcc (a user-level
CUDA 12.9 toolkit location; override via $NVCC if your install differs).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

DEMO_ROOT = Path(__file__).resolve().parent
BUILD = DEMO_ROOT / "build"
AST_CODEGEN = DEMO_ROOT.parent / "experiments" / "ast_codegen"


def find_nvcc():
    for cand in (os.environ.get("NVCC"), shutil.which("nvcc"),
                 "/tmp/cuda-12.9/bin/nvcc"):
        if cand and Path(cand).exists():
            return cand
    sys.exit("nvcc not found; set NVCC or see demo/README.md §2 for the "
             "user-level CUDA 12.9 install recipe.")


def wrap_compile_run(network, tensors):
    """Wrap build/<network>_cuda_output.cu, compile, stage bins, run.

    tensors: list of (var_name, np.float32 array).  The last tensor in the
    wrapper's TENSOR_SPECS entry is the output and is not staged.
    Returns the raw stdout of the executable.
    """
    body = BUILD / f"{network}_cuda_output.cu"
    cu = BUILD / f"{network}.cu"
    exe = BUILD / network
    if not body.exists():
        sys.exit(f"{body} not found; run `make kernels` (or `make demoN`) first.")

    subprocess.run(
        [sys.executable, str(AST_CODEGEN / "wrap_generated_cuda.py"),
         network, str(body), str(cu)],
        cwd=BUILD, check=True)

    nvcc = find_nvcc()
    nvcc_flags = os.environ.get("NVCC_FLAGS", "-O2").split()
    print(f"$ {nvcc} {' '.join(nvcc_flags)} -o {network} {network}.cu")
    subprocess.run([nvcc, *nvcc_flags, "-o", str(exe), str(cu)],
                   cwd=BUILD, check=True)

    for var_name, arr in tensors:
        assert arr.dtype == np.float32
        (BUILD / f"{network}_{var_name}.bin").write_bytes(arr.tobytes())

    result = subprocess.run([f"./{network}"], cwd=BUILD, check=True,
                            capture_output=True, text=True)
    print(result.stdout, end="")
    return result.stdout


def read_output(network, shape):
    out = np.frombuffer((BUILD / f"{network}_output.bin").read_bytes(),
                        dtype=np.float32)
    return out.reshape(shape)


def report(network, got, expected, atol=1e-5, rtol=1e-5):
    max_abs = float(np.max(np.abs(got - expected)))
    denom = np.maximum(np.abs(expected), 1e-20)
    max_rel = float(np.max(np.abs(got - expected) / denom))
    print(f"max abs error: {max_abs:.3e}")
    print(f"max rel error: {max_rel:.3e}")
    ok = np.allclose(got, expected, atol=atol, rtol=rtol)
    print(f"{network}: {'PASSED' if ok else 'FAILED'} (atol={atol}, rtol={rtol})")
    if not ok:
        sys.exit(1)
