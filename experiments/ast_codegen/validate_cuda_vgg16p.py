#!/usr/bin/env python3
"""End-to-end numerical validation of the generated VGG-16 (224x224) CUDA code.

Pipeline (mirrors validate_cuda_alexnet.py):
  1. comm_to_cuda_vgg16p (OCaml) prints vgg16p_test_cuda_output.cu from the
     extracted vgg16p_comm AST (VGG16_Par.v, M7 validation chain; note: this
     AST has the corner-crop topology -- pools 1..4 are computed but dead --
     and ends at the fc3 logits, no softmax).
  2. wrap_generated_cuda.py wraps the body into a compilable vgg16p_test.cu.
  3. nvcc compiles it; inputs/weights are staged as per-tensor .bin files.
  4. The executable is run REPS times; the output is compared element-wise
     against the M7 NumPy reference (same seed=16, He-like scaling, same
     buffer topology) and against the sequential generated C binary.
  5. GPU kernel-chain time (TIME), staging+H2D (MEMTIME_H2D) and D2H
     (MEMTIME_D2H) are parsed from the executable output (min over runs).

Set NVCC to the nvcc binary if it is not on PATH.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
NVCC = os.environ.get("NVCC", "nvcc")
REPS = int(os.environ.get("VGG16P_RUNS", "5"))

# Same data regime as validate_vgg16p_par.py (M7): seed 16, He-like scaling.
np.random.seed(16)

convs = [
    (1, 1, 3, 64, 224), (1, 2, 64, 64, 224),
    (2, 1, 64, 128, 112), (2, 2, 128, 128, 112),
    (3, 1, 128, 256, 56), (3, 2, 256, 256, 56), (3, 3, 256, 256, 56),
    (4, 1, 256, 512, 28), (4, 2, 512, 512, 28), (4, 3, 512, 512, 28),
    (5, 1, 512, 512, 14), (5, 2, 512, 512, 14), (5, 3, 512, 512, 14),
]
block_m = {1: 224, 2: 112, 3: 56, 4: 28, 5: 14}

params = {'input': np.random.randn(1, 3, 224, 224).astype(np.float32)}
for block, layer, inch, outch, m in convs:
    params[f'conv{block}_{layer}_w'] = (np.random.randn(outch, inch, 3, 3) / np.sqrt(9 * inch)).astype(np.float32)
    params[f'conv{block}_{layer}_b'] = (0.01 * np.random.randn(outch)).astype(np.float32)
params['fc1_w'] = (np.random.randn(4096, 25088) / np.sqrt(25088)).astype(np.float32)
params['fc1_b'] = (0.01 * np.random.randn(4096)).astype(np.float32)
params['fc2_w'] = (np.random.randn(4096, 4096) / np.sqrt(4096)).astype(np.float32)
params['fc2_b'] = (0.01 * np.random.randn(4096)).astype(np.float32)
params['fc3_w'] = (np.random.randn(1000, 4096) / np.sqrt(4096)).astype(np.float32)
params['fc3_b'] = (0.01 * np.random.randn(1000)).astype(np.float32)


def relu(x):
    return np.maximum(x, 0.0)


def conv2d(x, w, b):
    N, Cin, H, W = x.shape
    Cout = w.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
    patches = np.lib.stride_tricks.as_strided(
        xp, shape=(N, Cin, H, W, 3, 3),
        strides=(xp.strides[0], xp.strides[1], xp.strides[2], xp.strides[3], xp.strides[2], xp.strides[3]),
        writeable=False)
    out = np.tensordot(patches, w, axes=([1, 4, 5], [1, 2, 3]))
    return np.transpose(out, (0, 3, 1, 2)) + b.reshape(1, Cout, 1, 1)


def maxpool2d(x):
    N, C, H, W = x.shape
    return x.reshape(N, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))


def ref_forward():
    x = params['input']
    for block in range(1, 6):
        for b, layer, inch, outch, m in [c for c in convs if c[0] == block]:
            x = relu(conv2d(x, params[f'conv{b}_{layer}_w'], params[f'conv{b}_{layer}_b']))
        if block < 5:
            m_next = block_m[block + 1]
            x = x[:, :, :m_next, :m_next].copy()  # corner crop; pool b is dead
        else:
            x = maxpool2d(x)                       # pool5 feeds flatten
    x = x.reshape(1, -1)
    x = relu(x @ params['fc1_w'].T + params['fc1_b'])
    x = relu(x @ params['fc2_w'].T + params['fc2_b'])
    return x @ params['fc3_w'].T + params['fc3_b']


def run(cmd, **kw):
    subprocess.run(cmd, cwd=ROOT, check=True, **kw)


def main():
    name = "vgg16p_test"

    # 1-2. Regenerate the CUDA body and wrap it.
    run(["./comm_to_cuda_vgg16p"])
    run([sys.executable, "wrap_generated_cuda.py", name,
         f"{name}_cuda_output.cu", f"{name}.cu"])

    # 3. Compile.
    nvcc_flags = os.environ.get("NVCC_FLAGS", "-O2").split()
    run([NVCC, *nvcc_flags, "-o", name, f"{name}.cu"])

    # 4. Stage inputs (one .bin per tensor, as the wrapper expects).
    for var_name, arr in params.items():
        (ROOT / f"{name}_{var_name}.bin").write_bytes(arr.tobytes())

    # 5. Run REPS times; keep the output of the last run, min of timings.
    times, h2d, d2h = [], [], []
    for _ in range(REPS):
        result = subprocess.run([f"./{name}"], cwd=ROOT, check=True,
                                capture_output=True, text=True)
        for line in result.stdout.splitlines():
            m = re.match(r"(TIME|MEMTIME_H2D|MEMTIME_D2H)\s+\S+\s+([0-9.]+)", line)
            if m:
                {"TIME": times, "MEMTIME_H2D": h2d, "MEMTIME_D2H": d2h}[m.group(1)].append(float(m.group(2)))
    print(f"GPU kernel chain TIME (ms, {len(times)} runs): "
          + ", ".join(f"{t:.3f}" for t in times) + f"  min={min(times):.3f}")
    print(f"MEMTIME_H2D (ms): min={min(h2d):.3f}   MEMTIME_D2H (ms): min={min(d2h):.3f}")

    output = np.frombuffer((ROOT / f"{name}_output.bin").read_bytes(),
                           dtype=np.float32).reshape(1, 1000)
    expected = ref_forward()
    print(f"ref output stats: min={expected.min():.4f} max={expected.max():.4f} "
          f"mean|.|={np.abs(expected).mean():.4f}")

    max_abs = float(np.max(np.abs(output - expected)))
    max_rel = float(np.max(np.abs(output - expected) / np.maximum(np.abs(expected), 1e-20)))
    print("NumPy reference (first 10):", expected[0, :10])
    print("CUDA output     (first 10):", output[0, :10])
    print(f"max abs error vs numpy: {max_abs:.3e}")
    print(f"max rel error vs numpy: {max_rel:.3e}")
    ok = np.allclose(output, expected, atol=1e-5, rtol=1e-5)
    print(f"{name} vs numpy: {'PASSED' if ok else 'FAILED'} (atol=1e-5, rtol=1e-5)")

    # Cross-check against the M7 sequential C binary (bit-exactness probe).
    seq_bin = ROOT / "test_vgg16p_seq"
    seq_params = ROOT / "vgg16p_params.bin"
    if seq_bin.exists():
        if not seq_params.exists():
            with open(seq_params, 'wb') as f:
                f.write(params['input'].tobytes())
                for block, layer, inch, outch, m in convs:
                    f.write(params[f'conv{block}_{layer}_w'].tobytes())
                    f.write(params[f'conv{block}_{layer}_b'].tobytes())
                for k in ('fc1_w', 'fc1_b', 'fc2_w', 'fc2_b', 'fc3_w', 'fc3_b'):
                    f.write(params[k].tobytes())
        r = subprocess.run(["./test_vgg16p_seq"], cwd=ROOT, check=True,
                           capture_output=True, text=True)
        seq_out = np.array([float(x) for x in r.stdout.split()], dtype=np.float32)
        max_abs_c = float(np.max(np.abs(output[0] - seq_out)))
        print(f"max abs error vs sequential C: {max_abs_c:.3e}  "
              f"(bit-exact: {np.array_equal(output[0], seq_out)})")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
