#!/usr/bin/env python3
"""End-to-end numerical validation of the E4 fully-fused ResNet-18 CUDA code.

Identical data regime and NumPy reference as validate_cuda_resnet18.py
(M10b): seed 18, scale 1e-2, BN eps 1e-5, batch=1 inference.  Only the
generator (comm_to_cuda_resnet18_fused) and the network name
(resnet18_fused) differ; .bin staging reuses the M10 baseline files via
hardlink when present (same seed => identical bytes).

Set NVCC to the nvcc binary if it is not on PATH; NVCC_FLAGS selects the
compile flags (default "-O2 -arch=native").
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
NVCC = os.environ.get("NVCC", "nvcc")

# Same data regime as validate_resnet18.py (CPU C validation, M7).
np.random.seed(18)
scale = 1e-2
eps = 1e-5


def rand(*shape):
    return (scale * np.random.randn(*shape)).astype(np.float32)


input_ = rand(1, 3, 224, 224)

conv1_w = rand(64, 3, 7, 7)
conv1_b = rand(64)
bn1_mean = rand(64)
bn1_var = rand(64) ** 2 + 1e-4
bn1_gamma = rand(64)
bn1_beta = rand(64)

# Each entry: (prefix, inch, outch, m, proj)
blocks = [
    ('l1_b1', 64,  64,  56, False),
    ('l1_b2', 64,  64,  56, False),
    ('l2_b1', 64,  128, 28, True),
    ('l2_b2', 128, 128, 28, False),
    ('l3_b1', 128, 256, 14, True),
    ('l3_b2', 256, 256, 14, False),
    ('l4_b1', 256, 512, 7,  True),
    ('l4_b2', 512, 512, 7,  False),
]

params = {}
for prefix, inch, outch, m, proj in blocks:
    params[prefix] = {
        'inch': inch, 'outch': outch, 'm': m, 'proj': proj,
        'w1': rand(outch, inch, 3, 3), 'b1': rand(outch),
        'w2': rand(outch, outch, 3, 3), 'b2': rand(outch),
        'bn1': (rand(outch), rand(outch)**2+1e-4, rand(outch), rand(outch)),
        'bn2': (rand(outch), rand(outch)**2+1e-4, rand(outch), rand(outch)),
    }
    if proj:
        params[prefix]['shortcut_w'] = rand(outch, inch, 1, 1)
        params[prefix]['shortcut_b'] = rand(outch)
        params[prefix]['shortcut_bn'] = (
            rand(outch), rand(outch)**2+1e-4, rand(outch), rand(outch)
        )

fc_w = rand(1000, 512)
fc_b = rand(1000)


# ------------------------------------------------------------------
# Reference forward in NumPy (identical to validate_resnet18.py)
# ------------------------------------------------------------------
def relu(x):
    return np.maximum(x, 0.0)


def batch_norm(x, mean, var, gamma, beta, eps=1e-5):
    return (x - mean.reshape(1, -1, 1, 1)) / np.sqrt(var.reshape(1, -1, 1, 1) + eps) \
           * gamma.reshape(1, -1, 1, 1) + beta.reshape(1, -1, 1, 1)


def conv2d(x, w, b, padding, stride):
    N, Cin, H, W = x.shape
    Cout, _, K, _ = w.shape
    p = padding
    s = stride
    Ho = (H + 2*p - K) // s + 1
    Wo = (W + 2*p - K) // s + 1
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    patches = np.lib.stride_tricks.as_strided(
        xp,
        shape=(N, Cin, Ho, Wo, K, K),
        strides=(xp.strides[0], xp.strides[1], xp.strides[2]*s, xp.strides[3]*s, xp.strides[2], xp.strides[3]),
        writeable=False
    )
    out = np.tensordot(patches, w, axes=([1, 4, 5], [1, 2, 3]))  # (N, Ho, Wo, Cout)
    out = np.transpose(out, (0, 3, 1, 2))  # (N, Cout, Ho, Wo)
    out = out + b.reshape(1, Cout, 1, 1)
    return out


def maxpool2d(x, kernel=3, padding=1, stride=2):
    N, C, H, W = x.shape
    Ho = (H + 2*padding - kernel) // stride + 1
    Wo = (W + 2*padding - kernel) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=-np.inf)
    patches = np.lib.stride_tricks.as_strided(
        xp,
        shape=(N, C, Ho, Wo, kernel, kernel),
        strides=(xp.strides[0], xp.strides[1], xp.strides[2]*stride, xp.strides[3]*stride, xp.strides[2], xp.strides[3]),
        writeable=False
    )
    return patches.max(axis=(4, 5))


def basic_block(x, w1, b1, w2, b2, bn1, bn2, proj_w=None, proj_b=None, proj_bn=None):
    identity = x
    if proj_w is not None:
        identity = conv2d(x, proj_w, proj_b, padding=0, stride=2)
        identity = batch_norm(identity, *proj_bn, eps=eps)
    out = conv2d(x, w1, b1, padding=1, stride=(2 if proj_w is not None else 1))
    out = batch_norm(out, *bn1, eps=eps)
    out = relu(out)
    out = conv2d(out, w2, b2, padding=1, stride=1)
    out = batch_norm(out, *bn2, eps=eps)
    out = out + identity
    out = relu(out)
    return out


def numpy_reference():
    x = input_
    x = conv2d(x, conv1_w, conv1_b, padding=3, stride=2)
    x = batch_norm(x, bn1_mean, bn1_var, bn1_gamma, bn1_beta, eps=eps)
    x = relu(x)
    x = maxpool2d(x, kernel=3, padding=1, stride=2)
    for prefix, inch, outch, m, proj in blocks:
        p = params[prefix]
        if proj:
            x = basic_block(x, p['w1'], p['b1'], p['w2'], p['b2'],
                            p['bn1'], p['bn2'],
                            proj_w=p['shortcut_w'], proj_b=p['shortcut_b'],
                            proj_bn=p['shortcut_bn'])
        else:
            x = basic_block(x, p['w1'], p['b1'], p['w2'], p['b2'],
                            p['bn1'], p['bn2'])
    x = x.mean(axis=(2, 3))
    x = x.reshape(1, 512)
    x = x @ fc_w.T + fc_b
    return x


def tensor_bin_map():
    """Map tensor variable names to the arrays, in wrapper staging order."""
    m = {
        "input": input_,
        "conv1_w": conv1_w, "conv1_b": conv1_b,
        "bn1_mean": bn1_mean, "bn1_var": bn1_var,
        "bn1_gamma": bn1_gamma, "bn1_beta": bn1_beta,
        "fc_w": fc_w, "fc_b": fc_b,
    }
    for prefix, inch, outch, m_, proj in blocks:
        p = params[prefix]
        m[f"{prefix}_w1"] = p['w1']
        m[f"{prefix}_b1"] = p['b1']
        m[f"{prefix}_w2"] = p['w2']
        m[f"{prefix}_b2"] = p['b2']
        for i, s in enumerate(['mean', 'var', 'gamma', 'beta']):
            m[f"{prefix}_bn1_{s}"] = p['bn1'][i]
            m[f"{prefix}_bn2_{s}"] = p['bn2'][i]
        if proj:
            m[f"{prefix}_shortcut_w"] = p['shortcut_w']
            m[f"{prefix}_shortcut_b"] = p['shortcut_b']
            for i, s in enumerate(['mean', 'var', 'gamma', 'beta']):
                m[f"{prefix}_shortcut_{s}"] = p['shortcut_bn'][i]
    return m


def run(cmd, **kw):
    subprocess.run(cmd, cwd=ROOT, check=True, **kw)


def main():
    name = "resnet18_fused"

    # 1-2. Regenerate the CUDA body and wrap it.
    run(["./comm_to_cuda_resnet18_fused"])
    run([sys.executable, "wrap_generated_cuda.py", name,
         f"{name}_cuda_output.cu", f"{name}.cu"])

    # 3. Compile.
    nvcc_flags = os.environ.get("NVCC_FLAGS", "-O2 -arch=native").split()
    run([NVCC, *nvcc_flags, "-o", name, f"{name}.cu"])

    # 4. Stage inputs, reusing the M10 baseline .bin files where present.
    tensors = tensor_bin_map()
    for var_name, arr in tensors.items():
        dst = ROOT / f"{name}_{var_name}.bin"
        src = ROOT / f"resnet18_test_{var_name}.bin"
        if dst.exists() or dst.is_symlink():
            continue
        if src.exists():
            try:
                os.link(src, dst)
            except OSError:
                os.symlink(src.name, dst)
        else:
            dst.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    eps_dst = ROOT / f"{name}_bn_eps.bin"
    if not eps_dst.exists():
        eps_src = ROOT / "resnet18_test_bn_eps.bin"
        if eps_src.exists():
            try:
                os.link(eps_src, eps_dst)
            except OSError:
                os.symlink(eps_src.name, eps_dst)
        else:
            eps_dst.write_bytes(np.float32(eps).tobytes())

    # 5. Run.
    result = subprocess.run([f"./{name}"], cwd=ROOT, check=True,
                            capture_output=True, text=True)
    print(result.stdout, end="")

    output = np.frombuffer((ROOT / f"{name}_output.bin").read_bytes(),
                           dtype=np.float32).reshape(1, 1000)
    expected = numpy_reference()

    abs_err = np.abs(output - expected)
    max_abs = float(np.max(abs_err))
    max_rel = float(np.max(abs_err / np.maximum(np.abs(expected), 1e-20)))
    print("NumPy reference (first 10):", expected[0, :10])
    print("CUDA output (first 10):    ", output[0, :10])
    print(f"max abs error: {max_abs:.3e}")
    print(f"max rel error: {max_rel:.3e}")
    ok = np.allclose(output, expected, atol=1e-5, rtol=1e-5)
    print(f"{name}: {'PASSED' if ok else 'FAILED'} (atol=1e-5, rtol=1e-5)")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
