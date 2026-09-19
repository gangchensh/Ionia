#!/usr/bin/env python3
"""Host harness for the Triton ResNet-18 backend (G2 phase 2).

- Same data regime as validate_cuda_resnet18.py (seed 18, scale 1e-2,
  BN eps 1e-5, batch=1, 224x224) — identical tensors, NumPy reference
  copied verbatim.
- Runs the generated Triton kernel chain (resnet18_triton.py, produced by
  comm_to_triton.ml from the Coq-extracted resnet18_comm AST): 68 kernels
  (pad/stride convs via masked loads, BN, residual Add, GAP, FC).
- Numerical check vs the NumPy reference (atol=rtol=1e-4, same as the CUDA
  validation; 1e-5 also reported).
- Performance: CUDA events around the kernel chain only (no H2D/D2H),
  warmup 20 (covers Triton JIT), min/median of 50; same measurement for a
  PyTorch eager ResNet-18 with identical weights.
"""

import inspect

import numpy as np
import torch
import torch.nn.functional as Fn

import resnet18_triton as G

ITERS = 50

# ---- data: identical to validate_cuda_resnet18.py ------------------------
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


# ---- NumPy reference (verbatim from validate_cuda_resnet18.py) -----------
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
    out = np.tensordot(patches, w, axes=([1, 4, 5], [1, 2, 3]))
    out = np.transpose(out, (0, 3, 1, 2))
    out = out + b.reshape(1, Cout, 1, 1)
    return out


def maxpool2d(x, kernel=3, padding=1, stride=2):
    # The Coq AST zero-pads and zero-initialises the accumulator; on the
    # (non-negative) ReLU output this coincides with -inf padding.
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


ref_output = numpy_reference()

# ---- Triton forward ------------------------------------------------------
dev = "cuda"
t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)

g = {
    "input": t(input_),
    "conv1_w": t(conv1_w), "conv1_b": t(conv1_b),
    "bn1_mean": t(bn1_mean), "bn1_var": t(bn1_var),
    "bn1_gamma": t(bn1_gamma), "bn1_beta": t(bn1_beta),
    "bn_eps": t(np.float32(eps)),
    "fc_w": t(fc_w), "fc_b": t(fc_b),
}
for prefix, inch, outch, m, proj in blocks:
    p = params[prefix]
    g[f"{prefix}_w1"] = t(p['w1'])
    g[f"{prefix}_b1"] = t(p['b1'])
    g[f"{prefix}_w2"] = t(p['w2'])
    g[f"{prefix}_b2"] = t(p['b2'])
    for i, s in enumerate(['mean', 'var', 'gamma', 'beta']):
        g[f"{prefix}_bn1_{s}"] = t(p['bn1'][i])
        g[f"{prefix}_bn2_{s}"] = t(p['bn2'][i])
    if proj:
        g[f"{prefix}_shortcut_w"] = t(p['shortcut_w'])
        g[f"{prefix}_shortcut_b"] = t(p['shortcut_b'])
        for i, s in enumerate(['mean', 'var', 'gamma', 'beta']):
            g[f"{prefix}_shortcut_{s}"] = t(p['shortcut_bn'][i])

# forward() externals, in generated order
fwd_params = list(inspect.signature(G.forward).parameters)
missing = [p for p in fwd_params if p not in g]
assert not missing, f"missing externals: {missing}"


def triton_forward():
    return G.forward(**{p: g[p] for p in fwd_params})


out = triton_forward().detach().cpu().numpy().reshape(1, 1000)
abs_err = np.abs(out - ref_output)
max_abs = float(np.max(abs_err))
max_rel = float(np.max(abs_err / np.maximum(np.abs(ref_output), 1e-20)))
print("Triton output (first 10): ", out[0, :10])
print("Ref output (first 10):    ", ref_output[0, :10])
print(f"max abs error: {max_abs:.3e}")
print(f"max rel error: {max_rel:.3e}")
ok4 = np.allclose(out, ref_output, atol=1e-4, rtol=1e-4)
ok5 = np.allclose(out, ref_output, atol=1e-5, rtol=1e-5)
print(f"Match atol=rtol=1e-4: {ok4}   atol=rtol=1e-5: {ok5}")
assert ok4

# ---- PyTorch eager baseline (same weights) -------------------------------
def t_bn(x, mean, var, gamma, beta):
    return Fn.batch_norm(x, running_mean=g[mean], running_var=g[var],
                         weight=g[gamma], bias=g[beta],
                         training=False, eps=eps)


def torch_forward(x):
    x = Fn.conv2d(x, g["conv1_w"], g["conv1_b"], stride=2, padding=3)
    x = t_bn(x, "bn1_mean", "bn1_var", "bn1_gamma", "bn1_beta")
    x = Fn.relu(x)
    x = Fn.max_pool2d(x, 3, stride=2, padding=1)
    for prefix, inch, outch, m, proj in blocks:
        identity = x
        if proj:
            identity = Fn.conv2d(x, g[f"{prefix}_shortcut_w"],
                                 g[f"{prefix}_shortcut_b"], stride=2)
            identity = t_bn(identity, f"{prefix}_shortcut_mean",
                            f"{prefix}_shortcut_var",
                            f"{prefix}_shortcut_gamma", f"{prefix}_shortcut_beta")
        y = Fn.conv2d(x, g[f"{prefix}_w1"], g[f"{prefix}_b1"],
                      stride=(2 if proj else 1), padding=1)
        y = t_bn(y, f"{prefix}_bn1_mean", f"{prefix}_bn1_var",
                 f"{prefix}_bn1_gamma", f"{prefix}_bn1_beta")
        y = Fn.relu(y)
        y = Fn.conv2d(y, g[f"{prefix}_w2"], g[f"{prefix}_b2"], padding=1)
        y = t_bn(y, f"{prefix}_bn2_mean", f"{prefix}_bn2_var",
                 f"{prefix}_bn2_gamma", f"{prefix}_bn2_beta")
        x = Fn.relu(y + identity)
    x = x.mean(dim=(2, 3))
    x = x.reshape(1, 512)
    return x @ g["fc_w"].T + g["fc_b"]


t_out = torch_forward(g["input"]).detach().cpu().numpy().reshape(1, 1000)
terr = np.max(np.abs(t_out - ref_output))
print(f"max |torch eager - ref| = {terr:.3e}")


# ---- benchmark: CUDA events, kernel-chain only, warmup 20, n=50 ----------
def bench(fn, iters=ITERS, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times = np.array(times)
    return times.min(), np.median(times)


tmin, tmed = bench(triton_forward)
print(f"Triton kernel chain:  min {tmin*1000:9.1f} us   median {tmed*1000:9.1f} us  (n={ITERS})")

pmin, pmed = bench(lambda: torch_forward(g["input"]))
print(f"PyTorch eager (GPU):  min {pmin*1000:9.1f} us   median {pmed*1000:9.1f} us  (n={ITERS})")
