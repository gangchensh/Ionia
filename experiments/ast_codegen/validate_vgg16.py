#!/usr/bin/env python3
"""Numerical validation of the generated VGG-16 C code."""

import subprocess
import numpy as np
import struct

np.random.seed(16)
scale = 1e-2


def rand(*shape):
    return (scale * np.random.randn(*shape)).astype(np.float32)


# ------------------------------------------------------------------
# Architecture configuration (must match Coq AST)
# ------------------------------------------------------------------
convs = [
    (1, 1, 3, 64, 224),
    (1, 2, 64, 64, 224),
    (2, 1, 64, 128, 112),
    (2, 2, 128, 128, 112),
    (3, 1, 128, 256, 56),
    (3, 2, 256, 256, 56),
    (3, 3, 256, 256, 56),
    (4, 1, 256, 512, 28),
    (4, 2, 512, 512, 28),
    (4, 3, 512, 512, 28),
    (5, 1, 512, 512, 14),
    (5, 2, 512, 512, 14),
    (5, 3, 512, 512, 14),
]

pools = [1, 2, 3, 4, 5]

# ------------------------------------------------------------------
# Parameter initialization
# ------------------------------------------------------------------
params = {}
params['input'] = rand(1, 3, 224, 224)
for block, layer, inch, outch, m in convs:
    params[f'conv{block}_{layer}_w'] = rand(outch, inch, 3, 3)
    params[f'conv{block}_{layer}_b'] = rand(outch)
params['fc1_w'] = rand(4096, 25088)
params['fc1_b'] = rand(4096)
params['fc2_w'] = rand(4096, 4096)
params['fc2_b'] = rand(4096)
params['fc3_w'] = rand(1000, 4096)
params['fc3_b'] = rand(1000)

# ------------------------------------------------------------------
# Reference forward in NumPy
# ------------------------------------------------------------------
def relu(x):
    return np.maximum(x, 0.0)


def conv2d(x, w, b, padding=1, stride=1):
    N, Cin, H, W = x.shape
    Cout, _, K, _ = w.shape
    p = padding
    s = stride
    Ho = (H + 2 * p - K) // s + 1
    Wo = (W + 2 * p - K) // s + 1
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    patches = np.lib.stride_tricks.as_strided(
        xp,
        shape=(N, Cin, Ho, Wo, K, K),
        strides=(xp.strides[0], xp.strides[1], xp.strides[2] * s, xp.strides[3] * s, xp.strides[2], xp.strides[3]),
        writeable=False,
    )
    out = np.tensordot(patches, w, axes=([1, 4, 5], [1, 2, 3]))
    out = np.transpose(out, (0, 3, 1, 2))
    out = out + b.reshape(1, Cout, 1, 1)
    return out


def maxpool2d(x, kernel=2, padding=0, stride=2):
    N, C, H, W = x.shape
    Ho = (H + 2 * padding - kernel) // stride + 1
    Wo = (W + 2 * padding - kernel) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=-np.inf)
    patches = np.lib.stride_tricks.as_strided(
        xp,
        shape=(N, C, Ho, Wo, kernel, kernel),
        strides=(xp.strides[0], xp.strides[1], xp.strides[2] * stride, xp.strides[3] * stride, xp.strides[2], xp.strides[3]),
        writeable=False,
    )
    return patches.max(axis=(4, 5))


def ref_forward():
    x = params['input']
    current_block = 1
    for block, layer, inch, outch, m in convs:
        x = relu(conv2d(x, params[f'conv{block}_{layer}_w'], params[f'conv{block}_{layer}_b'], padding=1, stride=1))
        if layer == len([c for c in convs if c[0] == block]):
            x = maxpool2d(x, kernel=2, padding=0, stride=2)
    x = x.reshape(1, -1)
    x = relu(x @ params['fc1_w'].T + params['fc1_b'])
    x = relu(x @ params['fc2_w'].T + params['fc2_b'])
    x = x @ params['fc3_w'].T + params['fc3_b']
    return x


ref_output = ref_forward()

# ------------------------------------------------------------------
# Write binary parameter file
# ------------------------------------------------------------------
binary_order = []
binary_order.append(('input', (1, 3, 224, 224)))
for block, layer, inch, outch, m in convs:
    binary_order.append((f'conv{block}_{layer}_w', (outch, inch, 3, 3)))
    binary_order.append((f'conv{block}_{layer}_b', (outch,)))
binary_order.append(('fc1_w', (4096, 25088)))
binary_order.append(('fc1_b', (4096,)))
binary_order.append(('fc2_w', (4096, 4096)))
binary_order.append(('fc2_b', (4096,)))
binary_order.append(('fc3_w', (1000, 4096)))
binary_order.append(('fc3_b', (1000,)))

with open('vgg16_params.bin', 'wb') as f:
    for name, shape in binary_order:
        f.write(params[name].tobytes())

# ------------------------------------------------------------------
# Generate small C test driver
# ------------------------------------------------------------------
code = []
code.append('#include <stdio.h>')
code.append('#include <math.h>')
code.append('#include "externs/vgg16_externs.h"')
code.append('')

# Provide definitions for every parameter declared in the extern header.
for name, shape in binary_order:
    total = 1
    dims = []
    for d in shape:
        total *= d
        dims.append(str(d))
    code.append(f'float {name}[{"][".join(dims)}];')
code.append('')

# activation buffers (already declared in externs; provide definitions)
for block, layer, inch, outch, m in convs:
    code.append(f'float conv{block}_{layer}_out[1][{outch}][{m}][{m}];')
    code.append(f'float relu{block}_{layer}_out[1][{outch}][{m}][{m}];')
for block, ch, m in [(1, 64, 112), (2, 128, 56), (3, 256, 28), (4, 512, 14), (5, 512, 7)]:
    code.append(f'float pool{block}_out[1][{ch}][{m}][{m}];')
code.append('float flatten_out[1][25088];')
code.append('float fc1_out[1][4096];')
code.append('float fc1_relu_out[1][4096];')
code.append('float fc2_out[1][4096];')
code.append('float fc2_relu_out[1][4096];')
code.append('float output[1][1000];')
code.append('')

code.append('void vgg16(void);')
code.append('')

code.append('static void read_params(const char* path) {')
code.append('  FILE* f = fopen(path, "rb");')
code.append('  if (!f) { perror("fopen"); return; }')
for name, shape in binary_order:
    total = 1
    for d in shape:
        total *= d
    code.append(f'  fread({name}, sizeof(float), {total}, f);')
code.append('  fclose(f);')
code.append('}')
code.append('')

code.append('int main() {')
code.append('  read_params("vgg16_params.bin");')
code.append('  vgg16();')
code.append('  for (int i = 0; i < 1000; i++) printf("%.9g ", output[0][i]);')
code.append('  printf("\\n");')
code.append('  return 0;')
code.append('}')

c_code = '\n'.join(code)
with open('test_vgg16.c', 'w') as f:
    f.write(c_code)

# Include the wrapped C body at the end so the extern declarations are visible.
with open('vgg16_wrapped.c', 'r') as f:
    wrapped = f.read()
with open('test_vgg16.c', 'a') as f:
    f.write('\n')
    f.write(wrapped)

subprocess.run(['gcc', '-std=c99', 'test_vgg16.c', '-o', 'test_vgg16', '-lm'], check=True)
result = subprocess.run(['bash', '-c', 'ulimit -s unlimited && ./test_vgg16'],
                        capture_output=True, text=True, check=True)
c_out = np.array([float(x) for x in result.stdout.split()], dtype=np.float32)

print("Reference output (first 10):", ref_output[0, :10])
print("C output (first 10):        ", c_out[:10])

# Compare with a generous tolerance because the reference uses float32 throughout
# and the C code evaluates reductions in a different order.
match = np.allclose(ref_output[0], c_out, atol=1e-4, rtol=1e-4)
print("Match:", match)
if not match:
    raise SystemExit(1)
