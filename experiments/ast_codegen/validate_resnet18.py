#!/usr/bin/env python3
"""Numerical validation of the generated ResNet-18 C code."""

import subprocess
import numpy as np

np.random.seed(18)
scale = 1e-2
eps = 1e-5

def rand(*shape):
    return (scale * np.random.randn(*shape)).astype(np.float32)

# ------------------------------------------------------------------
# Parameter initialization
# ------------------------------------------------------------------
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
# Reference forward in NumPy
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
    xp = np.pad(x, ((0,0),(0,0),(p,p),(p,p)), mode='constant')
    # Extract sliding windows with stride s: shape (N, Cin, Ho, Wo, K, K)
    patches = np.lib.stride_tricks.as_strided(
        xp,
        shape=(N, Cin, Ho, Wo, K, K),
        strides=(xp.strides[0], xp.strides[1], xp.strides[2]*s, xp.strides[3]*s, xp.strides[2], xp.strides[3]),
        writeable=False
    )
    # Contract over Cin and the two kernel axes.
    out = np.tensordot(patches, w, axes=([1, 4, 5], [1, 2, 3]))  # (N, Ho, Wo, Cout)
    out = np.transpose(out, (0, 3, 1, 2))  # (N, Cout, Ho, Wo)
    out = out + b.reshape(1, Cout, 1, 1)
    return out

def maxpool2d(x, kernel=3, padding=1, stride=2):
    N, C, H, W = x.shape
    Ho = (H + 2*padding - kernel) // stride + 1
    Wo = (W + 2*padding - kernel) // stride + 1
    xp = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant', constant_values=-np.inf)
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

def ref_forward():
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
    x = x.mean(axis=(2,3))
    x = x.reshape(1, 512)
    x = x @ fc_w.T + fc_b
    return x

ref_output = ref_forward()

# ------------------------------------------------------------------
# Generate C test program
# ------------------------------------------------------------------
def arr2c(a, fmt='.9g'):
    return ', '.join(f'{v:{fmt}}f' for v in a.flatten())

def def_array(name, shape, init=None):
    total = int(np.prod(shape))
    if init is not None:
        return f'float {name}[{"][".join(map(str, shape))}] = {{{arr2c(init)}}};'
    return f'float {name}[{"][".join(map(str, shape))}];'

lines = []
lines.append('#include <math.h>')
lines.append('#include <stdio.h>')
lines.append('')

lines.append(def_array('input', (1,3,224,224), input_))
lines.append(def_array('conv1_w', (64,3,7,7), conv1_w))
lines.append(def_array('conv1_b', (64,), conv1_b))
lines.append(def_array('bn1_mean', (64,), bn1_mean))
lines.append(def_array('bn1_var', (64,), bn1_var))
lines.append(def_array('bn1_gamma', (64,), bn1_gamma))
lines.append(def_array('bn1_beta', (64,), bn1_beta))
lines.append(f'float bn_eps = {eps:.9g}f;')
lines.append(def_array('conv1_out', (1,64,112,112)))
lines.append(def_array('bn1_out', (1,64,112,112)))
lines.append(def_array('relu1_out', (1,64,112,112)))
lines.append(def_array('pool_out', (1,64,56,56)))

for prefix, inch, outch, m, proj in blocks:
    p = params[prefix]
    lines.append(def_array(prefix+'_w1', (outch, inch, 3, 3), p['w1']))
    lines.append(def_array(prefix+'_b1', (outch,), p['b1']))
    lines.append(def_array(prefix+'_w2', (outch, outch, 3, 3), p['w2']))
    lines.append(def_array(prefix+'_b2', (outch,), p['b2']))
    for i, s in enumerate(['mean', 'var', 'gamma', 'beta']):
        lines.append(def_array(prefix+'_bn1_'+s, (outch,), p['bn1'][i]))
        lines.append(def_array(prefix+'_bn2_'+s, (outch,), p['bn2'][i]))
    lines.append(def_array(prefix+'_conv1_out', (1,outch,m,m)))
    lines.append(def_array(prefix+'_bn1_out', (1,outch,m,m)))
    lines.append(def_array(prefix+'_relu1_out', (1,outch,m,m)))
    lines.append(def_array(prefix+'_conv2_out', (1,outch,m,m)))
    lines.append(def_array(prefix+'_bn2_out', (1,outch,m,m)))
    lines.append(def_array(prefix+'_add_out', (1,outch,m,m)))
    if proj:
        lines.append(def_array(prefix+'_shortcut_w', (outch, inch, 1, 1), p['shortcut_w']))
        lines.append(def_array(prefix+'_shortcut_b', (outch,), p['shortcut_b']))
        for i, s in enumerate(['mean', 'var', 'gamma', 'beta']):
            lines.append(def_array(prefix+'_shortcut_'+s, (outch,), p['shortcut_bn'][i]))
        lines.append(def_array(prefix+'_shortcut_out', (1,outch,m,m)))

lines.append(def_array('avgpool_out', (1,512)))
lines.append(def_array('fc_w', (1000,512), fc_w))
lines.append(def_array('fc_b', (1000,), fc_b))
lines.append(def_array('output', (1,1000)))

lines.append('')
lines.append('#include "resnet18_wrapped.c"')
lines.append('')
lines.append('int main() {')
lines.append('  resnet18();')
lines.append('  for (int i = 0; i < 1000; i++) printf("%.9g ", output[0][i]);')
lines.append('  printf("\\n");')
lines.append('  return 0;')
lines.append('}')

c_code = '\n'.join(lines)
with open('test_resnet18.c', 'w') as f:
    f.write(c_code)

subprocess.run(['gcc', '-std=c99', 'test_resnet18.c', '-o', 'test_resnet18', '-lm'], check=True)
result = subprocess.run(['bash', '-c', 'ulimit -s unlimited && ./test_resnet18'],
                        capture_output=True, text=True, check=True)
c_out = np.array([float(x) for x in result.stdout.split()], dtype=np.float32)

print("Reference output (first 10):", ref_output[0, :10])
print("C output (first 10):        ", c_out[:10])
print("Match:", np.allclose(ref_output.flatten(), c_out, atol=1e-4, rtol=1e-4))
