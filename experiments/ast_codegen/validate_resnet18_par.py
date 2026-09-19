#!/usr/bin/env python3
"""Numerical validation + performance comparison of the OpenMP ResNet-18.

Compares the parfor-based generated C (resnet18_par_wrapped.c, compiled with
-fopenmp) against the same NumPy reference used by validate_resnet18.py and
against the sequential generated C (resnet18_wrapped.c), then times both
versions with 2/4/8 OpenMP threads.

Full ImageNet input size 224x224 (the generated C is loop-based, so the
input size does not affect code size, only run time).
"""

import os
import subprocess
import numpy as np

np.random.seed(18)
scale = 1e-2
eps = 1e-5

def rand(*shape):
    return (scale * np.random.randn(*shape)).astype(np.float32)

# ------------------------------------------------------------------
# Parameter initialization (identical to validate_resnet18.py)
# ------------------------------------------------------------------
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
    xp = np.pad(x, ((0,0),(0,0),(p,p),(p,p)), mode='constant')
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
# C declarations (with / without initializers)
# ------------------------------------------------------------------
def arr2c(a, fmt='.9g'):
    return ', '.join(f'{v:{fmt}}f' for v in a.flatten())

def def_array(name, shape, init=None):
    if init is not None:
        return f'float {name}[{"][".join(map(str, shape))}] = {{{arr2c(init)}}};'
    return f'float {name}[{"][".join(map(str, shape))}];'

def decl_lines(with_init):
    g = lambda name, shape, init=None: def_array(name, shape, init if with_init else None)
    lines = []
    lines.append(g('input', (1,3,224,224), input_))
    lines.append(g('conv1_w', (64,3,7,7), conv1_w))
    lines.append(g('conv1_b', (64,), conv1_b))
    lines.append(g('bn1_mean', (64,), bn1_mean))
    lines.append(g('bn1_var', (64,), bn1_var))
    lines.append(g('bn1_gamma', (64,), bn1_gamma))
    lines.append(g('bn1_beta', (64,), bn1_beta))
    lines.append(f'float bn_eps = {eps:.9g}f;' if with_init else 'float bn_eps;')
    lines.append(g('conv1_out', (1,64,112,112)))
    lines.append(g('bn1_out', (1,64,112,112)))
    lines.append(g('relu1_out', (1,64,112,112)))
    lines.append(g('pool_out', (1,64,56,56)))
    for prefix, inch, outch, m, proj in blocks:
        p = params[prefix]
        lines.append(g(prefix+'_w1', (outch, inch, 3, 3), p['w1']))
        lines.append(g(prefix+'_b1', (outch,), p['b1']))
        lines.append(g(prefix+'_w2', (outch, outch, 3, 3), p['w2']))
        lines.append(g(prefix+'_b2', (outch,), p['b2']))
        for i, s in enumerate(['mean', 'var', 'gamma', 'beta']):
            lines.append(g(prefix+'_bn1_'+s, (outch,), p['bn1'][i]))
            lines.append(g(prefix+'_bn2_'+s, (outch,), p['bn2'][i]))
        lines.append(g(prefix+'_conv1_out', (1,outch,m,m)))
        lines.append(g(prefix+'_bn1_out', (1,outch,m,m)))
        lines.append(g(prefix+'_relu1_out', (1,outch,m,m)))
        lines.append(g(prefix+'_conv2_out', (1,outch,m,m)))
        lines.append(g(prefix+'_bn2_out', (1,outch,m,m)))
        lines.append(g(prefix+'_add_out', (1,outch,m,m)))
        if proj:
            lines.append(g(prefix+'_shortcut_w', (outch, inch, 1, 1), p['shortcut_w']))
            lines.append(g(prefix+'_shortcut_b', (outch,), p['shortcut_b']))
            for i, s in enumerate(['mean', 'var', 'gamma', 'beta']):
                lines.append(g(prefix+'_shortcut_'+s, (outch,), p['shortcut_bn'][i]))
            lines.append(g(prefix+'_shortcut_out', (1,outch,m,m)))
    lines.append(g('avgpool_out', (1,512)))
    lines.append(g('fc_w', (1000,512), fc_w))
    lines.append(g('fc_b', (1000,), fc_b))
    lines.append(g('output', (1,1000)))
    return lines

# ------------------------------------------------------------------
# Numerical check: one binary runs both the sequential and the OpenMP net
# ------------------------------------------------------------------
lines = ['#include <math.h>', '#include <stdio.h>', '']
lines += decl_lines(with_init=True)
lines.append('')
lines.append('#include "resnet18_wrapped.c"')
lines.append('#include "resnet18_par_wrapped.c"')
lines.append('')
lines.append('int main() {')
lines.append('  resnet18();')
lines.append('  printf("SEQ\\n");')
lines.append('  for (int i = 0; i < 1000; i++) printf("%.9g ", output[0][i]);')
lines.append('  printf("\\n");')
lines.append('  resnet18_par();')
lines.append('  printf("PAR\\n");')
lines.append('  for (int i = 0; i < 1000; i++) printf("%.9g ", output[0][i]);')
lines.append('  printf("\\n");')
lines.append('  return 0;')
lines.append('}')

with open('test_resnet18_par.c', 'w') as f:
    f.write('\n'.join(lines))

subprocess.run(['gcc', '-O2', '-std=gnu99', '-fopenmp',
                'test_resnet18_par.c', '-o', 'test_resnet18_par', '-lm'], check=True)
result = subprocess.run(['bash', '-c', 'ulimit -s unlimited && OMP_NUM_THREADS=4 ./test_resnet18_par'],
                        capture_output=True, text=True, check=True)
out = result.stdout.split()
seq_i = out.index('SEQ')
par_i = out.index('PAR')
seq_out = np.array([float(x) for x in out[seq_i+1:par_i]], dtype=np.float32)
par_out = np.array([float(x) for x in out[par_i+1:]], dtype=np.float32)

print("Reference output (first 10):", ref_output[0, :10])
print("Sequential C out (first 10):", seq_out[:10])
print("OpenMP C out (first 10):    ", par_out[:10])
max_err_seq = float(np.max(np.abs(ref_output.flatten() - seq_out)))
max_err_ref = float(np.max(np.abs(ref_output.flatten() - par_out)))
max_err_sp  = float(np.max(np.abs(seq_out - par_out)))
print(f"Max |seq - numpy_ref| = {max_err_seq:.3e}")
print(f"Max |par - numpy_ref| = {max_err_ref:.3e}")
print(f"Max |par - seq|       = {max_err_sp:.3e}")
print("Match (seq,  numpy, 1e-4):", np.allclose(ref_output.flatten(), seq_out, atol=1e-4, rtol=1e-4))
print("Match (par,  numpy, 1e-4):", np.allclose(ref_output.flatten(), par_out, atol=1e-4, rtol=1e-4))
print("Match (par == seq, exact):", np.array_equal(seq_out, par_out))

# ------------------------------------------------------------------
# Timing (declarations only, no initializers; best of 3 rounds x 5 reps)
# ------------------------------------------------------------------
tl = ['#include <stdio.h>', '#include <time.h>', '']
tl += decl_lines(with_init=False)
tl.append('#include "%s"')
tl.append('''
static double now(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
int main() {
  int reps = 5;
  %s();  /* warmup */
  double best = 1e30;
  for (int t = 0; t < 3; t++) {
    double t0 = now();
    for (int r = 0; r < reps; r++) %s();
    double dt = (now() - t0) / reps;
    if (dt < best) best = dt;
  }
  printf("%%.6f\\n", best * 1e3);  /* milliseconds per inference */
  return 0;
}
''')
timing_c = '\n'.join(tl)
with open('time_resnet18_seq.c', 'w') as f:
    f.write(timing_c % ('resnet18_wrapped.c', 'resnet18', 'resnet18'))
with open('time_resnet18_par.c', 'w') as f:
    f.write(timing_c % ('resnet18_par_wrapped.c', 'resnet18_par', 'resnet18_par'))
subprocess.run(['gcc', '-O2', '-std=gnu99', 'time_resnet18_seq.c', '-o', 'time_resnet18_seq', '-lm'], check=True)
subprocess.run(['gcc', '-O2', '-std=gnu99', '-fopenmp', 'time_resnet18_par.c', '-o', 'time_resnet18_par', '-lm'], check=True)

t_seq = float(subprocess.run(['./time_resnet18_seq'], capture_output=True, text=True, check=True).stdout.strip())
print(f"Sequential:           {t_seq:.3f} ms/inference")
for nt in (2, 4, 8):
    env = dict(os.environ, OMP_NUM_THREADS=str(nt))
    t_par = float(subprocess.run(['./time_resnet18_par'], capture_output=True, text=True, check=True, env=env).stdout.strip())
    print(f"OpenMP ({nt} threads):    {t_par:.3f} ms/inference  (speedup {t_seq / t_par:.2f}x)")
