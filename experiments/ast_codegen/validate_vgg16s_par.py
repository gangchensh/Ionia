#!/usr/bin/env python3
"""Numerical validation + performance comparison of the OpenMP VGG-16 (32x32).

Compares the parfor-based generated C (vgg16s_par_wrapped.c, -fopenmp) against
a NumPy reference and against the sequential generated C (vgg16s_seq_wrapped.c),
then times both at 1/2/4/8 threads.

The 32x32 network (vgg16s_*) is structure-identical to the 224x224
CNN_Examples_VGG16.vgg16_comm: same channel widths, same buffer topology --
in particular conv x_1 of blocks 2..5 reads the previous block's last ReLU
buffer (top-left m x m crop), pools 1..4 are computed but dead, and only
pool5 feeds the flatten layer.  The NumPy reference below replicates this
topology exactly.
"""

import os
import subprocess
import numpy as np

np.random.seed(16)

# ------------------------------------------------------------------
# Architecture configuration (must match VGG16_Par.v vgg16s_*)
# ------------------------------------------------------------------
convs = [
    (1, 1, 3, 64, 32), (1, 2, 64, 64, 32),
    (2, 1, 64, 128, 16), (2, 2, 128, 128, 16),
    (3, 1, 128, 256, 8), (3, 2, 256, 256, 8), (3, 3, 256, 256, 8),
    (4, 1, 256, 512, 4), (4, 2, 512, 512, 4), (4, 3, 512, 512, 4),
    (5, 1, 512, 512, 2), (5, 2, 512, 512, 2), (5, 3, 512, 512, 2),
]
# spatial size entering each block (after corner crop)
block_m = {1: 32, 2: 16, 3: 8, 4: 4, 5: 2}

# He-like scaling keeps activations O(1) through 13 convs so the numerical
# comparison is non-vacuous.
params = {'input': np.random.randn(1, 3, 32, 32).astype(np.float32)}
for block, layer, inch, outch, m in convs:
    params[f'conv{block}_{layer}_w'] = (np.random.randn(outch, inch, 3, 3) / np.sqrt(9 * inch)).astype(np.float32)
    params[f'conv{block}_{layer}_b'] = (0.01 * np.random.randn(outch)).astype(np.float32)
params['fc1_w'] = (np.random.randn(4096, 512) / np.sqrt(512)).astype(np.float32)
params['fc1_b'] = (0.01 * np.random.randn(4096)).astype(np.float32)
params['fc2_w'] = (np.random.randn(4096, 4096) / np.sqrt(4096)).astype(np.float32)
params['fc2_b'] = (0.01 * np.random.randn(4096)).astype(np.float32)
params['fc3_w'] = (np.random.randn(1000, 4096) / np.sqrt(4096)).astype(np.float32)
params['fc3_b'] = (0.01 * np.random.randn(1000)).astype(np.float32)

# ------------------------------------------------------------------
# NumPy reference forward (exact AST topology)
# ------------------------------------------------------------------
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

ref_output = ref_forward()
print(f"ref output stats: min={ref_output.min():.4f} max={ref_output.max():.4f} "
      f"mean|.|={np.abs(ref_output).mean():.4f}")

# ------------------------------------------------------------------
# Binary parameter file
# ------------------------------------------------------------------
binary_order = [('input', (1, 3, 32, 32))]
for block, layer, inch, outch, m in convs:
    binary_order.append((f'conv{block}_{layer}_w', (outch, inch, 3, 3)))
    binary_order.append((f'conv{block}_{layer}_b', (outch,)))
binary_order += [('fc1_w', (4096, 512)), ('fc1_b', (4096,)),
                 ('fc2_w', (4096, 4096)), ('fc2_b', (4096,)),
                 ('fc3_w', (1000, 4096)), ('fc3_b', (1000,))]

with open('vgg16s_params.bin', 'wb') as f:
    for name, shape in binary_order:
        f.write(params[name].tobytes())

# ------------------------------------------------------------------
# C driver generation (numeric + timing)
# ------------------------------------------------------------------
defs = []
for name, shape in binary_order:
    dims = ']['.join(str(d) for d in shape)
    defs.append(f'float {name}[{dims}];')
for block, layer, inch, outch, m in convs:
    defs.append(f'float conv{block}_{layer}_out[1][{outch}][{m}][{m}];')
    defs.append(f'float relu{block}_{layer}_out[1][{outch}][{m}][{m}];')
for block, ch, m in [(1, 64, 16), (2, 128, 8), (3, 256, 4), (4, 512, 2), (5, 512, 1)]:
    defs.append(f'float pool{block}_out[1][{ch}][{m}][{m}];')
defs += ['float flatten_out[1][512];', 'float fc1_out[1][4096];',
         'float fc1_relu_out[1][4096];', 'float fc2_out[1][4096];',
         'float fc2_relu_out[1][4096];', 'float output[1][1000];']

reads = '\n'.join(
    f'  fread({name}, sizeof(float), {int(np.prod(shape))}, f);'
    for name, shape in binary_order)

common = f'''
#include <stdio.h>
#include <math.h>
#include "externs/vgg16s_externs.h"
{chr(10).join(defs)}

static void read_params(const char* path) {{
  FILE* f = fopen(path, "rb");
  if (!f) {{ perror("fopen"); return; }}
{reads}
  fclose(f);
}}
'''

def write_driver(tag, fname, mode):
    if mode == 'numeric':
        main = f'''
void {fname}(void);
int main() {{
  read_params("vgg16s_params.bin");
  {fname}();
  for (int i = 0; i < 1000; i++) printf("%.9g ", output[0][i]);
  printf("\\n");
  return 0;
}}
'''
    else:
        main = f'''
#include <time.h>
void {fname}(void);
static double now(void) {{
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}}
int main() {{
  int reps = 5;
  read_params("vgg16s_params.bin");
  {fname}();  /* warmup */
  double best = 1e30;
  for (int t = 0; t < 3; t++) {{
    double t0 = now();
    for (int r = 0; r < reps; r++) {fname}();
    double dt = (now() - t0) / reps;
    if (dt < best) best = dt;
  }}
  printf("%.3f\\n", best * 1e6);  /* microseconds per inference */
  return 0;
}}
'''
    with open(f'{tag}.c', 'w') as f:
        f.write(common + main + '\n')
        with open(f'vgg16s_{"seq" if "seq" in tag else "par"}_wrapped.c') as g:
            f.write(g.read())

for tag, fname, mode in [('test_vgg16s_seq', 'vgg16s_seq', 'numeric'),
                         ('test_vgg16s_par', 'vgg16s_par', 'numeric'),
                         ('time_vgg16s_seq', 'vgg16s_seq', 'timing'),
                         ('time_vgg16s_par', 'vgg16s_par', 'timing')]:
    write_driver(tag, fname, mode)

subprocess.run(['gcc', '-O2', '-std=gnu99', 'test_vgg16s_seq.c', '-o', 'test_vgg16s_seq', '-lm'], check=True)
subprocess.run(['gcc', '-O2', '-std=gnu99', '-fopenmp', 'test_vgg16s_par.c', '-o', 'test_vgg16s_par', '-lm'], check=True)
subprocess.run(['gcc', '-O2', '-std=gnu99', 'time_vgg16s_seq.c', '-o', 'time_vgg16s_seq', '-lm'], check=True)
subprocess.run(['gcc', '-O2', '-std=gnu99', '-fopenmp', 'time_vgg16s_par.c', '-o', 'time_vgg16s_par', '-lm'], check=True)

def run(cmd, threads=None):
    env = dict(os.environ)
    if threads is not None:
        env['OMP_NUM_THREADS'] = str(threads)
    r = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    return r.stdout.strip()

seq_out = np.array([float(x) for x in run(['./test_vgg16s_seq']).split()], dtype=np.float32)
par_out = np.array([float(x) for x in run(['./test_vgg16s_par'], threads=4).split()], dtype=np.float32)

print("Reference output (first 10):", ref_output[0, :10])
print("Sequential C out (first 10):", seq_out[:10])
print("OpenMP C out (first 10):    ", par_out[:10])
max_err_ref = float(np.max(np.abs(ref_output[0] - par_out)))
max_err_seq = float(np.max(np.abs(seq_out - par_out)))
print(f"Max |par - numpy_ref| = {max_err_ref:.3e}")
print(f"Max |par - seq|       = {max_err_seq:.3e}")
print("Match (numpy, atol=1e-3, rtol=1e-3):",
      np.allclose(ref_output[0], par_out, atol=1e-3, rtol=1e-3))
print("Match (seq, bit-exact):", np.array_equal(seq_out, par_out))

# ---- timing ----
t_seq = float(run(['./time_vgg16s_seq']))
print(f"Sequential:              {t_seq:.3f} us/inference  (1.00x)")
for nt in (1, 2, 4, 8):
    t = float(run(['./time_vgg16s_par'], threads=nt))
    print(f"OpenMP ({nt} threads):     {t:.3f} us/inference  (speedup {t_seq / t:.2f}x)")
