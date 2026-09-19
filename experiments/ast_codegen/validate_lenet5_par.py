#!/usr/bin/env python3
"""Numerical validation + performance comparison of the OpenMP LeNet-5.

Compares the parfor-based generated C (lenet5_par_wrapped.c, compiled with
-fopenmp) against the same NumPy reference used by validate_lenet5.py and
against the sequential generated C, then times both versions.
"""

import subprocess
import time
import numpy as np

np.random.seed(7)

# Identical inputs/weights to validate_lenet5.py.
scale = 1e-3
input_  = (scale * np.random.randn(1, 32, 32)).astype(np.float32)
kc1     = (scale * np.random.randn(6, 1, 5, 5)).astype(np.float32)
bc1     = (scale * np.random.randn(6)).astype(np.float32)
kc2     = (scale * np.random.randn(16, 6, 5, 5)).astype(np.float32)
bc2     = (scale * np.random.randn(16)).astype(np.float32)
fc1_w   = (scale * np.random.randn(120, 400)).astype(np.float32)
fc1_b   = (scale * np.random.randn(120)).astype(np.float32)
fc2_w   = (scale * np.random.randn(84, 120)).astype(np.float32)
fc2_b   = (scale * np.random.randn(84)).astype(np.float32)
fc3_w   = (scale * np.random.randn(10, 84)).astype(np.float32)
fc3_b   = (scale * np.random.randn(10)).astype(np.float32)

def relu(x):
    return np.maximum(x, 0.0)

def conv3d_no_pad(x, w, b):
    N, Cin, H, W = x.shape
    Cout, _, K, _ = w.shape
    Ho, Wo = H - K + 1, W - K + 1
    out = np.zeros((N, Cout, Ho, Wo), dtype=np.float32)
    for n in range(N):
        for co in range(Cout):
            for i in range(Ho):
                for j in range(Wo):
                    s = 0.0
                    for ci in range(Cin):
                        for di in range(K):
                            for dj in range(K):
                                s += x[n, ci, i+di, j+dj] * w[co, ci, di, dj]
                    out[n, co, i, j] = s + b[co]
    return out

def maxpool2d_k2(x):
    N, C, H, W = x.shape
    out = np.zeros((N, C, H//2, W//2), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            for i in range(H//2):
                for j in range(W//2):
                    out[n, c, i, j] = np.max(x[n, c, 2*i:2*i+2, 2*j:2*j+2])
    return out

def softmax1d(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def ref_forward():
    x = input_.reshape(1, 1, 32, 32)
    x = conv3d_no_pad(x, kc1, bc1)
    x = relu(x)
    x = maxpool2d_k2(x)
    x = conv3d_no_pad(x, kc2, bc2)
    x = relu(x)
    x = maxpool2d_k2(x)
    x = x.reshape(-1)
    x = relu(x @ fc1_w.T + fc1_b)
    x = relu(x @ fc2_w.T + fc2_b)
    x = x @ fc3_w.T + fc3_b
    x = softmax1d(x)
    return x

ref_output = ref_forward()

def arr2c(a, fmt='.9g'):
    return ', '.join(f'{v:{fmt}}f' for v in a.flatten())

def build_and_run(wrapped, fname, flags, out):
    c_code = f'''
#include <math.h>
#include <stdio.h>

float input[1][32][32] = {{{arr2c(input_)}}};
float kc1[6][1][5][5] = {{{arr2c(kc1)}}};
float bc1[6] = {{{arr2c(bc1)}}};
float c1[6][28][28];
float r1[6][28][28];
float p1[6][14][14];
float kc2[16][6][5][5] = {{{arr2c(kc2)}}};
float bc2[16] = {{{arr2c(bc2)}}};
float c2[16][10][10];
float r2[16][10][10];
float p2[16][5][5];
float flat[400];
float fc1_w[120][400] = {{{arr2c(fc1_w)}}};
float fc1_b[120] = {{{arr2c(fc1_b)}}};
float fc1[120];
float fc1_relu[120];
float fc2_w[84][120] = {{{arr2c(fc2_w)}}};
float fc2_b[84] = {{{arr2c(fc2_b)}}};
float fc2[84];
float fc2_relu[84];
float fc3_w[10][84] = {{{arr2c(fc3_w)}}};
float fc3_b[10] = {{{arr2c(fc3_b)}}};
float fc3[10];
float output[10];

#include "{wrapped}"

int main() {{
  {fname}();
  for (int i = 0; i < 10; i++) printf("%.9g ", output[i]);
  printf("\\n");
  return 0;
}}
'''
    src = f'test_{out}.c'
    with open(src, 'w') as f:
        f.write(c_code)
    subprocess.run(['gcc', '-O2', '-std=c99', *flags, src, '-o', f'test_{out}', '-lm'], check=True)
    result = subprocess.run([f'./test_{out}'], capture_output=True, text=True, check=True)
    return np.array([float(x) for x in result.stdout.split()], dtype=np.float32)

seq_out = build_and_run('lenet5_wrapped.c', 'lenet5', [], 'lenet5_seq')
par_out = build_and_run('lenet5_par_wrapped.c', 'lenet5_par', ['-fopenmp'], 'lenet5_par')

print("Reference output: ", ref_output)
print("Sequential C out: ", seq_out)
print("OpenMP C out:     ", par_out)
max_err_ref = float(np.max(np.abs(ref_output - par_out)))
max_err_seq = float(np.max(np.abs(seq_out - par_out)))
print(f"Max |par - numpy_ref| = {max_err_ref:.3e}")
print(f"Max |par - seq|       = {max_err_seq:.3e}")
print("Match (numpy, 1e-4):", np.allclose(ref_output, par_out, atol=1e-4, rtol=1e-4))
print("Match (seq, exact): ", np.array_equal(seq_out, par_out))

# ---- timing ----
# In-process timing driver (2000 reps x 3 rounds, best-of) for accuracy.
timing_c = '''
#include <stdio.h>
#include <time.h>
float input[1][32][32];
float kc1[6][1][5][5]; float bc1[6];
float c1[6][28][28]; float r1[6][28][28]; float p1[6][14][14];
float kc2[16][6][5][5]; float bc2[16];
float c2[16][10][10]; float r2[16][10][10]; float p2[16][5][5];
float flat[400];
float fc1_w[120][400]; float fc1_b[120];
float fc1[120]; float fc1_relu[120];
float fc2_w[84][120]; float fc2_b[84];
float fc2[84]; float fc2_relu[84];
float fc3_w[10][84]; float fc3_b[10];
float fc3[10]; float output[10];
#include "%s"
static double now(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
int main() {
  int reps = 2000;
  %s();  /* warmup */
  double best = 1e30;
  for (int t = 0; t < 3; t++) {
    double t0 = now();
    for (int r = 0; r < reps; r++) %s();
    double dt = (now() - t0) / reps;
    if (dt < best) best = dt;
  }
  printf("%%.3f\\n", best * 1e6);  /* microseconds per inference */
  return 0;
}
'''
with open('time_seq.c', 'w') as f:
    f.write(timing_c % ('lenet5_wrapped.c', 'lenet5', 'lenet5'))
with open('time_par.c', 'w') as f:
    f.write(timing_c % ('lenet5_par_wrapped.c', 'lenet5_par', 'lenet5_par'))
subprocess.run(['gcc', '-O2', '-std=gnu99', 'time_seq.c', '-o', 'time_seq', '-lm'], check=True)
subprocess.run(['gcc', '-O2', '-std=gnu99', '-fopenmp', 'time_par.c', '-o', 'time_par', '-lm'], check=True)
t_seq = float(subprocess.run(['./time_seq'], capture_output=True, text=True, check=True).stdout.strip())
import os
# NOTE: a default (all-core) run is intentionally skipped here: with 16
# threads the 12 tiny parallel regions cost ~89 ms/inference (measured
# 2026-08-24), i.e. ~9 minutes for this
# driver alone.  Measure the useful configurations instead.
env4 = dict(os.environ, OMP_NUM_THREADS='4')
env2 = dict(os.environ, OMP_NUM_THREADS='2')
t_par4 = float(subprocess.run(['./time_par'], capture_output=True, text=True, check=True, env=env4).stdout.strip())
t_par2 = float(subprocess.run(['./time_par'], capture_output=True, text=True, check=True, env=env2).stdout.strip())
print(f"Sequential:            {t_seq:.3f} us/inference")
print(f"OpenMP (4 threads):    {t_par4:.3f} us/inference  (speedup {t_seq / t_par4:.2f}x)")
print(f"OpenMP (2 threads):    {t_par2:.3f} us/inference  (speedup {t_seq / t_par2:.2f}x)")
print("NOTE: LeNet-5 is tiny (~200 us sequential); with many threads the")
print("fork/join overhead of the 12 parallel regions dominates. Use")
print("OMP_NUM_THREADS=4 for a fair speedup measurement at this scale.")
