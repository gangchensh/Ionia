#!/usr/bin/env python3
"""Numerical validation of the generated lenet5 C code."""

import subprocess
import numpy as np

np.random.seed(7)

# LeNet-5 shapes.  Keep values tiny so exp() in softmax does not overflow
# in the generated C, which does not subtract the max.
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
    # x: (1,Cin,H,W), w: (Cout,Cin,K,K), b: (Cout,)
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
    x = conv3d_no_pad(x, kc1, bc1)     # 1x6x28x28
    x = relu(x)
    x = maxpool2d_k2(x)                 # 1x6x14x14
    x = conv3d_no_pad(x, kc2, bc2)     # 1x16x10x10
    x = relu(x)
    x = maxpool2d_k2(x)                 # 1x16x5x5
    x = x.reshape(-1)                   # 400
    x = relu(x @ fc1_w.T + fc1_b)
    x = relu(x @ fc2_w.T + fc2_b)
    x = x @ fc3_w.T + fc3_b
    x = softmax1d(x)
    return x

ref_output = ref_forward()

def arr2c(a, fmt='.9g'):
    return ', '.join(f'{v:{fmt}}f' for v in a.flatten())

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

#include "lenet5_wrapped.c"

int main() {{
  lenet5();
  for (int i = 0; i < 10; i++) printf("%.9g ", output[i]);
  printf("\\n");
  return 0;
}}
'''

with open('test_lenet5.c', 'w') as f:
    f.write(c_code)

subprocess.run(['gcc', '-std=c99', 'test_lenet5.c', '-o', 'test_lenet5', '-lm'], check=True)
result = subprocess.run(['./test_lenet5'], capture_output=True, text=True, check=True)
c_out = np.array([float(x) for x in result.stdout.split()], dtype=np.float32)

print("Reference output:", ref_output)
print("C output:        ", c_out)
print("Match:", np.allclose(ref_output, c_out, atol=1e-4, rtol=1e-4))
