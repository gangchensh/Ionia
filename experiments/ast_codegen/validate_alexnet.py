#!/usr/bin/env python3
"""Numerical validation of the generated alexnet C code."""

import subprocess
import numpy as np

np.random.seed(123)
scale = 1e-3

INPUT   = (scale * np.random.randn(1, 3, 8, 8)).astype(np.float32)
CONV1_W = (scale * np.random.randn(8, 3, 3, 3)).astype(np.float32)
CONV1_B = (scale * np.random.randn(8)).astype(np.float32)
CONV2_W = (scale * np.random.randn(16, 8, 3, 3)).astype(np.float32)
CONV2_B = (scale * np.random.randn(16)).astype(np.float32)
FC1_W   = (scale * np.random.randn(16, 64)).astype(np.float32)
FC1_B   = (scale * np.random.randn(16)).astype(np.float32)
FC2_W   = (scale * np.random.randn(10, 16)).astype(np.float32)
FC2_B   = (scale * np.random.randn(10)).astype(np.float32)

def relu(x):
    return np.maximum(x, 0.0)

def conv2d_pad1_k3(x, w, b):
    N, Cin, H, W = x.shape
    Cout, _, K, _ = w.shape
    out = np.zeros((N, Cout, H, W), dtype=np.float32)
    padded = np.pad(x, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
    for n in range(N):
        for co in range(Cout):
            for i in range(H):
                for j in range(W):
                    s = 0.0
                    for ci in range(Cin):
                        for di in range(K):
                            for dj in range(K):
                                s += padded[n, ci, i+di, j+dj] * w[co, ci, di, dj]
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

def softmax2d(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def ref_forward():
    x = INPUT
    x = conv2d_pad1_k3(x, CONV1_W, CONV1_B)
    x = relu(x)
    x = maxpool2d_k2(x)
    x = conv2d_pad1_k3(x, CONV2_W, CONV2_B)
    x = relu(x)
    x = maxpool2d_k2(x)
    x = x.reshape(1, -1)
    x = relu(x @ FC1_W.T + FC1_B)
    x = x @ FC2_W.T + FC2_B
    x = softmax2d(x)
    return x

ref_output = ref_forward()

def arr2c(a, fmt='.9g'):
    return ', '.join(f'{v:{fmt}}f' for v in a.flatten())

c_code = f'''
#include <math.h>
#include <stdio.h>

float input[1][3][8][8] = {{{arr2c(INPUT)}}};
float conv1_w[8][3][3][3] = {{{arr2c(CONV1_W)}}};
float conv1_b[8] = {{{arr2c(CONV1_B)}}};
float conv1[1][8][8][8];
float conv1_relu[1][8][8][8];
float conv1_pool[1][8][4][4];
float conv2_w[16][8][3][3] = {{{arr2c(CONV2_W)}}};
float conv2_b[16] = {{{arr2c(CONV2_B)}}};
float conv2[1][16][4][4];
float conv2_relu[1][16][4][4];
float conv2_pool[1][16][2][2];
float flat[1][64];
float fc1_w[16][64] = {{{arr2c(FC1_W)}}};
float fc1_b[16] = {{{arr2c(FC1_B)}}};
float fc1[1][16];
float fc2_w[10][16] = {{{arr2c(FC2_W)}}};
float fc2_b[10] = {{{arr2c(FC2_B)}}};
float fc2[1][10];
float output[1][10];

#include "alexnet_wrapped.c"

int main() {{
  alexnet();
  for (int i = 0; i < 10; i++) printf("%.9g ", output[0][i]);
  printf("\\n");
  return 0;
}}
'''

with open('test_alexnet.c', 'w') as f:
    f.write(c_code)

subprocess.run(['gcc', '-std=c99', 'test_alexnet.c', '-o', 'test_alexnet', '-lm'], check=True)
result = subprocess.run(['./test_alexnet'], capture_output=True, text=True, check=True)
c_out = np.array([float(x) for x in result.stdout.split()], dtype=np.float32)

print("Reference output:", ref_output[0])
print("C output:        ", c_out)
print("Match:", np.allclose(ref_output[0], c_out, atol=1e-4, rtol=1e-4))
