#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1); \
    } \
  } while (0)

__constant__ float (*input)[3][32][32];
__constant__ float (*conv1_1_w)[3][3][3];
__constant__ float* conv1_1_b;
__constant__ float (*conv1_2_w)[64][3][3];
__constant__ float* conv1_2_b;
__constant__ float (*conv2_1_w)[64][3][3];
__constant__ float* conv2_1_b;
__constant__ float (*conv2_2_w)[128][3][3];
__constant__ float* conv2_2_b;
__constant__ float (*conv3_1_w)[128][3][3];
__constant__ float* conv3_1_b;
__constant__ float (*conv3_2_w)[256][3][3];
__constant__ float* conv3_2_b;
__constant__ float (*conv3_3_w)[256][3][3];
__constant__ float* conv3_3_b;
__constant__ float (*conv4_1_w)[256][3][3];
__constant__ float* conv4_1_b;
__constant__ float (*conv4_2_w)[512][3][3];
__constant__ float* conv4_2_b;
__constant__ float (*conv4_3_w)[512][3][3];
__constant__ float* conv4_3_b;
__constant__ float (*conv5_1_w)[512][3][3];
__constant__ float* conv5_1_b;
__constant__ float (*conv5_2_w)[512][3][3];
__constant__ float* conv5_2_b;
__constant__ float (*conv5_3_w)[512][3][3];
__constant__ float* conv5_3_b;
__constant__ float (*fc1_w)[512];
__constant__ float* fc1_b;
__constant__ float (*fc2_w)[4096];
__constant__ float* fc2_b;
__constant__ float (*fc3_w)[4096];
__constant__ float* fc3_b;
__constant__ float (*output)[1000];
__constant__ float (*relu1_1_out)[64][32][32];
__constant__ float (*relu1_2_out)[64][32][32];
__constant__ float (*pool1_out)[64][16][16];
__constant__ float (*relu2_1_out)[128][16][16];
__constant__ float (*relu2_2_out)[128][16][16];
__constant__ float (*pool2_out)[128][8][8];
__constant__ float (*relu3_1_out)[256][8][8];
__constant__ float (*relu3_2_out)[256][8][8];
__constant__ float (*relu3_3_out)[256][8][8];
__constant__ float (*pool3_out)[256][4][4];
__constant__ float (*relu4_1_out)[512][4][4];
__constant__ float (*relu4_2_out)[512][4][4];
__constant__ float (*relu4_3_out)[512][4][4];
__constant__ float (*pool4_out)[512][2][2];
__constant__ float (*relu5_1_out)[512][2][2];
__constant__ float (*relu5_2_out)[512][2][2];
__constant__ float (*relu5_3_out)[512][2][2];
__constant__ float (*pool5_out)[512][1][1];
__constant__ float (*flatten_out)[512];
__constant__ float (*fc1_relu_out)[4096];
__constant__ float (*fc2_relu_out)[4096];

#include <cuda_runtime.h>
#include <math.h>

__global__ void ast_relu1_1_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 65536;
  if (idx >= N) return;
  int i3 = idx % 32; idx /= 32;
  int i2 = idx % 32; idx /= 32;
  int i1 = idx % 64; idx /= 64;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 3; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 33 || pj < 1 || pj >= 33)
                    ? 0.0f
                    : input[i0][c][pi - 1][pj - 1];
        acc += v * conv1_1_w[i1][c][k1][k2];
      }
  float result = acc + conv1_1_b[i1];
  relu1_1_out[i0][i1][i2][i3] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_relu1_2_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 65536;
  if (idx >= N) return;
  int i3 = idx % 32; idx /= 32;
  int i2 = idx % 32; idx /= 32;
  int i1 = idx % 64; idx /= 64;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 64; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 33 || pj < 1 || pj >= 33)
                    ? 0.0f
                    : relu1_1_out[i0][c][pi - 1][pj - 1];
        acc += v * conv1_2_w[i1][c][k1][k2];
      }
  float result = acc + conv1_2_b[i1];
  relu1_2_out[i0][i1][i2][i3] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_pool1_out()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 16384;
  if (idx >= N) return;
  int i3 = idx % 16; idx /= 16;
  int i2 = idx % 16; idx /= 16;
  int i1 = idx % 64; idx /= 64;
  int i0 = idx;
  float best = relu1_2_out[i0][i1][2 * i2][2 * i3];
  for (int k1 = 0; k1 < 2; k1 += 1)
    for (int k2 = 0; k2 < 2; k2 += 1) {
      float v = relu1_2_out[i0][i1][2 * i2 + k1][2 * i3 + k2];
      if (v > best) best = v;
    }
  pool1_out[i0][i1][i2][i3] = best;
}

__global__ void ast_relu2_1_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 32768;
  if (idx >= N) return;
  int i3 = idx % 16; idx /= 16;
  int i2 = idx % 16; idx /= 16;
  int i1 = idx % 128; idx /= 128;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 64; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 17 || pj < 1 || pj >= 17)
                    ? 0.0f
                    : relu1_2_out[i0][c][pi - 1][pj - 1];
        acc += v * conv2_1_w[i1][c][k1][k2];
      }
  float result = acc + conv2_1_b[i1];
  relu2_1_out[i0][i1][i2][i3] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_relu2_2_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 32768;
  if (idx >= N) return;
  int i3 = idx % 16; idx /= 16;
  int i2 = idx % 16; idx /= 16;
  int i1 = idx % 128; idx /= 128;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 128; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 17 || pj < 1 || pj >= 17)
                    ? 0.0f
                    : relu2_1_out[i0][c][pi - 1][pj - 1];
        acc += v * conv2_2_w[i1][c][k1][k2];
      }
  float result = acc + conv2_2_b[i1];
  relu2_2_out[i0][i1][i2][i3] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_pool2_out()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 8192;
  if (idx >= N) return;
  int i3 = idx % 8; idx /= 8;
  int i2 = idx % 8; idx /= 8;
  int i1 = idx % 128; idx /= 128;
  int i0 = idx;
  float best = relu2_2_out[i0][i1][2 * i2][2 * i3];
  for (int k1 = 0; k1 < 2; k1 += 1)
    for (int k2 = 0; k2 < 2; k2 += 1) {
      float v = relu2_2_out[i0][i1][2 * i2 + k1][2 * i3 + k2];
      if (v > best) best = v;
    }
  pool2_out[i0][i1][i2][i3] = best;
}

__global__ void ast_relu3_1_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 16384;
  if (idx >= N) return;
  int i3 = idx % 8; idx /= 8;
  int i2 = idx % 8; idx /= 8;
  int i1 = idx % 256; idx /= 256;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 128; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 9 || pj < 1 || pj >= 9)
                    ? 0.0f
                    : relu2_2_out[i0][c][pi - 1][pj - 1];
        acc += v * conv3_1_w[i1][c][k1][k2];
      }
  float result = acc + conv3_1_b[i1];
  relu3_1_out[i0][i1][i2][i3] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_relu3_2_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 16384;
  if (idx >= N) return;
  int i3 = idx % 8; idx /= 8;
  int i2 = idx % 8; idx /= 8;
  int i1 = idx % 256; idx /= 256;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 256; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 9 || pj < 1 || pj >= 9)
                    ? 0.0f
                    : relu3_1_out[i0][c][pi - 1][pj - 1];
        acc += v * conv3_2_w[i1][c][k1][k2];
      }
  float result = acc + conv3_2_b[i1];
  relu3_2_out[i0][i1][i2][i3] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_relu3_3_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 16384;
  if (idx >= N) return;
  int i3 = idx % 8; idx /= 8;
  int i2 = idx % 8; idx /= 8;
  int i1 = idx % 256; idx /= 256;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 256; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 9 || pj < 1 || pj >= 9)
                    ? 0.0f
                    : relu3_2_out[i0][c][pi - 1][pj - 1];
        acc += v * conv3_3_w[i1][c][k1][k2];
      }
  float result = acc + conv3_3_b[i1];
  relu3_3_out[i0][i1][i2][i3] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_pool3_out()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 4096;
  if (idx >= N) return;
  int i3 = idx % 4; idx /= 4;
  int i2 = idx % 4; idx /= 4;
  int i1 = idx % 256; idx /= 256;
  int i0 = idx;
  float best = relu3_3_out[i0][i1][2 * i2][2 * i3];
  for (int k1 = 0; k1 < 2; k1 += 1)
    for (int k2 = 0; k2 < 2; k2 += 1) {
      float v = relu3_3_out[i0][i1][2 * i2 + k1][2 * i3 + k2];
      if (v > best) best = v;
    }
  pool3_out[i0][i1][i2][i3] = best;
}

__global__ void ast_relu4_1_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 8192;
  if (idx >= N) return;
  int i3 = idx % 4; idx /= 4;
  int i2 = idx % 4; idx /= 4;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 256; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 5 || pj < 1 || pj >= 5)
                    ? 0.0f
                    : relu3_3_out[i0][c][pi - 1][pj - 1];
        acc += v * conv4_1_w[i1][c][k1][k2];
      }
  float result = acc + conv4_1_b[i1];
  relu4_1_out[i0][i1][i2][i3] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_relu4_2_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 8192;
  if (idx >= N) return;
  int i3 = idx % 4; idx /= 4;
  int i2 = idx % 4; idx /= 4;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 512; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 5 || pj < 1 || pj >= 5)
                    ? 0.0f
                    : relu4_1_out[i0][c][pi - 1][pj - 1];
        acc += v * conv4_2_w[i1][c][k1][k2];
      }
  float result = acc + conv4_2_b[i1];
  relu4_2_out[i0][i1][i2][i3] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_relu4_3_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 8192;
  if (idx >= N) return;
  int i3 = idx % 4; idx /= 4;
  int i2 = idx % 4; idx /= 4;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 512; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 5 || pj < 1 || pj >= 5)
                    ? 0.0f
                    : relu4_2_out[i0][c][pi - 1][pj - 1];
        acc += v * conv4_3_w[i1][c][k1][k2];
      }
  float result = acc + conv4_3_b[i1];
  relu4_3_out[i0][i1][i2][i3] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_pool4_out()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 2048;
  if (idx >= N) return;
  int i3 = idx % 2; idx /= 2;
  int i2 = idx % 2; idx /= 2;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float best = relu4_3_out[i0][i1][2 * i2][2 * i3];
  for (int k1 = 0; k1 < 2; k1 += 1)
    for (int k2 = 0; k2 < 2; k2 += 1) {
      float v = relu4_3_out[i0][i1][2 * i2 + k1][2 * i3 + k2];
      if (v > best) best = v;
    }
  pool4_out[i0][i1][i2][i3] = best;
}

__global__ void ast_relu5_1_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 2048;
  if (idx >= N) return;
  int i3 = idx % 2; idx /= 2;
  int i2 = idx % 2; idx /= 2;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 512; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 3 || pj < 1 || pj >= 3)
                    ? 0.0f
                    : relu4_3_out[i0][c][pi - 1][pj - 1];
        acc += v * conv5_1_w[i1][c][k1][k2];
      }
  float result = acc + conv5_1_b[i1];
  relu5_1_out[i0][i1][i2][i3] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_relu5_2_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 2048;
  if (idx >= N) return;
  int i3 = idx % 2; idx /= 2;
  int i2 = idx % 2; idx /= 2;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 512; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 3 || pj < 1 || pj >= 3)
                    ? 0.0f
                    : relu5_1_out[i0][c][pi - 1][pj - 1];
        acc += v * conv5_2_w[i1][c][k1][k2];
      }
  float result = acc + conv5_2_b[i1];
  relu5_2_out[i0][i1][i2][i3] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_relu5_3_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 2048;
  if (idx >= N) return;
  int i3 = idx % 2; idx /= 2;
  int i2 = idx % 2; idx /= 2;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 512; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 3 || pj < 1 || pj >= 3)
                    ? 0.0f
                    : relu5_2_out[i0][c][pi - 1][pj - 1];
        acc += v * conv5_3_w[i1][c][k1][k2];
      }
  float result = acc + conv5_3_b[i1];
  relu5_3_out[i0][i1][i2][i3] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_pool5_out()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 512;
  if (idx >= N) return;
  int i3 = idx % 1; idx /= 1;
  int i2 = idx % 1; idx /= 1;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float best = relu5_3_out[i0][i1][2 * i2][2 * i3];
  for (int k1 = 0; k1 < 2; k1 += 1)
    for (int k2 = 0; k2 < 2; k2 += 1) {
      float v = relu5_3_out[i0][i1][2 * i2 + k1][2 * i3 + k2];
      if (v > best) best = v;
    }
  pool5_out[i0][i1][i2][i3] = best;
}

__global__ void ast_flatten_out()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 512;
  if (idx >= N) return;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  int hw = 1;
  int c = i1 / hw;
  int rem = i1 % hw;
  int h0 = rem / 1;
  int w0 = rem % 1;
  flatten_out[i0][i1] = pool5_out[i0][c][h0][w0];
}

__global__ void ast_fc1_relu_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 4096;
  if (idx >= N) return;
  int i1 = idx % 4096; idx /= 4096;
  int i0 = idx;
  float acc = 0.0f;
  for (int i = 0; i < 512; i += 1)
    acc += flatten_out[i0][i] * fc1_w[i1][i];
  float result = acc + fc1_b[i1];
  fc1_relu_out[i0][i1] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_fc2_relu_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 4096;
  if (idx >= N) return;
  int i1 = idx % 4096; idx /= 4096;
  int i0 = idx;
  float acc = 0.0f;
  for (int i = 0; i < 4096; i += 1)
    acc += fc1_relu_out[i0][i] * fc2_w[i1][i];
  float result = acc + fc2_b[i1];
  fc2_relu_out[i0][i1] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_output()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 1000;
  if (idx >= N) return;
  int i1 = idx % 1000; idx /= 1000;
  int i0 = idx;
  float acc = 0.0f;
  for (int i = 0; i < 4096; i += 1)
    acc += fc2_relu_out[i0][i] * fc3_w[i1][i];
  output[i0][i1] = acc + fc3_b[i1];
}

void vgg16s_test()
{
  ast_relu1_1_out_fused<<<256, 256>>>();
  cudaDeviceSynchronize();
  ast_relu1_2_out_fused<<<256, 256>>>();
  cudaDeviceSynchronize();
  ast_pool1_out<<<64, 256>>>();
  cudaDeviceSynchronize();
  ast_relu2_1_out_fused<<<128, 256>>>();
  cudaDeviceSynchronize();
  ast_relu2_2_out_fused<<<128, 256>>>();
  cudaDeviceSynchronize();
  ast_pool2_out<<<32, 256>>>();
  cudaDeviceSynchronize();
  ast_relu3_1_out_fused<<<64, 256>>>();
  cudaDeviceSynchronize();
  ast_relu3_2_out_fused<<<64, 256>>>();
  cudaDeviceSynchronize();
  ast_relu3_3_out_fused<<<64, 256>>>();
  cudaDeviceSynchronize();
  ast_pool3_out<<<16, 256>>>();
  cudaDeviceSynchronize();
  ast_relu4_1_out_fused<<<32, 256>>>();
  cudaDeviceSynchronize();
  ast_relu4_2_out_fused<<<32, 256>>>();
  cudaDeviceSynchronize();
  ast_relu4_3_out_fused<<<32, 256>>>();
  cudaDeviceSynchronize();
  ast_pool4_out<<<8, 256>>>();
  cudaDeviceSynchronize();
  ast_relu5_1_out_fused<<<8, 256>>>();
  cudaDeviceSynchronize();
  ast_relu5_2_out_fused<<<8, 256>>>();
  cudaDeviceSynchronize();
  ast_relu5_3_out_fused<<<8, 256>>>();
  cudaDeviceSynchronize();
  ast_pool5_out<<<2, 256>>>();
  cudaDeviceSynchronize();
  ast_flatten_out<<<2, 256>>>();
  cudaDeviceSynchronize();
  ast_fc1_relu_out_fused<<<16, 256>>>();
  cudaDeviceSynchronize();
  ast_fc2_relu_out_fused<<<16, 256>>>();
  cudaDeviceSynchronize();
  ast_output<<<4, 256>>>();
  cudaDeviceSynchronize();
}


int main(int argc, char** argv)
{
  size_t input_bytes = 12288;
  float* d_input = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_input, input_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(input, &d_input, sizeof(d_input)));
  size_t conv1_1_w_bytes = 6912;
  float* d_conv1_1_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv1_1_w, conv1_1_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv1_1_w, &d_conv1_1_w, sizeof(d_conv1_1_w)));
  size_t conv1_1_b_bytes = 256;
  float* d_conv1_1_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv1_1_b, conv1_1_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv1_1_b, &d_conv1_1_b, sizeof(d_conv1_1_b)));
  size_t conv1_2_w_bytes = 147456;
  float* d_conv1_2_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv1_2_w, conv1_2_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv1_2_w, &d_conv1_2_w, sizeof(d_conv1_2_w)));
  size_t conv1_2_b_bytes = 256;
  float* d_conv1_2_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv1_2_b, conv1_2_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv1_2_b, &d_conv1_2_b, sizeof(d_conv1_2_b)));
  size_t conv2_1_w_bytes = 294912;
  float* d_conv2_1_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv2_1_w, conv2_1_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv2_1_w, &d_conv2_1_w, sizeof(d_conv2_1_w)));
  size_t conv2_1_b_bytes = 512;
  float* d_conv2_1_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv2_1_b, conv2_1_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv2_1_b, &d_conv2_1_b, sizeof(d_conv2_1_b)));
  size_t conv2_2_w_bytes = 589824;
  float* d_conv2_2_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv2_2_w, conv2_2_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv2_2_w, &d_conv2_2_w, sizeof(d_conv2_2_w)));
  size_t conv2_2_b_bytes = 512;
  float* d_conv2_2_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv2_2_b, conv2_2_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv2_2_b, &d_conv2_2_b, sizeof(d_conv2_2_b)));
  size_t conv3_1_w_bytes = 1179648;
  float* d_conv3_1_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv3_1_w, conv3_1_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv3_1_w, &d_conv3_1_w, sizeof(d_conv3_1_w)));
  size_t conv3_1_b_bytes = 1024;
  float* d_conv3_1_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv3_1_b, conv3_1_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv3_1_b, &d_conv3_1_b, sizeof(d_conv3_1_b)));
  size_t conv3_2_w_bytes = 2359296;
  float* d_conv3_2_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv3_2_w, conv3_2_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv3_2_w, &d_conv3_2_w, sizeof(d_conv3_2_w)));
  size_t conv3_2_b_bytes = 1024;
  float* d_conv3_2_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv3_2_b, conv3_2_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv3_2_b, &d_conv3_2_b, sizeof(d_conv3_2_b)));
  size_t conv3_3_w_bytes = 2359296;
  float* d_conv3_3_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv3_3_w, conv3_3_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv3_3_w, &d_conv3_3_w, sizeof(d_conv3_3_w)));
  size_t conv3_3_b_bytes = 1024;
  float* d_conv3_3_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv3_3_b, conv3_3_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv3_3_b, &d_conv3_3_b, sizeof(d_conv3_3_b)));
  size_t conv4_1_w_bytes = 4718592;
  float* d_conv4_1_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv4_1_w, conv4_1_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv4_1_w, &d_conv4_1_w, sizeof(d_conv4_1_w)));
  size_t conv4_1_b_bytes = 2048;
  float* d_conv4_1_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv4_1_b, conv4_1_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv4_1_b, &d_conv4_1_b, sizeof(d_conv4_1_b)));
  size_t conv4_2_w_bytes = 9437184;
  float* d_conv4_2_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv4_2_w, conv4_2_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv4_2_w, &d_conv4_2_w, sizeof(d_conv4_2_w)));
  size_t conv4_2_b_bytes = 2048;
  float* d_conv4_2_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv4_2_b, conv4_2_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv4_2_b, &d_conv4_2_b, sizeof(d_conv4_2_b)));
  size_t conv4_3_w_bytes = 9437184;
  float* d_conv4_3_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv4_3_w, conv4_3_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv4_3_w, &d_conv4_3_w, sizeof(d_conv4_3_w)));
  size_t conv4_3_b_bytes = 2048;
  float* d_conv4_3_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv4_3_b, conv4_3_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv4_3_b, &d_conv4_3_b, sizeof(d_conv4_3_b)));
  size_t conv5_1_w_bytes = 9437184;
  float* d_conv5_1_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv5_1_w, conv5_1_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv5_1_w, &d_conv5_1_w, sizeof(d_conv5_1_w)));
  size_t conv5_1_b_bytes = 2048;
  float* d_conv5_1_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv5_1_b, conv5_1_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv5_1_b, &d_conv5_1_b, sizeof(d_conv5_1_b)));
  size_t conv5_2_w_bytes = 9437184;
  float* d_conv5_2_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv5_2_w, conv5_2_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv5_2_w, &d_conv5_2_w, sizeof(d_conv5_2_w)));
  size_t conv5_2_b_bytes = 2048;
  float* d_conv5_2_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv5_2_b, conv5_2_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv5_2_b, &d_conv5_2_b, sizeof(d_conv5_2_b)));
  size_t conv5_3_w_bytes = 9437184;
  float* d_conv5_3_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv5_3_w, conv5_3_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv5_3_w, &d_conv5_3_w, sizeof(d_conv5_3_w)));
  size_t conv5_3_b_bytes = 2048;
  float* d_conv5_3_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv5_3_b, conv5_3_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv5_3_b, &d_conv5_3_b, sizeof(d_conv5_3_b)));
  size_t fc1_w_bytes = 8388608;
  float* d_fc1_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_fc1_w, fc1_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(fc1_w, &d_fc1_w, sizeof(d_fc1_w)));
  size_t fc1_b_bytes = 16384;
  float* d_fc1_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_fc1_b, fc1_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(fc1_b, &d_fc1_b, sizeof(d_fc1_b)));
  size_t fc2_w_bytes = 67108864;
  float* d_fc2_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_fc2_w, fc2_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(fc2_w, &d_fc2_w, sizeof(d_fc2_w)));
  size_t fc2_b_bytes = 16384;
  float* d_fc2_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_fc2_b, fc2_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(fc2_b, &d_fc2_b, sizeof(d_fc2_b)));
  size_t fc3_w_bytes = 16384000;
  float* d_fc3_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_fc3_w, fc3_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(fc3_w, &d_fc3_w, sizeof(d_fc3_w)));
  size_t fc3_b_bytes = 4000;
  float* d_fc3_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_fc3_b, fc3_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(fc3_b, &d_fc3_b, sizeof(d_fc3_b)));
  size_t output_bytes = 4000;
  float* d_output = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_output, output_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(output, &d_output, sizeof(d_output)));
  size_t relu1_1_out_bytes = 262144;
  float* d_relu1_1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu1_1_out, relu1_1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu1_1_out, &d_relu1_1_out, sizeof(d_relu1_1_out)));
  size_t relu1_2_out_bytes = 262144;
  float* d_relu1_2_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu1_2_out, relu1_2_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu1_2_out, &d_relu1_2_out, sizeof(d_relu1_2_out)));
  size_t pool1_out_bytes = 65536;
  float* d_pool1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_pool1_out, pool1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(pool1_out, &d_pool1_out, sizeof(d_pool1_out)));
  size_t relu2_1_out_bytes = 131072;
  float* d_relu2_1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu2_1_out, relu2_1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu2_1_out, &d_relu2_1_out, sizeof(d_relu2_1_out)));
  size_t relu2_2_out_bytes = 131072;
  float* d_relu2_2_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu2_2_out, relu2_2_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu2_2_out, &d_relu2_2_out, sizeof(d_relu2_2_out)));
  size_t pool2_out_bytes = 32768;
  float* d_pool2_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_pool2_out, pool2_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(pool2_out, &d_pool2_out, sizeof(d_pool2_out)));
  size_t relu3_1_out_bytes = 65536;
  float* d_relu3_1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu3_1_out, relu3_1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu3_1_out, &d_relu3_1_out, sizeof(d_relu3_1_out)));
  size_t relu3_2_out_bytes = 65536;
  float* d_relu3_2_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu3_2_out, relu3_2_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu3_2_out, &d_relu3_2_out, sizeof(d_relu3_2_out)));
  size_t relu3_3_out_bytes = 65536;
  float* d_relu3_3_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu3_3_out, relu3_3_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu3_3_out, &d_relu3_3_out, sizeof(d_relu3_3_out)));
  size_t pool3_out_bytes = 16384;
  float* d_pool3_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_pool3_out, pool3_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(pool3_out, &d_pool3_out, sizeof(d_pool3_out)));
  size_t relu4_1_out_bytes = 32768;
  float* d_relu4_1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu4_1_out, relu4_1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu4_1_out, &d_relu4_1_out, sizeof(d_relu4_1_out)));
  size_t relu4_2_out_bytes = 32768;
  float* d_relu4_2_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu4_2_out, relu4_2_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu4_2_out, &d_relu4_2_out, sizeof(d_relu4_2_out)));
  size_t relu4_3_out_bytes = 32768;
  float* d_relu4_3_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu4_3_out, relu4_3_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu4_3_out, &d_relu4_3_out, sizeof(d_relu4_3_out)));
  size_t pool4_out_bytes = 8192;
  float* d_pool4_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_pool4_out, pool4_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(pool4_out, &d_pool4_out, sizeof(d_pool4_out)));
  size_t relu5_1_out_bytes = 8192;
  float* d_relu5_1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu5_1_out, relu5_1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu5_1_out, &d_relu5_1_out, sizeof(d_relu5_1_out)));
  size_t relu5_2_out_bytes = 8192;
  float* d_relu5_2_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu5_2_out, relu5_2_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu5_2_out, &d_relu5_2_out, sizeof(d_relu5_2_out)));
  size_t relu5_3_out_bytes = 8192;
  float* d_relu5_3_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu5_3_out, relu5_3_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu5_3_out, &d_relu5_3_out, sizeof(d_relu5_3_out)));
  size_t pool5_out_bytes = 2048;
  float* d_pool5_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_pool5_out, pool5_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(pool5_out, &d_pool5_out, sizeof(d_pool5_out)));
  size_t flatten_out_bytes = 2048;
  float* d_flatten_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_flatten_out, flatten_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(flatten_out, &d_flatten_out, sizeof(d_flatten_out)));
  size_t fc1_relu_out_bytes = 16384;
  float* d_fc1_relu_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_fc1_relu_out, fc1_relu_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(fc1_relu_out, &d_fc1_relu_out, sizeof(d_fc1_relu_out)));
  size_t fc2_relu_out_bytes = 16384;
  float* d_fc2_relu_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_fc2_relu_out, fc2_relu_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(fc2_relu_out, &d_fc2_relu_out, sizeof(d_fc2_relu_out)));

  cudaEvent_t __io_start, __io_stop;
  CUDA_CHECK(cudaEventCreate(&__io_start));
  CUDA_CHECK(cudaEventCreate(&__io_stop));
  CUDA_CHECK(cudaEventRecord(__io_start));
  float* h_input = (float*)malloc(12288);
  if (!h_input) { fprintf(stderr, "malloc failed for input\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_input.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_input.bin\n"); return 1; }
    size_t nread = fread(h_input, sizeof(float), 3072, f);
    fclose(f);
    if (nread != 3072) { fprintf(stderr, "Failed to read vgg16s_test_input.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_input, h_input, 12288, cudaMemcpyHostToDevice));
  float* h_conv1_1_w = (float*)malloc(6912);
  if (!h_conv1_1_w) { fprintf(stderr, "malloc failed for conv1_1_w\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv1_1_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv1_1_w.bin\n"); return 1; }
    size_t nread = fread(h_conv1_1_w, sizeof(float), 1728, f);
    fclose(f);
    if (nread != 1728) { fprintf(stderr, "Failed to read vgg16s_test_conv1_1_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv1_1_w, h_conv1_1_w, 6912, cudaMemcpyHostToDevice));
  float* h_conv1_1_b = (float*)malloc(256);
  if (!h_conv1_1_b) { fprintf(stderr, "malloc failed for conv1_1_b\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv1_1_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv1_1_b.bin\n"); return 1; }
    size_t nread = fread(h_conv1_1_b, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read vgg16s_test_conv1_1_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv1_1_b, h_conv1_1_b, 256, cudaMemcpyHostToDevice));
  float* h_conv1_2_w = (float*)malloc(147456);
  if (!h_conv1_2_w) { fprintf(stderr, "malloc failed for conv1_2_w\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv1_2_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv1_2_w.bin\n"); return 1; }
    size_t nread = fread(h_conv1_2_w, sizeof(float), 36864, f);
    fclose(f);
    if (nread != 36864) { fprintf(stderr, "Failed to read vgg16s_test_conv1_2_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv1_2_w, h_conv1_2_w, 147456, cudaMemcpyHostToDevice));
  float* h_conv1_2_b = (float*)malloc(256);
  if (!h_conv1_2_b) { fprintf(stderr, "malloc failed for conv1_2_b\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv1_2_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv1_2_b.bin\n"); return 1; }
    size_t nread = fread(h_conv1_2_b, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read vgg16s_test_conv1_2_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv1_2_b, h_conv1_2_b, 256, cudaMemcpyHostToDevice));
  float* h_conv2_1_w = (float*)malloc(294912);
  if (!h_conv2_1_w) { fprintf(stderr, "malloc failed for conv2_1_w\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv2_1_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv2_1_w.bin\n"); return 1; }
    size_t nread = fread(h_conv2_1_w, sizeof(float), 73728, f);
    fclose(f);
    if (nread != 73728) { fprintf(stderr, "Failed to read vgg16s_test_conv2_1_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv2_1_w, h_conv2_1_w, 294912, cudaMemcpyHostToDevice));
  float* h_conv2_1_b = (float*)malloc(512);
  if (!h_conv2_1_b) { fprintf(stderr, "malloc failed for conv2_1_b\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv2_1_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv2_1_b.bin\n"); return 1; }
    size_t nread = fread(h_conv2_1_b, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read vgg16s_test_conv2_1_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv2_1_b, h_conv2_1_b, 512, cudaMemcpyHostToDevice));
  float* h_conv2_2_w = (float*)malloc(589824);
  if (!h_conv2_2_w) { fprintf(stderr, "malloc failed for conv2_2_w\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv2_2_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv2_2_w.bin\n"); return 1; }
    size_t nread = fread(h_conv2_2_w, sizeof(float), 147456, f);
    fclose(f);
    if (nread != 147456) { fprintf(stderr, "Failed to read vgg16s_test_conv2_2_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv2_2_w, h_conv2_2_w, 589824, cudaMemcpyHostToDevice));
  float* h_conv2_2_b = (float*)malloc(512);
  if (!h_conv2_2_b) { fprintf(stderr, "malloc failed for conv2_2_b\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv2_2_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv2_2_b.bin\n"); return 1; }
    size_t nread = fread(h_conv2_2_b, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read vgg16s_test_conv2_2_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv2_2_b, h_conv2_2_b, 512, cudaMemcpyHostToDevice));
  float* h_conv3_1_w = (float*)malloc(1179648);
  if (!h_conv3_1_w) { fprintf(stderr, "malloc failed for conv3_1_w\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv3_1_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv3_1_w.bin\n"); return 1; }
    size_t nread = fread(h_conv3_1_w, sizeof(float), 294912, f);
    fclose(f);
    if (nread != 294912) { fprintf(stderr, "Failed to read vgg16s_test_conv3_1_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv3_1_w, h_conv3_1_w, 1179648, cudaMemcpyHostToDevice));
  float* h_conv3_1_b = (float*)malloc(1024);
  if (!h_conv3_1_b) { fprintf(stderr, "malloc failed for conv3_1_b\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv3_1_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv3_1_b.bin\n"); return 1; }
    size_t nread = fread(h_conv3_1_b, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read vgg16s_test_conv3_1_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv3_1_b, h_conv3_1_b, 1024, cudaMemcpyHostToDevice));
  float* h_conv3_2_w = (float*)malloc(2359296);
  if (!h_conv3_2_w) { fprintf(stderr, "malloc failed for conv3_2_w\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv3_2_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv3_2_w.bin\n"); return 1; }
    size_t nread = fread(h_conv3_2_w, sizeof(float), 589824, f);
    fclose(f);
    if (nread != 589824) { fprintf(stderr, "Failed to read vgg16s_test_conv3_2_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv3_2_w, h_conv3_2_w, 2359296, cudaMemcpyHostToDevice));
  float* h_conv3_2_b = (float*)malloc(1024);
  if (!h_conv3_2_b) { fprintf(stderr, "malloc failed for conv3_2_b\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv3_2_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv3_2_b.bin\n"); return 1; }
    size_t nread = fread(h_conv3_2_b, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read vgg16s_test_conv3_2_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv3_2_b, h_conv3_2_b, 1024, cudaMemcpyHostToDevice));
  float* h_conv3_3_w = (float*)malloc(2359296);
  if (!h_conv3_3_w) { fprintf(stderr, "malloc failed for conv3_3_w\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv3_3_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv3_3_w.bin\n"); return 1; }
    size_t nread = fread(h_conv3_3_w, sizeof(float), 589824, f);
    fclose(f);
    if (nread != 589824) { fprintf(stderr, "Failed to read vgg16s_test_conv3_3_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv3_3_w, h_conv3_3_w, 2359296, cudaMemcpyHostToDevice));
  float* h_conv3_3_b = (float*)malloc(1024);
  if (!h_conv3_3_b) { fprintf(stderr, "malloc failed for conv3_3_b\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv3_3_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv3_3_b.bin\n"); return 1; }
    size_t nread = fread(h_conv3_3_b, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read vgg16s_test_conv3_3_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv3_3_b, h_conv3_3_b, 1024, cudaMemcpyHostToDevice));
  float* h_conv4_1_w = (float*)malloc(4718592);
  if (!h_conv4_1_w) { fprintf(stderr, "malloc failed for conv4_1_w\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv4_1_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv4_1_w.bin\n"); return 1; }
    size_t nread = fread(h_conv4_1_w, sizeof(float), 1179648, f);
    fclose(f);
    if (nread != 1179648) { fprintf(stderr, "Failed to read vgg16s_test_conv4_1_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv4_1_w, h_conv4_1_w, 4718592, cudaMemcpyHostToDevice));
  float* h_conv4_1_b = (float*)malloc(2048);
  if (!h_conv4_1_b) { fprintf(stderr, "malloc failed for conv4_1_b\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv4_1_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv4_1_b.bin\n"); return 1; }
    size_t nread = fread(h_conv4_1_b, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read vgg16s_test_conv4_1_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv4_1_b, h_conv4_1_b, 2048, cudaMemcpyHostToDevice));
  float* h_conv4_2_w = (float*)malloc(9437184);
  if (!h_conv4_2_w) { fprintf(stderr, "malloc failed for conv4_2_w\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv4_2_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv4_2_w.bin\n"); return 1; }
    size_t nread = fread(h_conv4_2_w, sizeof(float), 2359296, f);
    fclose(f);
    if (nread != 2359296) { fprintf(stderr, "Failed to read vgg16s_test_conv4_2_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv4_2_w, h_conv4_2_w, 9437184, cudaMemcpyHostToDevice));
  float* h_conv4_2_b = (float*)malloc(2048);
  if (!h_conv4_2_b) { fprintf(stderr, "malloc failed for conv4_2_b\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv4_2_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv4_2_b.bin\n"); return 1; }
    size_t nread = fread(h_conv4_2_b, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read vgg16s_test_conv4_2_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv4_2_b, h_conv4_2_b, 2048, cudaMemcpyHostToDevice));
  float* h_conv4_3_w = (float*)malloc(9437184);
  if (!h_conv4_3_w) { fprintf(stderr, "malloc failed for conv4_3_w\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv4_3_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv4_3_w.bin\n"); return 1; }
    size_t nread = fread(h_conv4_3_w, sizeof(float), 2359296, f);
    fclose(f);
    if (nread != 2359296) { fprintf(stderr, "Failed to read vgg16s_test_conv4_3_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv4_3_w, h_conv4_3_w, 9437184, cudaMemcpyHostToDevice));
  float* h_conv4_3_b = (float*)malloc(2048);
  if (!h_conv4_3_b) { fprintf(stderr, "malloc failed for conv4_3_b\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv4_3_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv4_3_b.bin\n"); return 1; }
    size_t nread = fread(h_conv4_3_b, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read vgg16s_test_conv4_3_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv4_3_b, h_conv4_3_b, 2048, cudaMemcpyHostToDevice));
  float* h_conv5_1_w = (float*)malloc(9437184);
  if (!h_conv5_1_w) { fprintf(stderr, "malloc failed for conv5_1_w\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv5_1_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv5_1_w.bin\n"); return 1; }
    size_t nread = fread(h_conv5_1_w, sizeof(float), 2359296, f);
    fclose(f);
    if (nread != 2359296) { fprintf(stderr, "Failed to read vgg16s_test_conv5_1_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv5_1_w, h_conv5_1_w, 9437184, cudaMemcpyHostToDevice));
  float* h_conv5_1_b = (float*)malloc(2048);
  if (!h_conv5_1_b) { fprintf(stderr, "malloc failed for conv5_1_b\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv5_1_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv5_1_b.bin\n"); return 1; }
    size_t nread = fread(h_conv5_1_b, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read vgg16s_test_conv5_1_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv5_1_b, h_conv5_1_b, 2048, cudaMemcpyHostToDevice));
  float* h_conv5_2_w = (float*)malloc(9437184);
  if (!h_conv5_2_w) { fprintf(stderr, "malloc failed for conv5_2_w\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv5_2_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv5_2_w.bin\n"); return 1; }
    size_t nread = fread(h_conv5_2_w, sizeof(float), 2359296, f);
    fclose(f);
    if (nread != 2359296) { fprintf(stderr, "Failed to read vgg16s_test_conv5_2_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv5_2_w, h_conv5_2_w, 9437184, cudaMemcpyHostToDevice));
  float* h_conv5_2_b = (float*)malloc(2048);
  if (!h_conv5_2_b) { fprintf(stderr, "malloc failed for conv5_2_b\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv5_2_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv5_2_b.bin\n"); return 1; }
    size_t nread = fread(h_conv5_2_b, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read vgg16s_test_conv5_2_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv5_2_b, h_conv5_2_b, 2048, cudaMemcpyHostToDevice));
  float* h_conv5_3_w = (float*)malloc(9437184);
  if (!h_conv5_3_w) { fprintf(stderr, "malloc failed for conv5_3_w\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv5_3_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv5_3_w.bin\n"); return 1; }
    size_t nread = fread(h_conv5_3_w, sizeof(float), 2359296, f);
    fclose(f);
    if (nread != 2359296) { fprintf(stderr, "Failed to read vgg16s_test_conv5_3_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv5_3_w, h_conv5_3_w, 9437184, cudaMemcpyHostToDevice));
  float* h_conv5_3_b = (float*)malloc(2048);
  if (!h_conv5_3_b) { fprintf(stderr, "malloc failed for conv5_3_b\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_conv5_3_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_conv5_3_b.bin\n"); return 1; }
    size_t nread = fread(h_conv5_3_b, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read vgg16s_test_conv5_3_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv5_3_b, h_conv5_3_b, 2048, cudaMemcpyHostToDevice));
  float* h_fc1_w = (float*)malloc(8388608);
  if (!h_fc1_w) { fprintf(stderr, "malloc failed for fc1_w\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_fc1_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_fc1_w.bin\n"); return 1; }
    size_t nread = fread(h_fc1_w, sizeof(float), 2097152, f);
    fclose(f);
    if (nread != 2097152) { fprintf(stderr, "Failed to read vgg16s_test_fc1_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc1_w, h_fc1_w, 8388608, cudaMemcpyHostToDevice));
  float* h_fc1_b = (float*)malloc(16384);
  if (!h_fc1_b) { fprintf(stderr, "malloc failed for fc1_b\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_fc1_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_fc1_b.bin\n"); return 1; }
    size_t nread = fread(h_fc1_b, sizeof(float), 4096, f);
    fclose(f);
    if (nread != 4096) { fprintf(stderr, "Failed to read vgg16s_test_fc1_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc1_b, h_fc1_b, 16384, cudaMemcpyHostToDevice));
  float* h_fc2_w = (float*)malloc(67108864);
  if (!h_fc2_w) { fprintf(stderr, "malloc failed for fc2_w\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_fc2_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_fc2_w.bin\n"); return 1; }
    size_t nread = fread(h_fc2_w, sizeof(float), 16777216, f);
    fclose(f);
    if (nread != 16777216) { fprintf(stderr, "Failed to read vgg16s_test_fc2_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc2_w, h_fc2_w, 67108864, cudaMemcpyHostToDevice));
  float* h_fc2_b = (float*)malloc(16384);
  if (!h_fc2_b) { fprintf(stderr, "malloc failed for fc2_b\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_fc2_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_fc2_b.bin\n"); return 1; }
    size_t nread = fread(h_fc2_b, sizeof(float), 4096, f);
    fclose(f);
    if (nread != 4096) { fprintf(stderr, "Failed to read vgg16s_test_fc2_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc2_b, h_fc2_b, 16384, cudaMemcpyHostToDevice));
  float* h_fc3_w = (float*)malloc(16384000);
  if (!h_fc3_w) { fprintf(stderr, "malloc failed for fc3_w\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_fc3_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_fc3_w.bin\n"); return 1; }
    size_t nread = fread(h_fc3_w, sizeof(float), 4096000, f);
    fclose(f);
    if (nread != 4096000) { fprintf(stderr, "Failed to read vgg16s_test_fc3_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc3_w, h_fc3_w, 16384000, cudaMemcpyHostToDevice));
  float* h_fc3_b = (float*)malloc(4000);
  if (!h_fc3_b) { fprintf(stderr, "malloc failed for fc3_b\n"); return 1; }
  { FILE* f = fopen("vgg16s_test_fc3_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16s_test_fc3_b.bin\n"); return 1; }
    size_t nread = fread(h_fc3_b, sizeof(float), 1000, f);
    fclose(f);
    if (nread != 1000) { fprintf(stderr, "Failed to read vgg16s_test_fc3_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc3_b, h_fc3_b, 4000, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(__io_stop));
  CUDA_CHECK(cudaEventSynchronize(__io_stop));
  {
    float __io_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&__io_ms, __io_start, __io_stop));
    printf("MEMTIME_H2D vgg16s_test %.6f\n", __io_ms);
  }

  float* h_output = (float*)malloc(4000);
  if (!h_output) { fprintf(stderr, "malloc failed for output\n"); return 1; }

  cudaEvent_t __timer_start, __timer_stop;
  CUDA_CHECK(cudaEventCreate(&__timer_start));
  CUDA_CHECK(cudaEventCreate(&__timer_stop));
  CUDA_CHECK(cudaEventRecord(__timer_start));
  vgg16s_test();
  CUDA_CHECK(cudaEventRecord(__timer_stop));
  CUDA_CHECK(cudaEventSynchronize(__timer_stop));
  {
    float __elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&__elapsed_ms, __timer_start, __timer_stop));
    printf("TIME vgg16s_test %.6f\n", __elapsed_ms);
  }
  CUDA_CHECK(cudaEventDestroy(__timer_start));
  CUDA_CHECK(cudaEventDestroy(__timer_stop));

  CUDA_CHECK(cudaEventRecord(__io_start));
  CUDA_CHECK(cudaMemcpy(h_output, d_output, 4000, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(__io_stop));
  CUDA_CHECK(cudaEventSynchronize(__io_stop));
  {
    float __d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&__d2h_ms, __io_start, __io_stop));
    printf("MEMTIME_D2H vgg16s_test %.6f\n", __d2h_ms);
  }
  CUDA_CHECK(cudaEventDestroy(__io_start));
  CUDA_CHECK(cudaEventDestroy(__io_stop));

  FILE* fout = fopen("vgg16s_test_output.bin", "wb");
  if (!fout) { fprintf(stderr, "Cannot open %s\n", "vgg16s_test_output.bin"); return 1; }
  fwrite(h_output, sizeof(float), 1000, fout);
  fclose(fout);

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_conv1_1_w));
  CUDA_CHECK(cudaFree(d_conv1_1_b));
  CUDA_CHECK(cudaFree(d_conv1_2_w));
  CUDA_CHECK(cudaFree(d_conv1_2_b));
  CUDA_CHECK(cudaFree(d_conv2_1_w));
  CUDA_CHECK(cudaFree(d_conv2_1_b));
  CUDA_CHECK(cudaFree(d_conv2_2_w));
  CUDA_CHECK(cudaFree(d_conv2_2_b));
  CUDA_CHECK(cudaFree(d_conv3_1_w));
  CUDA_CHECK(cudaFree(d_conv3_1_b));
  CUDA_CHECK(cudaFree(d_conv3_2_w));
  CUDA_CHECK(cudaFree(d_conv3_2_b));
  CUDA_CHECK(cudaFree(d_conv3_3_w));
  CUDA_CHECK(cudaFree(d_conv3_3_b));
  CUDA_CHECK(cudaFree(d_conv4_1_w));
  CUDA_CHECK(cudaFree(d_conv4_1_b));
  CUDA_CHECK(cudaFree(d_conv4_2_w));
  CUDA_CHECK(cudaFree(d_conv4_2_b));
  CUDA_CHECK(cudaFree(d_conv4_3_w));
  CUDA_CHECK(cudaFree(d_conv4_3_b));
  CUDA_CHECK(cudaFree(d_conv5_1_w));
  CUDA_CHECK(cudaFree(d_conv5_1_b));
  CUDA_CHECK(cudaFree(d_conv5_2_w));
  CUDA_CHECK(cudaFree(d_conv5_2_b));
  CUDA_CHECK(cudaFree(d_conv5_3_w));
  CUDA_CHECK(cudaFree(d_conv5_3_b));
  CUDA_CHECK(cudaFree(d_fc1_w));
  CUDA_CHECK(cudaFree(d_fc1_b));
  CUDA_CHECK(cudaFree(d_fc2_w));
  CUDA_CHECK(cudaFree(d_fc2_b));
  CUDA_CHECK(cudaFree(d_fc3_w));
  CUDA_CHECK(cudaFree(d_fc3_b));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_relu1_1_out));
  CUDA_CHECK(cudaFree(d_relu1_2_out));
  CUDA_CHECK(cudaFree(d_pool1_out));
  CUDA_CHECK(cudaFree(d_relu2_1_out));
  CUDA_CHECK(cudaFree(d_relu2_2_out));
  CUDA_CHECK(cudaFree(d_pool2_out));
  CUDA_CHECK(cudaFree(d_relu3_1_out));
  CUDA_CHECK(cudaFree(d_relu3_2_out));
  CUDA_CHECK(cudaFree(d_relu3_3_out));
  CUDA_CHECK(cudaFree(d_pool3_out));
  CUDA_CHECK(cudaFree(d_relu4_1_out));
  CUDA_CHECK(cudaFree(d_relu4_2_out));
  CUDA_CHECK(cudaFree(d_relu4_3_out));
  CUDA_CHECK(cudaFree(d_pool4_out));
  CUDA_CHECK(cudaFree(d_relu5_1_out));
  CUDA_CHECK(cudaFree(d_relu5_2_out));
  CUDA_CHECK(cudaFree(d_relu5_3_out));
  CUDA_CHECK(cudaFree(d_pool5_out));
  CUDA_CHECK(cudaFree(d_flatten_out));
  CUDA_CHECK(cudaFree(d_fc1_relu_out));
  CUDA_CHECK(cudaFree(d_fc2_relu_out));
  free(h_input);
  free(h_conv1_1_w);
  free(h_conv1_1_b);
  free(h_conv1_2_w);
  free(h_conv1_2_b);
  free(h_conv2_1_w);
  free(h_conv2_1_b);
  free(h_conv2_2_w);
  free(h_conv2_2_b);
  free(h_conv3_1_w);
  free(h_conv3_1_b);
  free(h_conv3_2_w);
  free(h_conv3_2_b);
  free(h_conv3_3_w);
  free(h_conv3_3_b);
  free(h_conv4_1_w);
  free(h_conv4_1_b);
  free(h_conv4_2_w);
  free(h_conv4_2_b);
  free(h_conv4_3_w);
  free(h_conv4_3_b);
  free(h_conv5_1_w);
  free(h_conv5_1_b);
  free(h_conv5_2_w);
  free(h_conv5_2_b);
  free(h_conv5_3_w);
  free(h_conv5_3_b);
  free(h_fc1_w);
  free(h_fc1_b);
  free(h_fc2_w);
  free(h_fc2_b);
  free(h_fc3_w);
  free(h_fc3_b);
  free(h_output);
  return 0;
}
