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

__constant__ float (*input)[3][224][224];
__constant__ float (*conv1_w)[3][7][7];
__constant__ float* conv1_b;
__constant__ float* bn1_mean;
__constant__ float* bn1_var;
__constant__ float* bn1_gamma;
__constant__ float* bn1_beta;
__constant__ float (*l1_b1_w1)[64][3][3];
__constant__ float* l1_b1_b1;
__constant__ float (*l1_b1_w2)[64][3][3];
__constant__ float* l1_b1_b2;
__constant__ float* l1_b1_bn1_mean;
__constant__ float* l1_b1_bn1_var;
__constant__ float* l1_b1_bn1_gamma;
__constant__ float* l1_b1_bn1_beta;
__constant__ float* l1_b1_bn2_mean;
__constant__ float* l1_b1_bn2_var;
__constant__ float* l1_b1_bn2_gamma;
__constant__ float* l1_b1_bn2_beta;
__constant__ float (*l1_b2_w1)[64][3][3];
__constant__ float* l1_b2_b1;
__constant__ float (*l1_b2_w2)[64][3][3];
__constant__ float* l1_b2_b2;
__constant__ float* l1_b2_bn1_mean;
__constant__ float* l1_b2_bn1_var;
__constant__ float* l1_b2_bn1_gamma;
__constant__ float* l1_b2_bn1_beta;
__constant__ float* l1_b2_bn2_mean;
__constant__ float* l1_b2_bn2_var;
__constant__ float* l1_b2_bn2_gamma;
__constant__ float* l1_b2_bn2_beta;
__constant__ float (*l2_b1_w1)[64][3][3];
__constant__ float* l2_b1_b1;
__constant__ float (*l2_b1_w2)[128][3][3];
__constant__ float* l2_b1_b2;
__constant__ float* l2_b1_bn1_mean;
__constant__ float* l2_b1_bn1_var;
__constant__ float* l2_b1_bn1_gamma;
__constant__ float* l2_b1_bn1_beta;
__constant__ float* l2_b1_bn2_mean;
__constant__ float* l2_b1_bn2_var;
__constant__ float* l2_b1_bn2_gamma;
__constant__ float* l2_b1_bn2_beta;
__constant__ float (*l2_b1_shortcut_w)[64][1][1];
__constant__ float* l2_b1_shortcut_b;
__constant__ float* l2_b1_shortcut_mean;
__constant__ float* l2_b1_shortcut_var;
__constant__ float* l2_b1_shortcut_gamma;
__constant__ float* l2_b1_shortcut_beta;
__constant__ float (*l2_b2_w1)[128][3][3];
__constant__ float* l2_b2_b1;
__constant__ float (*l2_b2_w2)[128][3][3];
__constant__ float* l2_b2_b2;
__constant__ float* l2_b2_bn1_mean;
__constant__ float* l2_b2_bn1_var;
__constant__ float* l2_b2_bn1_gamma;
__constant__ float* l2_b2_bn1_beta;
__constant__ float* l2_b2_bn2_mean;
__constant__ float* l2_b2_bn2_var;
__constant__ float* l2_b2_bn2_gamma;
__constant__ float* l2_b2_bn2_beta;
__constant__ float (*l3_b1_w1)[128][3][3];
__constant__ float* l3_b1_b1;
__constant__ float (*l3_b1_w2)[256][3][3];
__constant__ float* l3_b1_b2;
__constant__ float* l3_b1_bn1_mean;
__constant__ float* l3_b1_bn1_var;
__constant__ float* l3_b1_bn1_gamma;
__constant__ float* l3_b1_bn1_beta;
__constant__ float* l3_b1_bn2_mean;
__constant__ float* l3_b1_bn2_var;
__constant__ float* l3_b1_bn2_gamma;
__constant__ float* l3_b1_bn2_beta;
__constant__ float (*l3_b1_shortcut_w)[128][1][1];
__constant__ float* l3_b1_shortcut_b;
__constant__ float* l3_b1_shortcut_mean;
__constant__ float* l3_b1_shortcut_var;
__constant__ float* l3_b1_shortcut_gamma;
__constant__ float* l3_b1_shortcut_beta;
__constant__ float (*l3_b2_w1)[256][3][3];
__constant__ float* l3_b2_b1;
__constant__ float (*l3_b2_w2)[256][3][3];
__constant__ float* l3_b2_b2;
__constant__ float* l3_b2_bn1_mean;
__constant__ float* l3_b2_bn1_var;
__constant__ float* l3_b2_bn1_gamma;
__constant__ float* l3_b2_bn1_beta;
__constant__ float* l3_b2_bn2_mean;
__constant__ float* l3_b2_bn2_var;
__constant__ float* l3_b2_bn2_gamma;
__constant__ float* l3_b2_bn2_beta;
__constant__ float (*l4_b1_w1)[256][3][3];
__constant__ float* l4_b1_b1;
__constant__ float (*l4_b1_w2)[512][3][3];
__constant__ float* l4_b1_b2;
__constant__ float* l4_b1_bn1_mean;
__constant__ float* l4_b1_bn1_var;
__constant__ float* l4_b1_bn1_gamma;
__constant__ float* l4_b1_bn1_beta;
__constant__ float* l4_b1_bn2_mean;
__constant__ float* l4_b1_bn2_var;
__constant__ float* l4_b1_bn2_gamma;
__constant__ float* l4_b1_bn2_beta;
__constant__ float (*l4_b1_shortcut_w)[256][1][1];
__constant__ float* l4_b1_shortcut_b;
__constant__ float* l4_b1_shortcut_mean;
__constant__ float* l4_b1_shortcut_var;
__constant__ float* l4_b1_shortcut_gamma;
__constant__ float* l4_b1_shortcut_beta;
__constant__ float (*l4_b2_w1)[512][3][3];
__constant__ float* l4_b2_b1;
__constant__ float (*l4_b2_w2)[512][3][3];
__constant__ float* l4_b2_b2;
__constant__ float* l4_b2_bn1_mean;
__constant__ float* l4_b2_bn1_var;
__constant__ float* l4_b2_bn1_gamma;
__constant__ float* l4_b2_bn1_beta;
__constant__ float* l4_b2_bn2_mean;
__constant__ float* l4_b2_bn2_var;
__constant__ float* l4_b2_bn2_gamma;
__constant__ float* l4_b2_bn2_beta;
__constant__ float (*fc_w)[512];
__constant__ float* fc_b;
__constant__ float (*output)[1000];
__constant__ float (*relu1_out)[64][112][112];
__constant__ float (*pool_out)[64][56][56];
__constant__ float (*l1_b1_relu1_out)[64][56][56];
__constant__ float (*l1_b1_add_out)[64][56][56];
__constant__ float (*l1_b2_relu1_out)[64][56][56];
__constant__ float (*l1_b2_add_out)[64][56][56];
__constant__ float (*l2_b1_shortcut_out)[128][28][28];
__constant__ float (*l2_b1_relu1_out)[128][28][28];
__constant__ float (*l2_b1_add_out)[128][28][28];
__constant__ float (*l2_b2_relu1_out)[128][28][28];
__constant__ float (*l2_b2_add_out)[128][28][28];
__constant__ float (*l3_b1_shortcut_out)[256][14][14];
__constant__ float (*l3_b1_relu1_out)[256][14][14];
__constant__ float (*l3_b1_add_out)[256][14][14];
__constant__ float (*l3_b2_relu1_out)[256][14][14];
__constant__ float (*l3_b2_add_out)[256][14][14];
__constant__ float (*l4_b1_shortcut_out)[512][7][7];
__constant__ float (*l4_b1_relu1_out)[512][7][7];
__constant__ float (*l4_b1_add_out)[512][7][7];
__constant__ float (*l4_b2_relu1_out)[512][7][7];
__constant__ float (*l4_b2_add_out)[512][7][7];
__constant__ float bn_eps;

#include <cuda_runtime.h>
#include <math.h>

__global__ void ast0_relu1_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 802816;
  if (idx >= N) return;
  int i3 = idx % 112; idx /= 112;
  int i2 = idx % 112; idx /= 112;
  int i1 = idx % 64; idx /= 64;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 3; c += 1)
    for (int k1 = 0; k1 < 7; k1 += 1)
      for (int k2 = 0; k2 < 7; k2 += 1) {
        int pi = 2 * i2 + k1 - 3;
        int pj = 2 * i3 + k2 - 3;
        float v = (pi < 0 || pi >= 224 || pj < 0 || pj >= 224)
                    ? 0.0f
                    : input[i0][c][pi][pj];
        acc += v * conv1_w[i1][c][k1][k2];
      }
  float result = acc + conv1_b[i1];
  float fused_result = (((result - bn1_mean[i1]) / sqrtf((bn1_var[i1] + bn_eps))) * bn1_gamma[i1] + bn1_beta[i1]);
  relu1_out[i0][i1][i2][i3] = (fused_result <= 0.0f) ? 0.0f : fused_result;
}

__global__ void ast1_pool_out()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 200704;
  if (idx >= N) return;
  int i3 = idx % 56; idx /= 56;
  int i2 = idx % 56; idx /= 56;
  int i1 = idx % 64; idx /= 64;
  int i0 = idx;
  float best = 0.0f;
  for (int k1 = 0; k1 < 3; k1 += 1)
    for (int k2 = 0; k2 < 3; k2 += 1) {
      int pi = 2 * i2 + k1 - 1;
      int pj = 2 * i3 + k2 - 1;
      float v = (pi < 0 || pi >= 112 || pj < 0 || pj >= 112)
                    ? 0.0f
                    : relu1_out[i0][i1][pi][pj];
      if (v > best) best = v;
    }
  pool_out[i0][i1][i2][i3] = best;
}

__global__ void ast2_l1_b1_relu1_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 200704;
  if (idx >= N) return;
  int i3 = idx % 56; idx /= 56;
  int i2 = idx % 56; idx /= 56;
  int i1 = idx % 64; idx /= 64;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 64; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = 1 * i2 + k1 - 1;
        int pj = 1 * i3 + k2 - 1;
        float v = (pi < 0 || pi >= 56 || pj < 0 || pj >= 56)
                    ? 0.0f
                    : pool_out[i0][c][pi][pj];
        acc += v * l1_b1_w1[i1][c][k1][k2];
      }
  float result = acc + l1_b1_b1[i1];
  float fused_result = (((result - l1_b1_bn1_mean[i1]) / sqrtf((l1_b1_bn1_var[i1] + bn_eps))) * l1_b1_bn1_gamma[i1] + l1_b1_bn1_beta[i1]);
  l1_b1_relu1_out[i0][i1][i2][i3] = (fused_result <= 0.0f) ? 0.0f : fused_result;
}

__global__ void ast3_l1_b1_add_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 200704;
  if (idx >= N) return;
  int i3 = idx % 56; idx /= 56;
  int i2 = idx % 56; idx /= 56;
  int i1 = idx % 64; idx /= 64;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 64; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = 1 * i2 + k1 - 1;
        int pj = 1 * i3 + k2 - 1;
        float v = (pi < 0 || pi >= 56 || pj < 0 || pj >= 56)
                    ? 0.0f
                    : l1_b1_relu1_out[i0][c][pi][pj];
        acc += v * l1_b1_w2[i1][c][k1][k2];
      }
  float result = acc + l1_b1_b2[i1];
  l1_b1_add_out[i0][i1][i2][i3] = ((((result - l1_b1_bn2_mean[i1]) / sqrtf((l1_b1_bn2_var[i1] + bn_eps))) * l1_b1_bn2_gamma[i1] + l1_b1_bn2_beta[i1]) + pool_out[i0][i1][i2][i3]);
}

__global__ void ast4_l1_b1_add_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 200704;
  if (idx >= N) return;
  int i3 = idx % 56; idx /= 56;
  int i2 = idx % 56; idx /= 56;
  int i1 = idx % 64; idx /= 64;
  int i0 = idx;
  l1_b1_add_out[i0][i1][i2][i3] = (l1_b1_add_out[i0][i1][i2][i3] <= 0.0f) ? 0.0f : l1_b1_add_out[i0][i1][i2][i3];
}

__global__ void ast5_l1_b2_relu1_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 200704;
  if (idx >= N) return;
  int i3 = idx % 56; idx /= 56;
  int i2 = idx % 56; idx /= 56;
  int i1 = idx % 64; idx /= 64;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 64; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = 1 * i2 + k1 - 1;
        int pj = 1 * i3 + k2 - 1;
        float v = (pi < 0 || pi >= 56 || pj < 0 || pj >= 56)
                    ? 0.0f
                    : l1_b1_add_out[i0][c][pi][pj];
        acc += v * l1_b2_w1[i1][c][k1][k2];
      }
  float result = acc + l1_b2_b1[i1];
  float fused_result = (((result - l1_b2_bn1_mean[i1]) / sqrtf((l1_b2_bn1_var[i1] + bn_eps))) * l1_b2_bn1_gamma[i1] + l1_b2_bn1_beta[i1]);
  l1_b2_relu1_out[i0][i1][i2][i3] = (fused_result <= 0.0f) ? 0.0f : fused_result;
}

__global__ void ast6_l1_b2_add_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 200704;
  if (idx >= N) return;
  int i3 = idx % 56; idx /= 56;
  int i2 = idx % 56; idx /= 56;
  int i1 = idx % 64; idx /= 64;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 64; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = 1 * i2 + k1 - 1;
        int pj = 1 * i3 + k2 - 1;
        float v = (pi < 0 || pi >= 56 || pj < 0 || pj >= 56)
                    ? 0.0f
                    : l1_b2_relu1_out[i0][c][pi][pj];
        acc += v * l1_b2_w2[i1][c][k1][k2];
      }
  float result = acc + l1_b2_b2[i1];
  l1_b2_add_out[i0][i1][i2][i3] = ((((result - l1_b2_bn2_mean[i1]) / sqrtf((l1_b2_bn2_var[i1] + bn_eps))) * l1_b2_bn2_gamma[i1] + l1_b2_bn2_beta[i1]) + l1_b1_add_out[i0][i1][i2][i3]);
}

__global__ void ast7_l1_b2_add_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 200704;
  if (idx >= N) return;
  int i3 = idx % 56; idx /= 56;
  int i2 = idx % 56; idx /= 56;
  int i1 = idx % 64; idx /= 64;
  int i0 = idx;
  l1_b2_add_out[i0][i1][i2][i3] = (l1_b2_add_out[i0][i1][i2][i3] <= 0.0f) ? 0.0f : l1_b2_add_out[i0][i1][i2][i3];
}

__global__ void ast8_l2_b1_shortcut_out()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 100352;
  if (idx >= N) return;
  int i3 = idx % 28; idx /= 28;
  int i2 = idx % 28; idx /= 28;
  int i1 = idx % 128; idx /= 128;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 64; c += 1)
    for (int k1 = 0; k1 < 1; k1 += 1)
      for (int k2 = 0; k2 < 1; k2 += 1) {
        int pi = 2 * i2 + k1 - 0;
        int pj = 2 * i3 + k2 - 0;
        float v = (pi < 0 || pi >= 56 || pj < 0 || pj >= 56)
                    ? 0.0f
                    : l1_b2_add_out[i0][c][pi][pj];
        acc += v * l2_b1_shortcut_w[i1][c][k1][k2];
      }
  float result = acc + l2_b1_shortcut_b[i1];
  l2_b1_shortcut_out[i0][i1][i2][i3] = result;
}

__global__ void ast9_l2_b1_shortcut_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 100352;
  if (idx >= N) return;
  int i3 = idx % 28; idx /= 28;
  int i2 = idx % 28; idx /= 28;
  int i1 = idx % 128; idx /= 128;
  int i0 = idx;
  l2_b1_shortcut_out[i0][i1][i2][i3] = (((l2_b1_shortcut_out[i0][i1][i2][i3] - l2_b1_shortcut_mean[i1]) / sqrtf((l2_b1_shortcut_var[i1] + bn_eps))) * l2_b1_shortcut_gamma[i1] + l2_b1_shortcut_beta[i1]);
}

__global__ void ast10_l2_b1_relu1_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 100352;
  if (idx >= N) return;
  int i3 = idx % 28; idx /= 28;
  int i2 = idx % 28; idx /= 28;
  int i1 = idx % 128; idx /= 128;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 64; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = 2 * i2 + k1 - 1;
        int pj = 2 * i3 + k2 - 1;
        float v = (pi < 0 || pi >= 56 || pj < 0 || pj >= 56)
                    ? 0.0f
                    : l1_b2_add_out[i0][c][pi][pj];
        acc += v * l2_b1_w1[i1][c][k1][k2];
      }
  float result = acc + l2_b1_b1[i1];
  float fused_result = (((result - l2_b1_bn1_mean[i1]) / sqrtf((l2_b1_bn1_var[i1] + bn_eps))) * l2_b1_bn1_gamma[i1] + l2_b1_bn1_beta[i1]);
  l2_b1_relu1_out[i0][i1][i2][i3] = (fused_result <= 0.0f) ? 0.0f : fused_result;
}

__global__ void ast11_l2_b1_add_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 100352;
  if (idx >= N) return;
  int i3 = idx % 28; idx /= 28;
  int i2 = idx % 28; idx /= 28;
  int i1 = idx % 128; idx /= 128;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 128; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = 1 * i2 + k1 - 1;
        int pj = 1 * i3 + k2 - 1;
        float v = (pi < 0 || pi >= 28 || pj < 0 || pj >= 28)
                    ? 0.0f
                    : l2_b1_relu1_out[i0][c][pi][pj];
        acc += v * l2_b1_w2[i1][c][k1][k2];
      }
  float result = acc + l2_b1_b2[i1];
  l2_b1_add_out[i0][i1][i2][i3] = ((((result - l2_b1_bn2_mean[i1]) / sqrtf((l2_b1_bn2_var[i1] + bn_eps))) * l2_b1_bn2_gamma[i1] + l2_b1_bn2_beta[i1]) + l2_b1_shortcut_out[i0][i1][i2][i3]);
}

__global__ void ast12_l2_b1_add_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 100352;
  if (idx >= N) return;
  int i3 = idx % 28; idx /= 28;
  int i2 = idx % 28; idx /= 28;
  int i1 = idx % 128; idx /= 128;
  int i0 = idx;
  l2_b1_add_out[i0][i1][i2][i3] = (l2_b1_add_out[i0][i1][i2][i3] <= 0.0f) ? 0.0f : l2_b1_add_out[i0][i1][i2][i3];
}

__global__ void ast13_l2_b2_relu1_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 100352;
  if (idx >= N) return;
  int i3 = idx % 28; idx /= 28;
  int i2 = idx % 28; idx /= 28;
  int i1 = idx % 128; idx /= 128;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 128; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = 1 * i2 + k1 - 1;
        int pj = 1 * i3 + k2 - 1;
        float v = (pi < 0 || pi >= 28 || pj < 0 || pj >= 28)
                    ? 0.0f
                    : l2_b1_add_out[i0][c][pi][pj];
        acc += v * l2_b2_w1[i1][c][k1][k2];
      }
  float result = acc + l2_b2_b1[i1];
  float fused_result = (((result - l2_b2_bn1_mean[i1]) / sqrtf((l2_b2_bn1_var[i1] + bn_eps))) * l2_b2_bn1_gamma[i1] + l2_b2_bn1_beta[i1]);
  l2_b2_relu1_out[i0][i1][i2][i3] = (fused_result <= 0.0f) ? 0.0f : fused_result;
}

__global__ void ast14_l2_b2_add_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 100352;
  if (idx >= N) return;
  int i3 = idx % 28; idx /= 28;
  int i2 = idx % 28; idx /= 28;
  int i1 = idx % 128; idx /= 128;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 128; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = 1 * i2 + k1 - 1;
        int pj = 1 * i3 + k2 - 1;
        float v = (pi < 0 || pi >= 28 || pj < 0 || pj >= 28)
                    ? 0.0f
                    : l2_b2_relu1_out[i0][c][pi][pj];
        acc += v * l2_b2_w2[i1][c][k1][k2];
      }
  float result = acc + l2_b2_b2[i1];
  l2_b2_add_out[i0][i1][i2][i3] = ((((result - l2_b2_bn2_mean[i1]) / sqrtf((l2_b2_bn2_var[i1] + bn_eps))) * l2_b2_bn2_gamma[i1] + l2_b2_bn2_beta[i1]) + l2_b1_add_out[i0][i1][i2][i3]);
}

__global__ void ast15_l2_b2_add_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 100352;
  if (idx >= N) return;
  int i3 = idx % 28; idx /= 28;
  int i2 = idx % 28; idx /= 28;
  int i1 = idx % 128; idx /= 128;
  int i0 = idx;
  l2_b2_add_out[i0][i1][i2][i3] = (l2_b2_add_out[i0][i1][i2][i3] <= 0.0f) ? 0.0f : l2_b2_add_out[i0][i1][i2][i3];
}

__global__ void ast16_l3_b1_shortcut_out()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 50176;
  if (idx >= N) return;
  int i3 = idx % 14; idx /= 14;
  int i2 = idx % 14; idx /= 14;
  int i1 = idx % 256; idx /= 256;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 128; c += 1)
    for (int k1 = 0; k1 < 1; k1 += 1)
      for (int k2 = 0; k2 < 1; k2 += 1) {
        int pi = 2 * i2 + k1 - 0;
        int pj = 2 * i3 + k2 - 0;
        float v = (pi < 0 || pi >= 28 || pj < 0 || pj >= 28)
                    ? 0.0f
                    : l2_b2_add_out[i0][c][pi][pj];
        acc += v * l3_b1_shortcut_w[i1][c][k1][k2];
      }
  float result = acc + l3_b1_shortcut_b[i1];
  l3_b1_shortcut_out[i0][i1][i2][i3] = result;
}

__global__ void ast17_l3_b1_shortcut_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 50176;
  if (idx >= N) return;
  int i3 = idx % 14; idx /= 14;
  int i2 = idx % 14; idx /= 14;
  int i1 = idx % 256; idx /= 256;
  int i0 = idx;
  l3_b1_shortcut_out[i0][i1][i2][i3] = (((l3_b1_shortcut_out[i0][i1][i2][i3] - l3_b1_shortcut_mean[i1]) / sqrtf((l3_b1_shortcut_var[i1] + bn_eps))) * l3_b1_shortcut_gamma[i1] + l3_b1_shortcut_beta[i1]);
}

__global__ void ast18_l3_b1_relu1_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 50176;
  if (idx >= N) return;
  int i3 = idx % 14; idx /= 14;
  int i2 = idx % 14; idx /= 14;
  int i1 = idx % 256; idx /= 256;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 128; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = 2 * i2 + k1 - 1;
        int pj = 2 * i3 + k2 - 1;
        float v = (pi < 0 || pi >= 28 || pj < 0 || pj >= 28)
                    ? 0.0f
                    : l2_b2_add_out[i0][c][pi][pj];
        acc += v * l3_b1_w1[i1][c][k1][k2];
      }
  float result = acc + l3_b1_b1[i1];
  float fused_result = (((result - l3_b1_bn1_mean[i1]) / sqrtf((l3_b1_bn1_var[i1] + bn_eps))) * l3_b1_bn1_gamma[i1] + l3_b1_bn1_beta[i1]);
  l3_b1_relu1_out[i0][i1][i2][i3] = (fused_result <= 0.0f) ? 0.0f : fused_result;
}

__global__ void ast19_l3_b1_add_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 50176;
  if (idx >= N) return;
  int i3 = idx % 14; idx /= 14;
  int i2 = idx % 14; idx /= 14;
  int i1 = idx % 256; idx /= 256;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 256; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = 1 * i2 + k1 - 1;
        int pj = 1 * i3 + k2 - 1;
        float v = (pi < 0 || pi >= 14 || pj < 0 || pj >= 14)
                    ? 0.0f
                    : l3_b1_relu1_out[i0][c][pi][pj];
        acc += v * l3_b1_w2[i1][c][k1][k2];
      }
  float result = acc + l3_b1_b2[i1];
  l3_b1_add_out[i0][i1][i2][i3] = ((((result - l3_b1_bn2_mean[i1]) / sqrtf((l3_b1_bn2_var[i1] + bn_eps))) * l3_b1_bn2_gamma[i1] + l3_b1_bn2_beta[i1]) + l3_b1_shortcut_out[i0][i1][i2][i3]);
}

__global__ void ast20_l3_b1_add_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 50176;
  if (idx >= N) return;
  int i3 = idx % 14; idx /= 14;
  int i2 = idx % 14; idx /= 14;
  int i1 = idx % 256; idx /= 256;
  int i0 = idx;
  l3_b1_add_out[i0][i1][i2][i3] = (l3_b1_add_out[i0][i1][i2][i3] <= 0.0f) ? 0.0f : l3_b1_add_out[i0][i1][i2][i3];
}

__global__ void ast21_l3_b2_relu1_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 50176;
  if (idx >= N) return;
  int i3 = idx % 14; idx /= 14;
  int i2 = idx % 14; idx /= 14;
  int i1 = idx % 256; idx /= 256;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 256; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = 1 * i2 + k1 - 1;
        int pj = 1 * i3 + k2 - 1;
        float v = (pi < 0 || pi >= 14 || pj < 0 || pj >= 14)
                    ? 0.0f
                    : l3_b1_add_out[i0][c][pi][pj];
        acc += v * l3_b2_w1[i1][c][k1][k2];
      }
  float result = acc + l3_b2_b1[i1];
  float fused_result = (((result - l3_b2_bn1_mean[i1]) / sqrtf((l3_b2_bn1_var[i1] + bn_eps))) * l3_b2_bn1_gamma[i1] + l3_b2_bn1_beta[i1]);
  l3_b2_relu1_out[i0][i1][i2][i3] = (fused_result <= 0.0f) ? 0.0f : fused_result;
}

__global__ void ast22_l3_b2_add_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 50176;
  if (idx >= N) return;
  int i3 = idx % 14; idx /= 14;
  int i2 = idx % 14; idx /= 14;
  int i1 = idx % 256; idx /= 256;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 256; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = 1 * i2 + k1 - 1;
        int pj = 1 * i3 + k2 - 1;
        float v = (pi < 0 || pi >= 14 || pj < 0 || pj >= 14)
                    ? 0.0f
                    : l3_b2_relu1_out[i0][c][pi][pj];
        acc += v * l3_b2_w2[i1][c][k1][k2];
      }
  float result = acc + l3_b2_b2[i1];
  l3_b2_add_out[i0][i1][i2][i3] = ((((result - l3_b2_bn2_mean[i1]) / sqrtf((l3_b2_bn2_var[i1] + bn_eps))) * l3_b2_bn2_gamma[i1] + l3_b2_bn2_beta[i1]) + l3_b1_add_out[i0][i1][i2][i3]);
}

__global__ void ast23_l3_b2_add_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 50176;
  if (idx >= N) return;
  int i3 = idx % 14; idx /= 14;
  int i2 = idx % 14; idx /= 14;
  int i1 = idx % 256; idx /= 256;
  int i0 = idx;
  l3_b2_add_out[i0][i1][i2][i3] = (l3_b2_add_out[i0][i1][i2][i3] <= 0.0f) ? 0.0f : l3_b2_add_out[i0][i1][i2][i3];
}

__global__ void ast24_l4_b1_shortcut_out()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 25088;
  if (idx >= N) return;
  int i3 = idx % 7; idx /= 7;
  int i2 = idx % 7; idx /= 7;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 256; c += 1)
    for (int k1 = 0; k1 < 1; k1 += 1)
      for (int k2 = 0; k2 < 1; k2 += 1) {
        int pi = 2 * i2 + k1 - 0;
        int pj = 2 * i3 + k2 - 0;
        float v = (pi < 0 || pi >= 14 || pj < 0 || pj >= 14)
                    ? 0.0f
                    : l3_b2_add_out[i0][c][pi][pj];
        acc += v * l4_b1_shortcut_w[i1][c][k1][k2];
      }
  float result = acc + l4_b1_shortcut_b[i1];
  l4_b1_shortcut_out[i0][i1][i2][i3] = result;
}

__global__ void ast25_l4_b1_shortcut_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 25088;
  if (idx >= N) return;
  int i3 = idx % 7; idx /= 7;
  int i2 = idx % 7; idx /= 7;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  l4_b1_shortcut_out[i0][i1][i2][i3] = (((l4_b1_shortcut_out[i0][i1][i2][i3] - l4_b1_shortcut_mean[i1]) / sqrtf((l4_b1_shortcut_var[i1] + bn_eps))) * l4_b1_shortcut_gamma[i1] + l4_b1_shortcut_beta[i1]);
}

__global__ void ast26_l4_b1_relu1_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 25088;
  if (idx >= N) return;
  int i3 = idx % 7; idx /= 7;
  int i2 = idx % 7; idx /= 7;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 256; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = 2 * i2 + k1 - 1;
        int pj = 2 * i3 + k2 - 1;
        float v = (pi < 0 || pi >= 14 || pj < 0 || pj >= 14)
                    ? 0.0f
                    : l3_b2_add_out[i0][c][pi][pj];
        acc += v * l4_b1_w1[i1][c][k1][k2];
      }
  float result = acc + l4_b1_b1[i1];
  float fused_result = (((result - l4_b1_bn1_mean[i1]) / sqrtf((l4_b1_bn1_var[i1] + bn_eps))) * l4_b1_bn1_gamma[i1] + l4_b1_bn1_beta[i1]);
  l4_b1_relu1_out[i0][i1][i2][i3] = (fused_result <= 0.0f) ? 0.0f : fused_result;
}

__global__ void ast27_l4_b1_add_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 25088;
  if (idx >= N) return;
  int i3 = idx % 7; idx /= 7;
  int i2 = idx % 7; idx /= 7;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 512; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = 1 * i2 + k1 - 1;
        int pj = 1 * i3 + k2 - 1;
        float v = (pi < 0 || pi >= 7 || pj < 0 || pj >= 7)
                    ? 0.0f
                    : l4_b1_relu1_out[i0][c][pi][pj];
        acc += v * l4_b1_w2[i1][c][k1][k2];
      }
  float result = acc + l4_b1_b2[i1];
  l4_b1_add_out[i0][i1][i2][i3] = ((((result - l4_b1_bn2_mean[i1]) / sqrtf((l4_b1_bn2_var[i1] + bn_eps))) * l4_b1_bn2_gamma[i1] + l4_b1_bn2_beta[i1]) + l4_b1_shortcut_out[i0][i1][i2][i3]);
}

__global__ void ast28_l4_b1_add_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 25088;
  if (idx >= N) return;
  int i3 = idx % 7; idx /= 7;
  int i2 = idx % 7; idx /= 7;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  l4_b1_add_out[i0][i1][i2][i3] = (l4_b1_add_out[i0][i1][i2][i3] <= 0.0f) ? 0.0f : l4_b1_add_out[i0][i1][i2][i3];
}

__global__ void ast29_l4_b2_relu1_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 25088;
  if (idx >= N) return;
  int i3 = idx % 7; idx /= 7;
  int i2 = idx % 7; idx /= 7;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 512; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = 1 * i2 + k1 - 1;
        int pj = 1 * i3 + k2 - 1;
        float v = (pi < 0 || pi >= 7 || pj < 0 || pj >= 7)
                    ? 0.0f
                    : l4_b1_add_out[i0][c][pi][pj];
        acc += v * l4_b2_w1[i1][c][k1][k2];
      }
  float result = acc + l4_b2_b1[i1];
  float fused_result = (((result - l4_b2_bn1_mean[i1]) / sqrtf((l4_b2_bn1_var[i1] + bn_eps))) * l4_b2_bn1_gamma[i1] + l4_b2_bn1_beta[i1]);
  l4_b2_relu1_out[i0][i1][i2][i3] = (fused_result <= 0.0f) ? 0.0f : fused_result;
}

__global__ void ast30_l4_b2_add_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 25088;
  if (idx >= N) return;
  int i3 = idx % 7; idx /= 7;
  int i2 = idx % 7; idx /= 7;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 512; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = 1 * i2 + k1 - 1;
        int pj = 1 * i3 + k2 - 1;
        float v = (pi < 0 || pi >= 7 || pj < 0 || pj >= 7)
                    ? 0.0f
                    : l4_b2_relu1_out[i0][c][pi][pj];
        acc += v * l4_b2_w2[i1][c][k1][k2];
      }
  float result = acc + l4_b2_b2[i1];
  l4_b2_add_out[i0][i1][i2][i3] = ((((result - l4_b2_bn2_mean[i1]) / sqrtf((l4_b2_bn2_var[i1] + bn_eps))) * l4_b2_bn2_gamma[i1] + l4_b2_bn2_beta[i1]) + l4_b1_add_out[i0][i1][i2][i3]);
}

__global__ void ast31_l4_b2_add_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 25088;
  if (idx >= N) return;
  int i3 = idx % 7; idx /= 7;
  int i2 = idx % 7; idx /= 7;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  l4_b2_add_out[i0][i1][i2][i3] = (l4_b2_add_out[i0][i1][i2][i3] <= 0.0f) ? 0.0f : l4_b2_add_out[i0][i1][i2][i3];
}

__global__ void ast32_output_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 1000;
  if (idx >= N) return;
  int i1 = idx % 1000; idx /= 1000;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 512; c += 1) {
    float sum = 0.0f;
    for (int ii = 0; ii < 7; ii += 1)
      for (int jj = 0; jj < 7; jj += 1)
        sum += l4_b2_add_out[i0][c][ii][jj];
    float avg = sum / 49.f;
    acc += avg * fc_w[i1][c];
  }
  output[i0][i1] = acc + fc_b[i1];
}

void resnet18_test()
{
  ast0_relu1_out_fused<<<3136, 256>>>();
  cudaDeviceSynchronize();
  ast1_pool_out<<<784, 256>>>();
  cudaDeviceSynchronize();
  ast2_l1_b1_relu1_out_fused<<<784, 256>>>();
  cudaDeviceSynchronize();
  ast3_l1_b1_add_out_fused<<<784, 256>>>();
  cudaDeviceSynchronize();
  ast4_l1_b1_add_out_fused<<<784, 256>>>();
  cudaDeviceSynchronize();
  ast5_l1_b2_relu1_out_fused<<<784, 256>>>();
  cudaDeviceSynchronize();
  ast6_l1_b2_add_out_fused<<<784, 256>>>();
  cudaDeviceSynchronize();
  ast7_l1_b2_add_out_fused<<<784, 256>>>();
  cudaDeviceSynchronize();
  ast8_l2_b1_shortcut_out<<<392, 256>>>();
  cudaDeviceSynchronize();
  ast9_l2_b1_shortcut_out_fused<<<392, 256>>>();
  cudaDeviceSynchronize();
  ast10_l2_b1_relu1_out_fused<<<392, 256>>>();
  cudaDeviceSynchronize();
  ast11_l2_b1_add_out_fused<<<392, 256>>>();
  cudaDeviceSynchronize();
  ast12_l2_b1_add_out_fused<<<392, 256>>>();
  cudaDeviceSynchronize();
  ast13_l2_b2_relu1_out_fused<<<392, 256>>>();
  cudaDeviceSynchronize();
  ast14_l2_b2_add_out_fused<<<392, 256>>>();
  cudaDeviceSynchronize();
  ast15_l2_b2_add_out_fused<<<392, 256>>>();
  cudaDeviceSynchronize();
  ast16_l3_b1_shortcut_out<<<196, 256>>>();
  cudaDeviceSynchronize();
  ast17_l3_b1_shortcut_out_fused<<<196, 256>>>();
  cudaDeviceSynchronize();
  ast18_l3_b1_relu1_out_fused<<<196, 256>>>();
  cudaDeviceSynchronize();
  ast19_l3_b1_add_out_fused<<<196, 256>>>();
  cudaDeviceSynchronize();
  ast20_l3_b1_add_out_fused<<<196, 256>>>();
  cudaDeviceSynchronize();
  ast21_l3_b2_relu1_out_fused<<<196, 256>>>();
  cudaDeviceSynchronize();
  ast22_l3_b2_add_out_fused<<<196, 256>>>();
  cudaDeviceSynchronize();
  ast23_l3_b2_add_out_fused<<<196, 256>>>();
  cudaDeviceSynchronize();
  ast24_l4_b1_shortcut_out<<<98, 256>>>();
  cudaDeviceSynchronize();
  ast25_l4_b1_shortcut_out_fused<<<98, 256>>>();
  cudaDeviceSynchronize();
  ast26_l4_b1_relu1_out_fused<<<98, 256>>>();
  cudaDeviceSynchronize();
  ast27_l4_b1_add_out_fused<<<98, 256>>>();
  cudaDeviceSynchronize();
  ast28_l4_b1_add_out_fused<<<98, 256>>>();
  cudaDeviceSynchronize();
  ast29_l4_b2_relu1_out_fused<<<98, 256>>>();
  cudaDeviceSynchronize();
  ast30_l4_b2_add_out_fused<<<98, 256>>>();
  cudaDeviceSynchronize();
  ast31_l4_b2_add_out_fused<<<98, 256>>>();
  cudaDeviceSynchronize();
  ast32_output_fused<<<4, 256>>>();
  cudaDeviceSynchronize();
}


int main(int argc, char** argv)
{
  size_t input_bytes = 602112;
  float* d_input = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_input, input_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(input, &d_input, sizeof(d_input)));
  size_t conv1_w_bytes = 37632;
  float* d_conv1_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv1_w, conv1_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv1_w, &d_conv1_w, sizeof(d_conv1_w)));
  size_t conv1_b_bytes = 256;
  float* d_conv1_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv1_b, conv1_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv1_b, &d_conv1_b, sizeof(d_conv1_b)));
  size_t bn1_mean_bytes = 256;
  float* d_bn1_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_bn1_mean, bn1_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(bn1_mean, &d_bn1_mean, sizeof(d_bn1_mean)));
  size_t bn1_var_bytes = 256;
  float* d_bn1_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_bn1_var, bn1_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(bn1_var, &d_bn1_var, sizeof(d_bn1_var)));
  size_t bn1_gamma_bytes = 256;
  float* d_bn1_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_bn1_gamma, bn1_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(bn1_gamma, &d_bn1_gamma, sizeof(d_bn1_gamma)));
  size_t bn1_beta_bytes = 256;
  float* d_bn1_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_bn1_beta, bn1_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(bn1_beta, &d_bn1_beta, sizeof(d_bn1_beta)));
  size_t l1_b1_w1_bytes = 147456;
  float* d_l1_b1_w1 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b1_w1, l1_b1_w1_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b1_w1, &d_l1_b1_w1, sizeof(d_l1_b1_w1)));
  size_t l1_b1_b1_bytes = 256;
  float* d_l1_b1_b1 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b1_b1, l1_b1_b1_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b1_b1, &d_l1_b1_b1, sizeof(d_l1_b1_b1)));
  size_t l1_b1_w2_bytes = 147456;
  float* d_l1_b1_w2 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b1_w2, l1_b1_w2_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b1_w2, &d_l1_b1_w2, sizeof(d_l1_b1_w2)));
  size_t l1_b1_b2_bytes = 256;
  float* d_l1_b1_b2 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b1_b2, l1_b1_b2_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b1_b2, &d_l1_b1_b2, sizeof(d_l1_b1_b2)));
  size_t l1_b1_bn1_mean_bytes = 256;
  float* d_l1_b1_bn1_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b1_bn1_mean, l1_b1_bn1_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b1_bn1_mean, &d_l1_b1_bn1_mean, sizeof(d_l1_b1_bn1_mean)));
  size_t l1_b1_bn1_var_bytes = 256;
  float* d_l1_b1_bn1_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b1_bn1_var, l1_b1_bn1_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b1_bn1_var, &d_l1_b1_bn1_var, sizeof(d_l1_b1_bn1_var)));
  size_t l1_b1_bn1_gamma_bytes = 256;
  float* d_l1_b1_bn1_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b1_bn1_gamma, l1_b1_bn1_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b1_bn1_gamma, &d_l1_b1_bn1_gamma, sizeof(d_l1_b1_bn1_gamma)));
  size_t l1_b1_bn1_beta_bytes = 256;
  float* d_l1_b1_bn1_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b1_bn1_beta, l1_b1_bn1_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b1_bn1_beta, &d_l1_b1_bn1_beta, sizeof(d_l1_b1_bn1_beta)));
  size_t l1_b1_bn2_mean_bytes = 256;
  float* d_l1_b1_bn2_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b1_bn2_mean, l1_b1_bn2_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b1_bn2_mean, &d_l1_b1_bn2_mean, sizeof(d_l1_b1_bn2_mean)));
  size_t l1_b1_bn2_var_bytes = 256;
  float* d_l1_b1_bn2_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b1_bn2_var, l1_b1_bn2_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b1_bn2_var, &d_l1_b1_bn2_var, sizeof(d_l1_b1_bn2_var)));
  size_t l1_b1_bn2_gamma_bytes = 256;
  float* d_l1_b1_bn2_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b1_bn2_gamma, l1_b1_bn2_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b1_bn2_gamma, &d_l1_b1_bn2_gamma, sizeof(d_l1_b1_bn2_gamma)));
  size_t l1_b1_bn2_beta_bytes = 256;
  float* d_l1_b1_bn2_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b1_bn2_beta, l1_b1_bn2_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b1_bn2_beta, &d_l1_b1_bn2_beta, sizeof(d_l1_b1_bn2_beta)));
  size_t l1_b2_w1_bytes = 147456;
  float* d_l1_b2_w1 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b2_w1, l1_b2_w1_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b2_w1, &d_l1_b2_w1, sizeof(d_l1_b2_w1)));
  size_t l1_b2_b1_bytes = 256;
  float* d_l1_b2_b1 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b2_b1, l1_b2_b1_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b2_b1, &d_l1_b2_b1, sizeof(d_l1_b2_b1)));
  size_t l1_b2_w2_bytes = 147456;
  float* d_l1_b2_w2 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b2_w2, l1_b2_w2_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b2_w2, &d_l1_b2_w2, sizeof(d_l1_b2_w2)));
  size_t l1_b2_b2_bytes = 256;
  float* d_l1_b2_b2 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b2_b2, l1_b2_b2_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b2_b2, &d_l1_b2_b2, sizeof(d_l1_b2_b2)));
  size_t l1_b2_bn1_mean_bytes = 256;
  float* d_l1_b2_bn1_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b2_bn1_mean, l1_b2_bn1_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b2_bn1_mean, &d_l1_b2_bn1_mean, sizeof(d_l1_b2_bn1_mean)));
  size_t l1_b2_bn1_var_bytes = 256;
  float* d_l1_b2_bn1_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b2_bn1_var, l1_b2_bn1_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b2_bn1_var, &d_l1_b2_bn1_var, sizeof(d_l1_b2_bn1_var)));
  size_t l1_b2_bn1_gamma_bytes = 256;
  float* d_l1_b2_bn1_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b2_bn1_gamma, l1_b2_bn1_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b2_bn1_gamma, &d_l1_b2_bn1_gamma, sizeof(d_l1_b2_bn1_gamma)));
  size_t l1_b2_bn1_beta_bytes = 256;
  float* d_l1_b2_bn1_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b2_bn1_beta, l1_b2_bn1_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b2_bn1_beta, &d_l1_b2_bn1_beta, sizeof(d_l1_b2_bn1_beta)));
  size_t l1_b2_bn2_mean_bytes = 256;
  float* d_l1_b2_bn2_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b2_bn2_mean, l1_b2_bn2_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b2_bn2_mean, &d_l1_b2_bn2_mean, sizeof(d_l1_b2_bn2_mean)));
  size_t l1_b2_bn2_var_bytes = 256;
  float* d_l1_b2_bn2_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b2_bn2_var, l1_b2_bn2_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b2_bn2_var, &d_l1_b2_bn2_var, sizeof(d_l1_b2_bn2_var)));
  size_t l1_b2_bn2_gamma_bytes = 256;
  float* d_l1_b2_bn2_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b2_bn2_gamma, l1_b2_bn2_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b2_bn2_gamma, &d_l1_b2_bn2_gamma, sizeof(d_l1_b2_bn2_gamma)));
  size_t l1_b2_bn2_beta_bytes = 256;
  float* d_l1_b2_bn2_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b2_bn2_beta, l1_b2_bn2_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b2_bn2_beta, &d_l1_b2_bn2_beta, sizeof(d_l1_b2_bn2_beta)));
  size_t l2_b1_w1_bytes = 294912;
  float* d_l2_b1_w1 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_w1, l2_b1_w1_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_w1, &d_l2_b1_w1, sizeof(d_l2_b1_w1)));
  size_t l2_b1_b1_bytes = 512;
  float* d_l2_b1_b1 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_b1, l2_b1_b1_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_b1, &d_l2_b1_b1, sizeof(d_l2_b1_b1)));
  size_t l2_b1_w2_bytes = 589824;
  float* d_l2_b1_w2 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_w2, l2_b1_w2_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_w2, &d_l2_b1_w2, sizeof(d_l2_b1_w2)));
  size_t l2_b1_b2_bytes = 512;
  float* d_l2_b1_b2 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_b2, l2_b1_b2_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_b2, &d_l2_b1_b2, sizeof(d_l2_b1_b2)));
  size_t l2_b1_bn1_mean_bytes = 512;
  float* d_l2_b1_bn1_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_bn1_mean, l2_b1_bn1_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_bn1_mean, &d_l2_b1_bn1_mean, sizeof(d_l2_b1_bn1_mean)));
  size_t l2_b1_bn1_var_bytes = 512;
  float* d_l2_b1_bn1_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_bn1_var, l2_b1_bn1_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_bn1_var, &d_l2_b1_bn1_var, sizeof(d_l2_b1_bn1_var)));
  size_t l2_b1_bn1_gamma_bytes = 512;
  float* d_l2_b1_bn1_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_bn1_gamma, l2_b1_bn1_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_bn1_gamma, &d_l2_b1_bn1_gamma, sizeof(d_l2_b1_bn1_gamma)));
  size_t l2_b1_bn1_beta_bytes = 512;
  float* d_l2_b1_bn1_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_bn1_beta, l2_b1_bn1_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_bn1_beta, &d_l2_b1_bn1_beta, sizeof(d_l2_b1_bn1_beta)));
  size_t l2_b1_bn2_mean_bytes = 512;
  float* d_l2_b1_bn2_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_bn2_mean, l2_b1_bn2_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_bn2_mean, &d_l2_b1_bn2_mean, sizeof(d_l2_b1_bn2_mean)));
  size_t l2_b1_bn2_var_bytes = 512;
  float* d_l2_b1_bn2_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_bn2_var, l2_b1_bn2_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_bn2_var, &d_l2_b1_bn2_var, sizeof(d_l2_b1_bn2_var)));
  size_t l2_b1_bn2_gamma_bytes = 512;
  float* d_l2_b1_bn2_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_bn2_gamma, l2_b1_bn2_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_bn2_gamma, &d_l2_b1_bn2_gamma, sizeof(d_l2_b1_bn2_gamma)));
  size_t l2_b1_bn2_beta_bytes = 512;
  float* d_l2_b1_bn2_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_bn2_beta, l2_b1_bn2_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_bn2_beta, &d_l2_b1_bn2_beta, sizeof(d_l2_b1_bn2_beta)));
  size_t l2_b1_shortcut_w_bytes = 32768;
  float* d_l2_b1_shortcut_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_shortcut_w, l2_b1_shortcut_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_shortcut_w, &d_l2_b1_shortcut_w, sizeof(d_l2_b1_shortcut_w)));
  size_t l2_b1_shortcut_b_bytes = 512;
  float* d_l2_b1_shortcut_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_shortcut_b, l2_b1_shortcut_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_shortcut_b, &d_l2_b1_shortcut_b, sizeof(d_l2_b1_shortcut_b)));
  size_t l2_b1_shortcut_mean_bytes = 512;
  float* d_l2_b1_shortcut_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_shortcut_mean, l2_b1_shortcut_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_shortcut_mean, &d_l2_b1_shortcut_mean, sizeof(d_l2_b1_shortcut_mean)));
  size_t l2_b1_shortcut_var_bytes = 512;
  float* d_l2_b1_shortcut_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_shortcut_var, l2_b1_shortcut_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_shortcut_var, &d_l2_b1_shortcut_var, sizeof(d_l2_b1_shortcut_var)));
  size_t l2_b1_shortcut_gamma_bytes = 512;
  float* d_l2_b1_shortcut_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_shortcut_gamma, l2_b1_shortcut_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_shortcut_gamma, &d_l2_b1_shortcut_gamma, sizeof(d_l2_b1_shortcut_gamma)));
  size_t l2_b1_shortcut_beta_bytes = 512;
  float* d_l2_b1_shortcut_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_shortcut_beta, l2_b1_shortcut_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_shortcut_beta, &d_l2_b1_shortcut_beta, sizeof(d_l2_b1_shortcut_beta)));
  size_t l2_b2_w1_bytes = 589824;
  float* d_l2_b2_w1 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b2_w1, l2_b2_w1_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b2_w1, &d_l2_b2_w1, sizeof(d_l2_b2_w1)));
  size_t l2_b2_b1_bytes = 512;
  float* d_l2_b2_b1 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b2_b1, l2_b2_b1_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b2_b1, &d_l2_b2_b1, sizeof(d_l2_b2_b1)));
  size_t l2_b2_w2_bytes = 589824;
  float* d_l2_b2_w2 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b2_w2, l2_b2_w2_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b2_w2, &d_l2_b2_w2, sizeof(d_l2_b2_w2)));
  size_t l2_b2_b2_bytes = 512;
  float* d_l2_b2_b2 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b2_b2, l2_b2_b2_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b2_b2, &d_l2_b2_b2, sizeof(d_l2_b2_b2)));
  size_t l2_b2_bn1_mean_bytes = 512;
  float* d_l2_b2_bn1_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b2_bn1_mean, l2_b2_bn1_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b2_bn1_mean, &d_l2_b2_bn1_mean, sizeof(d_l2_b2_bn1_mean)));
  size_t l2_b2_bn1_var_bytes = 512;
  float* d_l2_b2_bn1_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b2_bn1_var, l2_b2_bn1_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b2_bn1_var, &d_l2_b2_bn1_var, sizeof(d_l2_b2_bn1_var)));
  size_t l2_b2_bn1_gamma_bytes = 512;
  float* d_l2_b2_bn1_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b2_bn1_gamma, l2_b2_bn1_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b2_bn1_gamma, &d_l2_b2_bn1_gamma, sizeof(d_l2_b2_bn1_gamma)));
  size_t l2_b2_bn1_beta_bytes = 512;
  float* d_l2_b2_bn1_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b2_bn1_beta, l2_b2_bn1_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b2_bn1_beta, &d_l2_b2_bn1_beta, sizeof(d_l2_b2_bn1_beta)));
  size_t l2_b2_bn2_mean_bytes = 512;
  float* d_l2_b2_bn2_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b2_bn2_mean, l2_b2_bn2_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b2_bn2_mean, &d_l2_b2_bn2_mean, sizeof(d_l2_b2_bn2_mean)));
  size_t l2_b2_bn2_var_bytes = 512;
  float* d_l2_b2_bn2_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b2_bn2_var, l2_b2_bn2_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b2_bn2_var, &d_l2_b2_bn2_var, sizeof(d_l2_b2_bn2_var)));
  size_t l2_b2_bn2_gamma_bytes = 512;
  float* d_l2_b2_bn2_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b2_bn2_gamma, l2_b2_bn2_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b2_bn2_gamma, &d_l2_b2_bn2_gamma, sizeof(d_l2_b2_bn2_gamma)));
  size_t l2_b2_bn2_beta_bytes = 512;
  float* d_l2_b2_bn2_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b2_bn2_beta, l2_b2_bn2_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b2_bn2_beta, &d_l2_b2_bn2_beta, sizeof(d_l2_b2_bn2_beta)));
  size_t l3_b1_w1_bytes = 1179648;
  float* d_l3_b1_w1 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_w1, l3_b1_w1_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_w1, &d_l3_b1_w1, sizeof(d_l3_b1_w1)));
  size_t l3_b1_b1_bytes = 1024;
  float* d_l3_b1_b1 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_b1, l3_b1_b1_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_b1, &d_l3_b1_b1, sizeof(d_l3_b1_b1)));
  size_t l3_b1_w2_bytes = 2359296;
  float* d_l3_b1_w2 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_w2, l3_b1_w2_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_w2, &d_l3_b1_w2, sizeof(d_l3_b1_w2)));
  size_t l3_b1_b2_bytes = 1024;
  float* d_l3_b1_b2 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_b2, l3_b1_b2_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_b2, &d_l3_b1_b2, sizeof(d_l3_b1_b2)));
  size_t l3_b1_bn1_mean_bytes = 1024;
  float* d_l3_b1_bn1_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_bn1_mean, l3_b1_bn1_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_bn1_mean, &d_l3_b1_bn1_mean, sizeof(d_l3_b1_bn1_mean)));
  size_t l3_b1_bn1_var_bytes = 1024;
  float* d_l3_b1_bn1_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_bn1_var, l3_b1_bn1_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_bn1_var, &d_l3_b1_bn1_var, sizeof(d_l3_b1_bn1_var)));
  size_t l3_b1_bn1_gamma_bytes = 1024;
  float* d_l3_b1_bn1_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_bn1_gamma, l3_b1_bn1_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_bn1_gamma, &d_l3_b1_bn1_gamma, sizeof(d_l3_b1_bn1_gamma)));
  size_t l3_b1_bn1_beta_bytes = 1024;
  float* d_l3_b1_bn1_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_bn1_beta, l3_b1_bn1_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_bn1_beta, &d_l3_b1_bn1_beta, sizeof(d_l3_b1_bn1_beta)));
  size_t l3_b1_bn2_mean_bytes = 1024;
  float* d_l3_b1_bn2_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_bn2_mean, l3_b1_bn2_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_bn2_mean, &d_l3_b1_bn2_mean, sizeof(d_l3_b1_bn2_mean)));
  size_t l3_b1_bn2_var_bytes = 1024;
  float* d_l3_b1_bn2_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_bn2_var, l3_b1_bn2_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_bn2_var, &d_l3_b1_bn2_var, sizeof(d_l3_b1_bn2_var)));
  size_t l3_b1_bn2_gamma_bytes = 1024;
  float* d_l3_b1_bn2_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_bn2_gamma, l3_b1_bn2_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_bn2_gamma, &d_l3_b1_bn2_gamma, sizeof(d_l3_b1_bn2_gamma)));
  size_t l3_b1_bn2_beta_bytes = 1024;
  float* d_l3_b1_bn2_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_bn2_beta, l3_b1_bn2_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_bn2_beta, &d_l3_b1_bn2_beta, sizeof(d_l3_b1_bn2_beta)));
  size_t l3_b1_shortcut_w_bytes = 131072;
  float* d_l3_b1_shortcut_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_shortcut_w, l3_b1_shortcut_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_shortcut_w, &d_l3_b1_shortcut_w, sizeof(d_l3_b1_shortcut_w)));
  size_t l3_b1_shortcut_b_bytes = 1024;
  float* d_l3_b1_shortcut_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_shortcut_b, l3_b1_shortcut_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_shortcut_b, &d_l3_b1_shortcut_b, sizeof(d_l3_b1_shortcut_b)));
  size_t l3_b1_shortcut_mean_bytes = 1024;
  float* d_l3_b1_shortcut_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_shortcut_mean, l3_b1_shortcut_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_shortcut_mean, &d_l3_b1_shortcut_mean, sizeof(d_l3_b1_shortcut_mean)));
  size_t l3_b1_shortcut_var_bytes = 1024;
  float* d_l3_b1_shortcut_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_shortcut_var, l3_b1_shortcut_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_shortcut_var, &d_l3_b1_shortcut_var, sizeof(d_l3_b1_shortcut_var)));
  size_t l3_b1_shortcut_gamma_bytes = 1024;
  float* d_l3_b1_shortcut_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_shortcut_gamma, l3_b1_shortcut_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_shortcut_gamma, &d_l3_b1_shortcut_gamma, sizeof(d_l3_b1_shortcut_gamma)));
  size_t l3_b1_shortcut_beta_bytes = 1024;
  float* d_l3_b1_shortcut_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_shortcut_beta, l3_b1_shortcut_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_shortcut_beta, &d_l3_b1_shortcut_beta, sizeof(d_l3_b1_shortcut_beta)));
  size_t l3_b2_w1_bytes = 2359296;
  float* d_l3_b2_w1 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b2_w1, l3_b2_w1_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b2_w1, &d_l3_b2_w1, sizeof(d_l3_b2_w1)));
  size_t l3_b2_b1_bytes = 1024;
  float* d_l3_b2_b1 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b2_b1, l3_b2_b1_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b2_b1, &d_l3_b2_b1, sizeof(d_l3_b2_b1)));
  size_t l3_b2_w2_bytes = 2359296;
  float* d_l3_b2_w2 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b2_w2, l3_b2_w2_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b2_w2, &d_l3_b2_w2, sizeof(d_l3_b2_w2)));
  size_t l3_b2_b2_bytes = 1024;
  float* d_l3_b2_b2 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b2_b2, l3_b2_b2_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b2_b2, &d_l3_b2_b2, sizeof(d_l3_b2_b2)));
  size_t l3_b2_bn1_mean_bytes = 1024;
  float* d_l3_b2_bn1_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b2_bn1_mean, l3_b2_bn1_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b2_bn1_mean, &d_l3_b2_bn1_mean, sizeof(d_l3_b2_bn1_mean)));
  size_t l3_b2_bn1_var_bytes = 1024;
  float* d_l3_b2_bn1_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b2_bn1_var, l3_b2_bn1_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b2_bn1_var, &d_l3_b2_bn1_var, sizeof(d_l3_b2_bn1_var)));
  size_t l3_b2_bn1_gamma_bytes = 1024;
  float* d_l3_b2_bn1_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b2_bn1_gamma, l3_b2_bn1_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b2_bn1_gamma, &d_l3_b2_bn1_gamma, sizeof(d_l3_b2_bn1_gamma)));
  size_t l3_b2_bn1_beta_bytes = 1024;
  float* d_l3_b2_bn1_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b2_bn1_beta, l3_b2_bn1_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b2_bn1_beta, &d_l3_b2_bn1_beta, sizeof(d_l3_b2_bn1_beta)));
  size_t l3_b2_bn2_mean_bytes = 1024;
  float* d_l3_b2_bn2_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b2_bn2_mean, l3_b2_bn2_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b2_bn2_mean, &d_l3_b2_bn2_mean, sizeof(d_l3_b2_bn2_mean)));
  size_t l3_b2_bn2_var_bytes = 1024;
  float* d_l3_b2_bn2_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b2_bn2_var, l3_b2_bn2_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b2_bn2_var, &d_l3_b2_bn2_var, sizeof(d_l3_b2_bn2_var)));
  size_t l3_b2_bn2_gamma_bytes = 1024;
  float* d_l3_b2_bn2_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b2_bn2_gamma, l3_b2_bn2_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b2_bn2_gamma, &d_l3_b2_bn2_gamma, sizeof(d_l3_b2_bn2_gamma)));
  size_t l3_b2_bn2_beta_bytes = 1024;
  float* d_l3_b2_bn2_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b2_bn2_beta, l3_b2_bn2_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b2_bn2_beta, &d_l3_b2_bn2_beta, sizeof(d_l3_b2_bn2_beta)));
  size_t l4_b1_w1_bytes = 4718592;
  float* d_l4_b1_w1 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_w1, l4_b1_w1_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_w1, &d_l4_b1_w1, sizeof(d_l4_b1_w1)));
  size_t l4_b1_b1_bytes = 2048;
  float* d_l4_b1_b1 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_b1, l4_b1_b1_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_b1, &d_l4_b1_b1, sizeof(d_l4_b1_b1)));
  size_t l4_b1_w2_bytes = 9437184;
  float* d_l4_b1_w2 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_w2, l4_b1_w2_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_w2, &d_l4_b1_w2, sizeof(d_l4_b1_w2)));
  size_t l4_b1_b2_bytes = 2048;
  float* d_l4_b1_b2 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_b2, l4_b1_b2_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_b2, &d_l4_b1_b2, sizeof(d_l4_b1_b2)));
  size_t l4_b1_bn1_mean_bytes = 2048;
  float* d_l4_b1_bn1_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_bn1_mean, l4_b1_bn1_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_bn1_mean, &d_l4_b1_bn1_mean, sizeof(d_l4_b1_bn1_mean)));
  size_t l4_b1_bn1_var_bytes = 2048;
  float* d_l4_b1_bn1_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_bn1_var, l4_b1_bn1_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_bn1_var, &d_l4_b1_bn1_var, sizeof(d_l4_b1_bn1_var)));
  size_t l4_b1_bn1_gamma_bytes = 2048;
  float* d_l4_b1_bn1_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_bn1_gamma, l4_b1_bn1_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_bn1_gamma, &d_l4_b1_bn1_gamma, sizeof(d_l4_b1_bn1_gamma)));
  size_t l4_b1_bn1_beta_bytes = 2048;
  float* d_l4_b1_bn1_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_bn1_beta, l4_b1_bn1_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_bn1_beta, &d_l4_b1_bn1_beta, sizeof(d_l4_b1_bn1_beta)));
  size_t l4_b1_bn2_mean_bytes = 2048;
  float* d_l4_b1_bn2_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_bn2_mean, l4_b1_bn2_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_bn2_mean, &d_l4_b1_bn2_mean, sizeof(d_l4_b1_bn2_mean)));
  size_t l4_b1_bn2_var_bytes = 2048;
  float* d_l4_b1_bn2_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_bn2_var, l4_b1_bn2_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_bn2_var, &d_l4_b1_bn2_var, sizeof(d_l4_b1_bn2_var)));
  size_t l4_b1_bn2_gamma_bytes = 2048;
  float* d_l4_b1_bn2_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_bn2_gamma, l4_b1_bn2_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_bn2_gamma, &d_l4_b1_bn2_gamma, sizeof(d_l4_b1_bn2_gamma)));
  size_t l4_b1_bn2_beta_bytes = 2048;
  float* d_l4_b1_bn2_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_bn2_beta, l4_b1_bn2_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_bn2_beta, &d_l4_b1_bn2_beta, sizeof(d_l4_b1_bn2_beta)));
  size_t l4_b1_shortcut_w_bytes = 524288;
  float* d_l4_b1_shortcut_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_shortcut_w, l4_b1_shortcut_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_shortcut_w, &d_l4_b1_shortcut_w, sizeof(d_l4_b1_shortcut_w)));
  size_t l4_b1_shortcut_b_bytes = 2048;
  float* d_l4_b1_shortcut_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_shortcut_b, l4_b1_shortcut_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_shortcut_b, &d_l4_b1_shortcut_b, sizeof(d_l4_b1_shortcut_b)));
  size_t l4_b1_shortcut_mean_bytes = 2048;
  float* d_l4_b1_shortcut_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_shortcut_mean, l4_b1_shortcut_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_shortcut_mean, &d_l4_b1_shortcut_mean, sizeof(d_l4_b1_shortcut_mean)));
  size_t l4_b1_shortcut_var_bytes = 2048;
  float* d_l4_b1_shortcut_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_shortcut_var, l4_b1_shortcut_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_shortcut_var, &d_l4_b1_shortcut_var, sizeof(d_l4_b1_shortcut_var)));
  size_t l4_b1_shortcut_gamma_bytes = 2048;
  float* d_l4_b1_shortcut_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_shortcut_gamma, l4_b1_shortcut_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_shortcut_gamma, &d_l4_b1_shortcut_gamma, sizeof(d_l4_b1_shortcut_gamma)));
  size_t l4_b1_shortcut_beta_bytes = 2048;
  float* d_l4_b1_shortcut_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_shortcut_beta, l4_b1_shortcut_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_shortcut_beta, &d_l4_b1_shortcut_beta, sizeof(d_l4_b1_shortcut_beta)));
  size_t l4_b2_w1_bytes = 9437184;
  float* d_l4_b2_w1 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b2_w1, l4_b2_w1_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b2_w1, &d_l4_b2_w1, sizeof(d_l4_b2_w1)));
  size_t l4_b2_b1_bytes = 2048;
  float* d_l4_b2_b1 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b2_b1, l4_b2_b1_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b2_b1, &d_l4_b2_b1, sizeof(d_l4_b2_b1)));
  size_t l4_b2_w2_bytes = 9437184;
  float* d_l4_b2_w2 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b2_w2, l4_b2_w2_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b2_w2, &d_l4_b2_w2, sizeof(d_l4_b2_w2)));
  size_t l4_b2_b2_bytes = 2048;
  float* d_l4_b2_b2 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b2_b2, l4_b2_b2_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b2_b2, &d_l4_b2_b2, sizeof(d_l4_b2_b2)));
  size_t l4_b2_bn1_mean_bytes = 2048;
  float* d_l4_b2_bn1_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b2_bn1_mean, l4_b2_bn1_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b2_bn1_mean, &d_l4_b2_bn1_mean, sizeof(d_l4_b2_bn1_mean)));
  size_t l4_b2_bn1_var_bytes = 2048;
  float* d_l4_b2_bn1_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b2_bn1_var, l4_b2_bn1_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b2_bn1_var, &d_l4_b2_bn1_var, sizeof(d_l4_b2_bn1_var)));
  size_t l4_b2_bn1_gamma_bytes = 2048;
  float* d_l4_b2_bn1_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b2_bn1_gamma, l4_b2_bn1_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b2_bn1_gamma, &d_l4_b2_bn1_gamma, sizeof(d_l4_b2_bn1_gamma)));
  size_t l4_b2_bn1_beta_bytes = 2048;
  float* d_l4_b2_bn1_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b2_bn1_beta, l4_b2_bn1_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b2_bn1_beta, &d_l4_b2_bn1_beta, sizeof(d_l4_b2_bn1_beta)));
  size_t l4_b2_bn2_mean_bytes = 2048;
  float* d_l4_b2_bn2_mean = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b2_bn2_mean, l4_b2_bn2_mean_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b2_bn2_mean, &d_l4_b2_bn2_mean, sizeof(d_l4_b2_bn2_mean)));
  size_t l4_b2_bn2_var_bytes = 2048;
  float* d_l4_b2_bn2_var = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b2_bn2_var, l4_b2_bn2_var_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b2_bn2_var, &d_l4_b2_bn2_var, sizeof(d_l4_b2_bn2_var)));
  size_t l4_b2_bn2_gamma_bytes = 2048;
  float* d_l4_b2_bn2_gamma = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b2_bn2_gamma, l4_b2_bn2_gamma_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b2_bn2_gamma, &d_l4_b2_bn2_gamma, sizeof(d_l4_b2_bn2_gamma)));
  size_t l4_b2_bn2_beta_bytes = 2048;
  float* d_l4_b2_bn2_beta = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b2_bn2_beta, l4_b2_bn2_beta_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b2_bn2_beta, &d_l4_b2_bn2_beta, sizeof(d_l4_b2_bn2_beta)));
  size_t fc_w_bytes = 2048000;
  float* d_fc_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_fc_w, fc_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(fc_w, &d_fc_w, sizeof(d_fc_w)));
  size_t fc_b_bytes = 4000;
  float* d_fc_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_fc_b, fc_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(fc_b, &d_fc_b, sizeof(d_fc_b)));
  size_t output_bytes = 4000;
  float* d_output = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_output, output_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(output, &d_output, sizeof(d_output)));
  size_t relu1_out_bytes = 3211264;
  float* d_relu1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu1_out, relu1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu1_out, &d_relu1_out, sizeof(d_relu1_out)));
  size_t pool_out_bytes = 802816;
  float* d_pool_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_pool_out, pool_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(pool_out, &d_pool_out, sizeof(d_pool_out)));
  size_t l1_b1_relu1_out_bytes = 802816;
  float* d_l1_b1_relu1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b1_relu1_out, l1_b1_relu1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b1_relu1_out, &d_l1_b1_relu1_out, sizeof(d_l1_b1_relu1_out)));
  size_t l1_b1_add_out_bytes = 802816;
  float* d_l1_b1_add_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b1_add_out, l1_b1_add_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b1_add_out, &d_l1_b1_add_out, sizeof(d_l1_b1_add_out)));
  size_t l1_b2_relu1_out_bytes = 802816;
  float* d_l1_b2_relu1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b2_relu1_out, l1_b2_relu1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b2_relu1_out, &d_l1_b2_relu1_out, sizeof(d_l1_b2_relu1_out)));
  size_t l1_b2_add_out_bytes = 802816;
  float* d_l1_b2_add_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b2_add_out, l1_b2_add_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b2_add_out, &d_l1_b2_add_out, sizeof(d_l1_b2_add_out)));
  size_t l2_b1_shortcut_out_bytes = 401408;
  float* d_l2_b1_shortcut_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_shortcut_out, l2_b1_shortcut_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_shortcut_out, &d_l2_b1_shortcut_out, sizeof(d_l2_b1_shortcut_out)));
  size_t l2_b1_relu1_out_bytes = 401408;
  float* d_l2_b1_relu1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_relu1_out, l2_b1_relu1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_relu1_out, &d_l2_b1_relu1_out, sizeof(d_l2_b1_relu1_out)));
  size_t l2_b1_add_out_bytes = 401408;
  float* d_l2_b1_add_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_add_out, l2_b1_add_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_add_out, &d_l2_b1_add_out, sizeof(d_l2_b1_add_out)));
  size_t l2_b2_relu1_out_bytes = 401408;
  float* d_l2_b2_relu1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b2_relu1_out, l2_b2_relu1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b2_relu1_out, &d_l2_b2_relu1_out, sizeof(d_l2_b2_relu1_out)));
  size_t l2_b2_add_out_bytes = 401408;
  float* d_l2_b2_add_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b2_add_out, l2_b2_add_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b2_add_out, &d_l2_b2_add_out, sizeof(d_l2_b2_add_out)));
  size_t l3_b1_shortcut_out_bytes = 200704;
  float* d_l3_b1_shortcut_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_shortcut_out, l3_b1_shortcut_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_shortcut_out, &d_l3_b1_shortcut_out, sizeof(d_l3_b1_shortcut_out)));
  size_t l3_b1_relu1_out_bytes = 200704;
  float* d_l3_b1_relu1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_relu1_out, l3_b1_relu1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_relu1_out, &d_l3_b1_relu1_out, sizeof(d_l3_b1_relu1_out)));
  size_t l3_b1_add_out_bytes = 200704;
  float* d_l3_b1_add_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_add_out, l3_b1_add_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_add_out, &d_l3_b1_add_out, sizeof(d_l3_b1_add_out)));
  size_t l3_b2_relu1_out_bytes = 200704;
  float* d_l3_b2_relu1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b2_relu1_out, l3_b2_relu1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b2_relu1_out, &d_l3_b2_relu1_out, sizeof(d_l3_b2_relu1_out)));
  size_t l3_b2_add_out_bytes = 200704;
  float* d_l3_b2_add_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b2_add_out, l3_b2_add_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b2_add_out, &d_l3_b2_add_out, sizeof(d_l3_b2_add_out)));
  size_t l4_b1_shortcut_out_bytes = 100352;
  float* d_l4_b1_shortcut_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_shortcut_out, l4_b1_shortcut_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_shortcut_out, &d_l4_b1_shortcut_out, sizeof(d_l4_b1_shortcut_out)));
  size_t l4_b1_relu1_out_bytes = 100352;
  float* d_l4_b1_relu1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_relu1_out, l4_b1_relu1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_relu1_out, &d_l4_b1_relu1_out, sizeof(d_l4_b1_relu1_out)));
  size_t l4_b1_add_out_bytes = 100352;
  float* d_l4_b1_add_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_add_out, l4_b1_add_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_add_out, &d_l4_b1_add_out, sizeof(d_l4_b1_add_out)));
  size_t l4_b2_relu1_out_bytes = 100352;
  float* d_l4_b2_relu1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b2_relu1_out, l4_b2_relu1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b2_relu1_out, &d_l4_b2_relu1_out, sizeof(d_l4_b2_relu1_out)));
  size_t l4_b2_add_out_bytes = 100352;
  float* d_l4_b2_add_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b2_add_out, l4_b2_add_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b2_add_out, &d_l4_b2_add_out, sizeof(d_l4_b2_add_out)));

  cudaEvent_t __io_start, __io_stop;
  CUDA_CHECK(cudaEventCreate(&__io_start));
  CUDA_CHECK(cudaEventCreate(&__io_stop));
  CUDA_CHECK(cudaEventRecord(__io_start));
  float* h_input = (float*)malloc(602112);
  if (!h_input) { fprintf(stderr, "malloc failed for input\n"); return 1; }
  { FILE* f = fopen("resnet18_test_input.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_input.bin\n"); return 1; }
    size_t nread = fread(h_input, sizeof(float), 150528, f);
    fclose(f);
    if (nread != 150528) { fprintf(stderr, "Failed to read resnet18_test_input.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_input, h_input, 602112, cudaMemcpyHostToDevice));
  float* h_conv1_w = (float*)malloc(37632);
  if (!h_conv1_w) { fprintf(stderr, "malloc failed for conv1_w\n"); return 1; }
  { FILE* f = fopen("resnet18_test_conv1_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_conv1_w.bin\n"); return 1; }
    size_t nread = fread(h_conv1_w, sizeof(float), 9408, f);
    fclose(f);
    if (nread != 9408) { fprintf(stderr, "Failed to read resnet18_test_conv1_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv1_w, h_conv1_w, 37632, cudaMemcpyHostToDevice));
  float* h_conv1_b = (float*)malloc(256);
  if (!h_conv1_b) { fprintf(stderr, "malloc failed for conv1_b\n"); return 1; }
  { FILE* f = fopen("resnet18_test_conv1_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_conv1_b.bin\n"); return 1; }
    size_t nread = fread(h_conv1_b, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_conv1_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv1_b, h_conv1_b, 256, cudaMemcpyHostToDevice));
  float* h_bn1_mean = (float*)malloc(256);
  if (!h_bn1_mean) { fprintf(stderr, "malloc failed for bn1_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_bn1_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_bn1_mean.bin\n"); return 1; }
    size_t nread = fread(h_bn1_mean, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_bn1_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_bn1_mean, h_bn1_mean, 256, cudaMemcpyHostToDevice));
  float* h_bn1_var = (float*)malloc(256);
  if (!h_bn1_var) { fprintf(stderr, "malloc failed for bn1_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_bn1_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_bn1_var.bin\n"); return 1; }
    size_t nread = fread(h_bn1_var, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_bn1_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_bn1_var, h_bn1_var, 256, cudaMemcpyHostToDevice));
  float* h_bn1_gamma = (float*)malloc(256);
  if (!h_bn1_gamma) { fprintf(stderr, "malloc failed for bn1_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_bn1_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_bn1_gamma.bin\n"); return 1; }
    size_t nread = fread(h_bn1_gamma, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_bn1_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_bn1_gamma, h_bn1_gamma, 256, cudaMemcpyHostToDevice));
  float* h_bn1_beta = (float*)malloc(256);
  if (!h_bn1_beta) { fprintf(stderr, "malloc failed for bn1_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_bn1_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_bn1_beta.bin\n"); return 1; }
    size_t nread = fread(h_bn1_beta, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_bn1_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_bn1_beta, h_bn1_beta, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_w1 = (float*)malloc(147456);
  if (!h_l1_b1_w1) { fprintf(stderr, "malloc failed for l1_b1_w1\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b1_w1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b1_w1.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_w1, sizeof(float), 36864, f);
    fclose(f);
    if (nread != 36864) { fprintf(stderr, "Failed to read resnet18_test_l1_b1_w1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_w1, h_l1_b1_w1, 147456, cudaMemcpyHostToDevice));
  float* h_l1_b1_b1 = (float*)malloc(256);
  if (!h_l1_b1_b1) { fprintf(stderr, "malloc failed for l1_b1_b1\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b1_b1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b1_b1.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_b1, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b1_b1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_b1, h_l1_b1_b1, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_w2 = (float*)malloc(147456);
  if (!h_l1_b1_w2) { fprintf(stderr, "malloc failed for l1_b1_w2\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b1_w2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b1_w2.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_w2, sizeof(float), 36864, f);
    fclose(f);
    if (nread != 36864) { fprintf(stderr, "Failed to read resnet18_test_l1_b1_w2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_w2, h_l1_b1_w2, 147456, cudaMemcpyHostToDevice));
  float* h_l1_b1_b2 = (float*)malloc(256);
  if (!h_l1_b1_b2) { fprintf(stderr, "malloc failed for l1_b1_b2\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b1_b2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b1_b2.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_b2, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b1_b2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_b2, h_l1_b1_b2, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_bn1_mean = (float*)malloc(256);
  if (!h_l1_b1_bn1_mean) { fprintf(stderr, "malloc failed for l1_b1_bn1_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b1_bn1_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b1_bn1_mean.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_bn1_mean, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b1_bn1_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_bn1_mean, h_l1_b1_bn1_mean, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_bn1_var = (float*)malloc(256);
  if (!h_l1_b1_bn1_var) { fprintf(stderr, "malloc failed for l1_b1_bn1_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b1_bn1_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b1_bn1_var.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_bn1_var, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b1_bn1_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_bn1_var, h_l1_b1_bn1_var, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_bn1_gamma = (float*)malloc(256);
  if (!h_l1_b1_bn1_gamma) { fprintf(stderr, "malloc failed for l1_b1_bn1_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b1_bn1_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b1_bn1_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_bn1_gamma, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b1_bn1_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_bn1_gamma, h_l1_b1_bn1_gamma, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_bn1_beta = (float*)malloc(256);
  if (!h_l1_b1_bn1_beta) { fprintf(stderr, "malloc failed for l1_b1_bn1_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b1_bn1_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b1_bn1_beta.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_bn1_beta, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b1_bn1_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_bn1_beta, h_l1_b1_bn1_beta, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_bn2_mean = (float*)malloc(256);
  if (!h_l1_b1_bn2_mean) { fprintf(stderr, "malloc failed for l1_b1_bn2_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b1_bn2_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b1_bn2_mean.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_bn2_mean, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b1_bn2_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_bn2_mean, h_l1_b1_bn2_mean, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_bn2_var = (float*)malloc(256);
  if (!h_l1_b1_bn2_var) { fprintf(stderr, "malloc failed for l1_b1_bn2_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b1_bn2_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b1_bn2_var.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_bn2_var, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b1_bn2_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_bn2_var, h_l1_b1_bn2_var, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_bn2_gamma = (float*)malloc(256);
  if (!h_l1_b1_bn2_gamma) { fprintf(stderr, "malloc failed for l1_b1_bn2_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b1_bn2_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b1_bn2_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_bn2_gamma, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b1_bn2_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_bn2_gamma, h_l1_b1_bn2_gamma, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_bn2_beta = (float*)malloc(256);
  if (!h_l1_b1_bn2_beta) { fprintf(stderr, "malloc failed for l1_b1_bn2_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b1_bn2_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b1_bn2_beta.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_bn2_beta, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b1_bn2_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_bn2_beta, h_l1_b1_bn2_beta, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_w1 = (float*)malloc(147456);
  if (!h_l1_b2_w1) { fprintf(stderr, "malloc failed for l1_b2_w1\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b2_w1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b2_w1.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_w1, sizeof(float), 36864, f);
    fclose(f);
    if (nread != 36864) { fprintf(stderr, "Failed to read resnet18_test_l1_b2_w1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_w1, h_l1_b2_w1, 147456, cudaMemcpyHostToDevice));
  float* h_l1_b2_b1 = (float*)malloc(256);
  if (!h_l1_b2_b1) { fprintf(stderr, "malloc failed for l1_b2_b1\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b2_b1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b2_b1.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_b1, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b2_b1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_b1, h_l1_b2_b1, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_w2 = (float*)malloc(147456);
  if (!h_l1_b2_w2) { fprintf(stderr, "malloc failed for l1_b2_w2\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b2_w2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b2_w2.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_w2, sizeof(float), 36864, f);
    fclose(f);
    if (nread != 36864) { fprintf(stderr, "Failed to read resnet18_test_l1_b2_w2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_w2, h_l1_b2_w2, 147456, cudaMemcpyHostToDevice));
  float* h_l1_b2_b2 = (float*)malloc(256);
  if (!h_l1_b2_b2) { fprintf(stderr, "malloc failed for l1_b2_b2\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b2_b2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b2_b2.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_b2, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b2_b2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_b2, h_l1_b2_b2, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_bn1_mean = (float*)malloc(256);
  if (!h_l1_b2_bn1_mean) { fprintf(stderr, "malloc failed for l1_b2_bn1_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b2_bn1_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b2_bn1_mean.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_bn1_mean, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b2_bn1_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_bn1_mean, h_l1_b2_bn1_mean, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_bn1_var = (float*)malloc(256);
  if (!h_l1_b2_bn1_var) { fprintf(stderr, "malloc failed for l1_b2_bn1_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b2_bn1_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b2_bn1_var.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_bn1_var, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b2_bn1_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_bn1_var, h_l1_b2_bn1_var, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_bn1_gamma = (float*)malloc(256);
  if (!h_l1_b2_bn1_gamma) { fprintf(stderr, "malloc failed for l1_b2_bn1_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b2_bn1_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b2_bn1_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_bn1_gamma, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b2_bn1_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_bn1_gamma, h_l1_b2_bn1_gamma, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_bn1_beta = (float*)malloc(256);
  if (!h_l1_b2_bn1_beta) { fprintf(stderr, "malloc failed for l1_b2_bn1_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b2_bn1_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b2_bn1_beta.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_bn1_beta, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b2_bn1_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_bn1_beta, h_l1_b2_bn1_beta, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_bn2_mean = (float*)malloc(256);
  if (!h_l1_b2_bn2_mean) { fprintf(stderr, "malloc failed for l1_b2_bn2_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b2_bn2_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b2_bn2_mean.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_bn2_mean, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b2_bn2_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_bn2_mean, h_l1_b2_bn2_mean, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_bn2_var = (float*)malloc(256);
  if (!h_l1_b2_bn2_var) { fprintf(stderr, "malloc failed for l1_b2_bn2_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b2_bn2_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b2_bn2_var.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_bn2_var, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b2_bn2_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_bn2_var, h_l1_b2_bn2_var, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_bn2_gamma = (float*)malloc(256);
  if (!h_l1_b2_bn2_gamma) { fprintf(stderr, "malloc failed for l1_b2_bn2_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b2_bn2_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b2_bn2_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_bn2_gamma, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b2_bn2_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_bn2_gamma, h_l1_b2_bn2_gamma, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_bn2_beta = (float*)malloc(256);
  if (!h_l1_b2_bn2_beta) { fprintf(stderr, "malloc failed for l1_b2_bn2_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l1_b2_bn2_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l1_b2_bn2_beta.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_bn2_beta, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_test_l1_b2_bn2_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_bn2_beta, h_l1_b2_bn2_beta, 256, cudaMemcpyHostToDevice));
  float* h_l2_b1_w1 = (float*)malloc(294912);
  if (!h_l2_b1_w1) { fprintf(stderr, "malloc failed for l2_b1_w1\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b1_w1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b1_w1.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_w1, sizeof(float), 73728, f);
    fclose(f);
    if (nread != 73728) { fprintf(stderr, "Failed to read resnet18_test_l2_b1_w1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_w1, h_l2_b1_w1, 294912, cudaMemcpyHostToDevice));
  float* h_l2_b1_b1 = (float*)malloc(512);
  if (!h_l2_b1_b1) { fprintf(stderr, "malloc failed for l2_b1_b1\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b1_b1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b1_b1.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_b1, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b1_b1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_b1, h_l2_b1_b1, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_w2 = (float*)malloc(589824);
  if (!h_l2_b1_w2) { fprintf(stderr, "malloc failed for l2_b1_w2\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b1_w2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b1_w2.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_w2, sizeof(float), 147456, f);
    fclose(f);
    if (nread != 147456) { fprintf(stderr, "Failed to read resnet18_test_l2_b1_w2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_w2, h_l2_b1_w2, 589824, cudaMemcpyHostToDevice));
  float* h_l2_b1_b2 = (float*)malloc(512);
  if (!h_l2_b1_b2) { fprintf(stderr, "malloc failed for l2_b1_b2\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b1_b2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b1_b2.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_b2, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b1_b2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_b2, h_l2_b1_b2, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_bn1_mean = (float*)malloc(512);
  if (!h_l2_b1_bn1_mean) { fprintf(stderr, "malloc failed for l2_b1_bn1_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b1_bn1_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b1_bn1_mean.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_bn1_mean, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b1_bn1_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_bn1_mean, h_l2_b1_bn1_mean, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_bn1_var = (float*)malloc(512);
  if (!h_l2_b1_bn1_var) { fprintf(stderr, "malloc failed for l2_b1_bn1_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b1_bn1_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b1_bn1_var.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_bn1_var, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b1_bn1_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_bn1_var, h_l2_b1_bn1_var, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_bn1_gamma = (float*)malloc(512);
  if (!h_l2_b1_bn1_gamma) { fprintf(stderr, "malloc failed for l2_b1_bn1_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b1_bn1_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b1_bn1_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_bn1_gamma, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b1_bn1_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_bn1_gamma, h_l2_b1_bn1_gamma, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_bn1_beta = (float*)malloc(512);
  if (!h_l2_b1_bn1_beta) { fprintf(stderr, "malloc failed for l2_b1_bn1_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b1_bn1_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b1_bn1_beta.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_bn1_beta, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b1_bn1_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_bn1_beta, h_l2_b1_bn1_beta, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_bn2_mean = (float*)malloc(512);
  if (!h_l2_b1_bn2_mean) { fprintf(stderr, "malloc failed for l2_b1_bn2_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b1_bn2_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b1_bn2_mean.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_bn2_mean, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b1_bn2_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_bn2_mean, h_l2_b1_bn2_mean, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_bn2_var = (float*)malloc(512);
  if (!h_l2_b1_bn2_var) { fprintf(stderr, "malloc failed for l2_b1_bn2_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b1_bn2_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b1_bn2_var.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_bn2_var, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b1_bn2_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_bn2_var, h_l2_b1_bn2_var, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_bn2_gamma = (float*)malloc(512);
  if (!h_l2_b1_bn2_gamma) { fprintf(stderr, "malloc failed for l2_b1_bn2_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b1_bn2_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b1_bn2_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_bn2_gamma, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b1_bn2_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_bn2_gamma, h_l2_b1_bn2_gamma, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_bn2_beta = (float*)malloc(512);
  if (!h_l2_b1_bn2_beta) { fprintf(stderr, "malloc failed for l2_b1_bn2_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b1_bn2_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b1_bn2_beta.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_bn2_beta, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b1_bn2_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_bn2_beta, h_l2_b1_bn2_beta, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_shortcut_w = (float*)malloc(32768);
  if (!h_l2_b1_shortcut_w) { fprintf(stderr, "malloc failed for l2_b1_shortcut_w\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b1_shortcut_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b1_shortcut_w.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_shortcut_w, sizeof(float), 8192, f);
    fclose(f);
    if (nread != 8192) { fprintf(stderr, "Failed to read resnet18_test_l2_b1_shortcut_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_shortcut_w, h_l2_b1_shortcut_w, 32768, cudaMemcpyHostToDevice));
  float* h_l2_b1_shortcut_b = (float*)malloc(512);
  if (!h_l2_b1_shortcut_b) { fprintf(stderr, "malloc failed for l2_b1_shortcut_b\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b1_shortcut_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b1_shortcut_b.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_shortcut_b, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b1_shortcut_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_shortcut_b, h_l2_b1_shortcut_b, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_shortcut_mean = (float*)malloc(512);
  if (!h_l2_b1_shortcut_mean) { fprintf(stderr, "malloc failed for l2_b1_shortcut_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b1_shortcut_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b1_shortcut_mean.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_shortcut_mean, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b1_shortcut_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_shortcut_mean, h_l2_b1_shortcut_mean, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_shortcut_var = (float*)malloc(512);
  if (!h_l2_b1_shortcut_var) { fprintf(stderr, "malloc failed for l2_b1_shortcut_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b1_shortcut_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b1_shortcut_var.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_shortcut_var, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b1_shortcut_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_shortcut_var, h_l2_b1_shortcut_var, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_shortcut_gamma = (float*)malloc(512);
  if (!h_l2_b1_shortcut_gamma) { fprintf(stderr, "malloc failed for l2_b1_shortcut_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b1_shortcut_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b1_shortcut_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_shortcut_gamma, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b1_shortcut_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_shortcut_gamma, h_l2_b1_shortcut_gamma, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_shortcut_beta = (float*)malloc(512);
  if (!h_l2_b1_shortcut_beta) { fprintf(stderr, "malloc failed for l2_b1_shortcut_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b1_shortcut_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b1_shortcut_beta.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_shortcut_beta, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b1_shortcut_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_shortcut_beta, h_l2_b1_shortcut_beta, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_w1 = (float*)malloc(589824);
  if (!h_l2_b2_w1) { fprintf(stderr, "malloc failed for l2_b2_w1\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b2_w1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b2_w1.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_w1, sizeof(float), 147456, f);
    fclose(f);
    if (nread != 147456) { fprintf(stderr, "Failed to read resnet18_test_l2_b2_w1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_w1, h_l2_b2_w1, 589824, cudaMemcpyHostToDevice));
  float* h_l2_b2_b1 = (float*)malloc(512);
  if (!h_l2_b2_b1) { fprintf(stderr, "malloc failed for l2_b2_b1\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b2_b1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b2_b1.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_b1, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b2_b1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_b1, h_l2_b2_b1, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_w2 = (float*)malloc(589824);
  if (!h_l2_b2_w2) { fprintf(stderr, "malloc failed for l2_b2_w2\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b2_w2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b2_w2.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_w2, sizeof(float), 147456, f);
    fclose(f);
    if (nread != 147456) { fprintf(stderr, "Failed to read resnet18_test_l2_b2_w2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_w2, h_l2_b2_w2, 589824, cudaMemcpyHostToDevice));
  float* h_l2_b2_b2 = (float*)malloc(512);
  if (!h_l2_b2_b2) { fprintf(stderr, "malloc failed for l2_b2_b2\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b2_b2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b2_b2.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_b2, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b2_b2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_b2, h_l2_b2_b2, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_bn1_mean = (float*)malloc(512);
  if (!h_l2_b2_bn1_mean) { fprintf(stderr, "malloc failed for l2_b2_bn1_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b2_bn1_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b2_bn1_mean.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_bn1_mean, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b2_bn1_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_bn1_mean, h_l2_b2_bn1_mean, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_bn1_var = (float*)malloc(512);
  if (!h_l2_b2_bn1_var) { fprintf(stderr, "malloc failed for l2_b2_bn1_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b2_bn1_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b2_bn1_var.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_bn1_var, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b2_bn1_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_bn1_var, h_l2_b2_bn1_var, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_bn1_gamma = (float*)malloc(512);
  if (!h_l2_b2_bn1_gamma) { fprintf(stderr, "malloc failed for l2_b2_bn1_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b2_bn1_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b2_bn1_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_bn1_gamma, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b2_bn1_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_bn1_gamma, h_l2_b2_bn1_gamma, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_bn1_beta = (float*)malloc(512);
  if (!h_l2_b2_bn1_beta) { fprintf(stderr, "malloc failed for l2_b2_bn1_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b2_bn1_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b2_bn1_beta.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_bn1_beta, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b2_bn1_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_bn1_beta, h_l2_b2_bn1_beta, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_bn2_mean = (float*)malloc(512);
  if (!h_l2_b2_bn2_mean) { fprintf(stderr, "malloc failed for l2_b2_bn2_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b2_bn2_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b2_bn2_mean.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_bn2_mean, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b2_bn2_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_bn2_mean, h_l2_b2_bn2_mean, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_bn2_var = (float*)malloc(512);
  if (!h_l2_b2_bn2_var) { fprintf(stderr, "malloc failed for l2_b2_bn2_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b2_bn2_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b2_bn2_var.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_bn2_var, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b2_bn2_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_bn2_var, h_l2_b2_bn2_var, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_bn2_gamma = (float*)malloc(512);
  if (!h_l2_b2_bn2_gamma) { fprintf(stderr, "malloc failed for l2_b2_bn2_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b2_bn2_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b2_bn2_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_bn2_gamma, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b2_bn2_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_bn2_gamma, h_l2_b2_bn2_gamma, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_bn2_beta = (float*)malloc(512);
  if (!h_l2_b2_bn2_beta) { fprintf(stderr, "malloc failed for l2_b2_bn2_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l2_b2_bn2_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l2_b2_bn2_beta.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_bn2_beta, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_test_l2_b2_bn2_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_bn2_beta, h_l2_b2_bn2_beta, 512, cudaMemcpyHostToDevice));
  float* h_l3_b1_w1 = (float*)malloc(1179648);
  if (!h_l3_b1_w1) { fprintf(stderr, "malloc failed for l3_b1_w1\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b1_w1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b1_w1.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_w1, sizeof(float), 294912, f);
    fclose(f);
    if (nread != 294912) { fprintf(stderr, "Failed to read resnet18_test_l3_b1_w1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_w1, h_l3_b1_w1, 1179648, cudaMemcpyHostToDevice));
  float* h_l3_b1_b1 = (float*)malloc(1024);
  if (!h_l3_b1_b1) { fprintf(stderr, "malloc failed for l3_b1_b1\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b1_b1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b1_b1.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_b1, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b1_b1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_b1, h_l3_b1_b1, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_w2 = (float*)malloc(2359296);
  if (!h_l3_b1_w2) { fprintf(stderr, "malloc failed for l3_b1_w2\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b1_w2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b1_w2.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_w2, sizeof(float), 589824, f);
    fclose(f);
    if (nread != 589824) { fprintf(stderr, "Failed to read resnet18_test_l3_b1_w2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_w2, h_l3_b1_w2, 2359296, cudaMemcpyHostToDevice));
  float* h_l3_b1_b2 = (float*)malloc(1024);
  if (!h_l3_b1_b2) { fprintf(stderr, "malloc failed for l3_b1_b2\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b1_b2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b1_b2.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_b2, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b1_b2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_b2, h_l3_b1_b2, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_bn1_mean = (float*)malloc(1024);
  if (!h_l3_b1_bn1_mean) { fprintf(stderr, "malloc failed for l3_b1_bn1_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b1_bn1_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b1_bn1_mean.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_bn1_mean, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b1_bn1_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_bn1_mean, h_l3_b1_bn1_mean, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_bn1_var = (float*)malloc(1024);
  if (!h_l3_b1_bn1_var) { fprintf(stderr, "malloc failed for l3_b1_bn1_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b1_bn1_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b1_bn1_var.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_bn1_var, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b1_bn1_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_bn1_var, h_l3_b1_bn1_var, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_bn1_gamma = (float*)malloc(1024);
  if (!h_l3_b1_bn1_gamma) { fprintf(stderr, "malloc failed for l3_b1_bn1_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b1_bn1_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b1_bn1_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_bn1_gamma, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b1_bn1_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_bn1_gamma, h_l3_b1_bn1_gamma, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_bn1_beta = (float*)malloc(1024);
  if (!h_l3_b1_bn1_beta) { fprintf(stderr, "malloc failed for l3_b1_bn1_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b1_bn1_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b1_bn1_beta.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_bn1_beta, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b1_bn1_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_bn1_beta, h_l3_b1_bn1_beta, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_bn2_mean = (float*)malloc(1024);
  if (!h_l3_b1_bn2_mean) { fprintf(stderr, "malloc failed for l3_b1_bn2_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b1_bn2_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b1_bn2_mean.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_bn2_mean, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b1_bn2_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_bn2_mean, h_l3_b1_bn2_mean, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_bn2_var = (float*)malloc(1024);
  if (!h_l3_b1_bn2_var) { fprintf(stderr, "malloc failed for l3_b1_bn2_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b1_bn2_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b1_bn2_var.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_bn2_var, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b1_bn2_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_bn2_var, h_l3_b1_bn2_var, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_bn2_gamma = (float*)malloc(1024);
  if (!h_l3_b1_bn2_gamma) { fprintf(stderr, "malloc failed for l3_b1_bn2_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b1_bn2_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b1_bn2_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_bn2_gamma, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b1_bn2_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_bn2_gamma, h_l3_b1_bn2_gamma, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_bn2_beta = (float*)malloc(1024);
  if (!h_l3_b1_bn2_beta) { fprintf(stderr, "malloc failed for l3_b1_bn2_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b1_bn2_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b1_bn2_beta.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_bn2_beta, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b1_bn2_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_bn2_beta, h_l3_b1_bn2_beta, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_shortcut_w = (float*)malloc(131072);
  if (!h_l3_b1_shortcut_w) { fprintf(stderr, "malloc failed for l3_b1_shortcut_w\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b1_shortcut_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b1_shortcut_w.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_shortcut_w, sizeof(float), 32768, f);
    fclose(f);
    if (nread != 32768) { fprintf(stderr, "Failed to read resnet18_test_l3_b1_shortcut_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_shortcut_w, h_l3_b1_shortcut_w, 131072, cudaMemcpyHostToDevice));
  float* h_l3_b1_shortcut_b = (float*)malloc(1024);
  if (!h_l3_b1_shortcut_b) { fprintf(stderr, "malloc failed for l3_b1_shortcut_b\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b1_shortcut_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b1_shortcut_b.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_shortcut_b, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b1_shortcut_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_shortcut_b, h_l3_b1_shortcut_b, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_shortcut_mean = (float*)malloc(1024);
  if (!h_l3_b1_shortcut_mean) { fprintf(stderr, "malloc failed for l3_b1_shortcut_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b1_shortcut_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b1_shortcut_mean.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_shortcut_mean, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b1_shortcut_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_shortcut_mean, h_l3_b1_shortcut_mean, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_shortcut_var = (float*)malloc(1024);
  if (!h_l3_b1_shortcut_var) { fprintf(stderr, "malloc failed for l3_b1_shortcut_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b1_shortcut_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b1_shortcut_var.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_shortcut_var, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b1_shortcut_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_shortcut_var, h_l3_b1_shortcut_var, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_shortcut_gamma = (float*)malloc(1024);
  if (!h_l3_b1_shortcut_gamma) { fprintf(stderr, "malloc failed for l3_b1_shortcut_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b1_shortcut_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b1_shortcut_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_shortcut_gamma, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b1_shortcut_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_shortcut_gamma, h_l3_b1_shortcut_gamma, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_shortcut_beta = (float*)malloc(1024);
  if (!h_l3_b1_shortcut_beta) { fprintf(stderr, "malloc failed for l3_b1_shortcut_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b1_shortcut_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b1_shortcut_beta.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_shortcut_beta, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b1_shortcut_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_shortcut_beta, h_l3_b1_shortcut_beta, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_w1 = (float*)malloc(2359296);
  if (!h_l3_b2_w1) { fprintf(stderr, "malloc failed for l3_b2_w1\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b2_w1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b2_w1.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_w1, sizeof(float), 589824, f);
    fclose(f);
    if (nread != 589824) { fprintf(stderr, "Failed to read resnet18_test_l3_b2_w1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_w1, h_l3_b2_w1, 2359296, cudaMemcpyHostToDevice));
  float* h_l3_b2_b1 = (float*)malloc(1024);
  if (!h_l3_b2_b1) { fprintf(stderr, "malloc failed for l3_b2_b1\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b2_b1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b2_b1.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_b1, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b2_b1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_b1, h_l3_b2_b1, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_w2 = (float*)malloc(2359296);
  if (!h_l3_b2_w2) { fprintf(stderr, "malloc failed for l3_b2_w2\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b2_w2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b2_w2.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_w2, sizeof(float), 589824, f);
    fclose(f);
    if (nread != 589824) { fprintf(stderr, "Failed to read resnet18_test_l3_b2_w2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_w2, h_l3_b2_w2, 2359296, cudaMemcpyHostToDevice));
  float* h_l3_b2_b2 = (float*)malloc(1024);
  if (!h_l3_b2_b2) { fprintf(stderr, "malloc failed for l3_b2_b2\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b2_b2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b2_b2.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_b2, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b2_b2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_b2, h_l3_b2_b2, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_bn1_mean = (float*)malloc(1024);
  if (!h_l3_b2_bn1_mean) { fprintf(stderr, "malloc failed for l3_b2_bn1_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b2_bn1_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b2_bn1_mean.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_bn1_mean, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b2_bn1_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_bn1_mean, h_l3_b2_bn1_mean, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_bn1_var = (float*)malloc(1024);
  if (!h_l3_b2_bn1_var) { fprintf(stderr, "malloc failed for l3_b2_bn1_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b2_bn1_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b2_bn1_var.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_bn1_var, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b2_bn1_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_bn1_var, h_l3_b2_bn1_var, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_bn1_gamma = (float*)malloc(1024);
  if (!h_l3_b2_bn1_gamma) { fprintf(stderr, "malloc failed for l3_b2_bn1_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b2_bn1_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b2_bn1_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_bn1_gamma, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b2_bn1_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_bn1_gamma, h_l3_b2_bn1_gamma, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_bn1_beta = (float*)malloc(1024);
  if (!h_l3_b2_bn1_beta) { fprintf(stderr, "malloc failed for l3_b2_bn1_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b2_bn1_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b2_bn1_beta.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_bn1_beta, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b2_bn1_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_bn1_beta, h_l3_b2_bn1_beta, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_bn2_mean = (float*)malloc(1024);
  if (!h_l3_b2_bn2_mean) { fprintf(stderr, "malloc failed for l3_b2_bn2_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b2_bn2_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b2_bn2_mean.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_bn2_mean, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b2_bn2_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_bn2_mean, h_l3_b2_bn2_mean, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_bn2_var = (float*)malloc(1024);
  if (!h_l3_b2_bn2_var) { fprintf(stderr, "malloc failed for l3_b2_bn2_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b2_bn2_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b2_bn2_var.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_bn2_var, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b2_bn2_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_bn2_var, h_l3_b2_bn2_var, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_bn2_gamma = (float*)malloc(1024);
  if (!h_l3_b2_bn2_gamma) { fprintf(stderr, "malloc failed for l3_b2_bn2_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b2_bn2_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b2_bn2_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_bn2_gamma, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b2_bn2_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_bn2_gamma, h_l3_b2_bn2_gamma, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_bn2_beta = (float*)malloc(1024);
  if (!h_l3_b2_bn2_beta) { fprintf(stderr, "malloc failed for l3_b2_bn2_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l3_b2_bn2_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l3_b2_bn2_beta.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_bn2_beta, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_test_l3_b2_bn2_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_bn2_beta, h_l3_b2_bn2_beta, 1024, cudaMemcpyHostToDevice));
  float* h_l4_b1_w1 = (float*)malloc(4718592);
  if (!h_l4_b1_w1) { fprintf(stderr, "malloc failed for l4_b1_w1\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b1_w1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b1_w1.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_w1, sizeof(float), 1179648, f);
    fclose(f);
    if (nread != 1179648) { fprintf(stderr, "Failed to read resnet18_test_l4_b1_w1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_w1, h_l4_b1_w1, 4718592, cudaMemcpyHostToDevice));
  float* h_l4_b1_b1 = (float*)malloc(2048);
  if (!h_l4_b1_b1) { fprintf(stderr, "malloc failed for l4_b1_b1\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b1_b1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b1_b1.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_b1, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b1_b1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_b1, h_l4_b1_b1, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_w2 = (float*)malloc(9437184);
  if (!h_l4_b1_w2) { fprintf(stderr, "malloc failed for l4_b1_w2\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b1_w2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b1_w2.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_w2, sizeof(float), 2359296, f);
    fclose(f);
    if (nread != 2359296) { fprintf(stderr, "Failed to read resnet18_test_l4_b1_w2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_w2, h_l4_b1_w2, 9437184, cudaMemcpyHostToDevice));
  float* h_l4_b1_b2 = (float*)malloc(2048);
  if (!h_l4_b1_b2) { fprintf(stderr, "malloc failed for l4_b1_b2\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b1_b2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b1_b2.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_b2, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b1_b2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_b2, h_l4_b1_b2, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_bn1_mean = (float*)malloc(2048);
  if (!h_l4_b1_bn1_mean) { fprintf(stderr, "malloc failed for l4_b1_bn1_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b1_bn1_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b1_bn1_mean.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_bn1_mean, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b1_bn1_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_bn1_mean, h_l4_b1_bn1_mean, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_bn1_var = (float*)malloc(2048);
  if (!h_l4_b1_bn1_var) { fprintf(stderr, "malloc failed for l4_b1_bn1_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b1_bn1_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b1_bn1_var.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_bn1_var, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b1_bn1_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_bn1_var, h_l4_b1_bn1_var, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_bn1_gamma = (float*)malloc(2048);
  if (!h_l4_b1_bn1_gamma) { fprintf(stderr, "malloc failed for l4_b1_bn1_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b1_bn1_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b1_bn1_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_bn1_gamma, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b1_bn1_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_bn1_gamma, h_l4_b1_bn1_gamma, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_bn1_beta = (float*)malloc(2048);
  if (!h_l4_b1_bn1_beta) { fprintf(stderr, "malloc failed for l4_b1_bn1_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b1_bn1_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b1_bn1_beta.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_bn1_beta, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b1_bn1_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_bn1_beta, h_l4_b1_bn1_beta, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_bn2_mean = (float*)malloc(2048);
  if (!h_l4_b1_bn2_mean) { fprintf(stderr, "malloc failed for l4_b1_bn2_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b1_bn2_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b1_bn2_mean.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_bn2_mean, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b1_bn2_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_bn2_mean, h_l4_b1_bn2_mean, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_bn2_var = (float*)malloc(2048);
  if (!h_l4_b1_bn2_var) { fprintf(stderr, "malloc failed for l4_b1_bn2_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b1_bn2_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b1_bn2_var.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_bn2_var, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b1_bn2_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_bn2_var, h_l4_b1_bn2_var, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_bn2_gamma = (float*)malloc(2048);
  if (!h_l4_b1_bn2_gamma) { fprintf(stderr, "malloc failed for l4_b1_bn2_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b1_bn2_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b1_bn2_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_bn2_gamma, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b1_bn2_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_bn2_gamma, h_l4_b1_bn2_gamma, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_bn2_beta = (float*)malloc(2048);
  if (!h_l4_b1_bn2_beta) { fprintf(stderr, "malloc failed for l4_b1_bn2_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b1_bn2_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b1_bn2_beta.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_bn2_beta, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b1_bn2_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_bn2_beta, h_l4_b1_bn2_beta, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_shortcut_w = (float*)malloc(524288);
  if (!h_l4_b1_shortcut_w) { fprintf(stderr, "malloc failed for l4_b1_shortcut_w\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b1_shortcut_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b1_shortcut_w.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_shortcut_w, sizeof(float), 131072, f);
    fclose(f);
    if (nread != 131072) { fprintf(stderr, "Failed to read resnet18_test_l4_b1_shortcut_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_shortcut_w, h_l4_b1_shortcut_w, 524288, cudaMemcpyHostToDevice));
  float* h_l4_b1_shortcut_b = (float*)malloc(2048);
  if (!h_l4_b1_shortcut_b) { fprintf(stderr, "malloc failed for l4_b1_shortcut_b\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b1_shortcut_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b1_shortcut_b.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_shortcut_b, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b1_shortcut_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_shortcut_b, h_l4_b1_shortcut_b, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_shortcut_mean = (float*)malloc(2048);
  if (!h_l4_b1_shortcut_mean) { fprintf(stderr, "malloc failed for l4_b1_shortcut_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b1_shortcut_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b1_shortcut_mean.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_shortcut_mean, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b1_shortcut_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_shortcut_mean, h_l4_b1_shortcut_mean, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_shortcut_var = (float*)malloc(2048);
  if (!h_l4_b1_shortcut_var) { fprintf(stderr, "malloc failed for l4_b1_shortcut_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b1_shortcut_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b1_shortcut_var.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_shortcut_var, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b1_shortcut_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_shortcut_var, h_l4_b1_shortcut_var, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_shortcut_gamma = (float*)malloc(2048);
  if (!h_l4_b1_shortcut_gamma) { fprintf(stderr, "malloc failed for l4_b1_shortcut_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b1_shortcut_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b1_shortcut_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_shortcut_gamma, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b1_shortcut_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_shortcut_gamma, h_l4_b1_shortcut_gamma, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_shortcut_beta = (float*)malloc(2048);
  if (!h_l4_b1_shortcut_beta) { fprintf(stderr, "malloc failed for l4_b1_shortcut_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b1_shortcut_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b1_shortcut_beta.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_shortcut_beta, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b1_shortcut_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_shortcut_beta, h_l4_b1_shortcut_beta, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_w1 = (float*)malloc(9437184);
  if (!h_l4_b2_w1) { fprintf(stderr, "malloc failed for l4_b2_w1\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b2_w1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b2_w1.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_w1, sizeof(float), 2359296, f);
    fclose(f);
    if (nread != 2359296) { fprintf(stderr, "Failed to read resnet18_test_l4_b2_w1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_w1, h_l4_b2_w1, 9437184, cudaMemcpyHostToDevice));
  float* h_l4_b2_b1 = (float*)malloc(2048);
  if (!h_l4_b2_b1) { fprintf(stderr, "malloc failed for l4_b2_b1\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b2_b1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b2_b1.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_b1, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b2_b1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_b1, h_l4_b2_b1, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_w2 = (float*)malloc(9437184);
  if (!h_l4_b2_w2) { fprintf(stderr, "malloc failed for l4_b2_w2\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b2_w2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b2_w2.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_w2, sizeof(float), 2359296, f);
    fclose(f);
    if (nread != 2359296) { fprintf(stderr, "Failed to read resnet18_test_l4_b2_w2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_w2, h_l4_b2_w2, 9437184, cudaMemcpyHostToDevice));
  float* h_l4_b2_b2 = (float*)malloc(2048);
  if (!h_l4_b2_b2) { fprintf(stderr, "malloc failed for l4_b2_b2\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b2_b2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b2_b2.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_b2, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b2_b2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_b2, h_l4_b2_b2, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_bn1_mean = (float*)malloc(2048);
  if (!h_l4_b2_bn1_mean) { fprintf(stderr, "malloc failed for l4_b2_bn1_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b2_bn1_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b2_bn1_mean.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_bn1_mean, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b2_bn1_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_bn1_mean, h_l4_b2_bn1_mean, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_bn1_var = (float*)malloc(2048);
  if (!h_l4_b2_bn1_var) { fprintf(stderr, "malloc failed for l4_b2_bn1_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b2_bn1_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b2_bn1_var.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_bn1_var, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b2_bn1_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_bn1_var, h_l4_b2_bn1_var, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_bn1_gamma = (float*)malloc(2048);
  if (!h_l4_b2_bn1_gamma) { fprintf(stderr, "malloc failed for l4_b2_bn1_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b2_bn1_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b2_bn1_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_bn1_gamma, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b2_bn1_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_bn1_gamma, h_l4_b2_bn1_gamma, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_bn1_beta = (float*)malloc(2048);
  if (!h_l4_b2_bn1_beta) { fprintf(stderr, "malloc failed for l4_b2_bn1_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b2_bn1_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b2_bn1_beta.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_bn1_beta, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b2_bn1_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_bn1_beta, h_l4_b2_bn1_beta, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_bn2_mean = (float*)malloc(2048);
  if (!h_l4_b2_bn2_mean) { fprintf(stderr, "malloc failed for l4_b2_bn2_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b2_bn2_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b2_bn2_mean.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_bn2_mean, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b2_bn2_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_bn2_mean, h_l4_b2_bn2_mean, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_bn2_var = (float*)malloc(2048);
  if (!h_l4_b2_bn2_var) { fprintf(stderr, "malloc failed for l4_b2_bn2_var\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b2_bn2_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b2_bn2_var.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_bn2_var, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b2_bn2_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_bn2_var, h_l4_b2_bn2_var, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_bn2_gamma = (float*)malloc(2048);
  if (!h_l4_b2_bn2_gamma) { fprintf(stderr, "malloc failed for l4_b2_bn2_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b2_bn2_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b2_bn2_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_bn2_gamma, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b2_bn2_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_bn2_gamma, h_l4_b2_bn2_gamma, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_bn2_beta = (float*)malloc(2048);
  if (!h_l4_b2_bn2_beta) { fprintf(stderr, "malloc failed for l4_b2_bn2_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_test_l4_b2_bn2_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_l4_b2_bn2_beta.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_bn2_beta, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_test_l4_b2_bn2_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_bn2_beta, h_l4_b2_bn2_beta, 2048, cudaMemcpyHostToDevice));
  float* h_fc_w = (float*)malloc(2048000);
  if (!h_fc_w) { fprintf(stderr, "malloc failed for fc_w\n"); return 1; }
  { FILE* f = fopen("resnet18_test_fc_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_fc_w.bin\n"); return 1; }
    size_t nread = fread(h_fc_w, sizeof(float), 512000, f);
    fclose(f);
    if (nread != 512000) { fprintf(stderr, "Failed to read resnet18_test_fc_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc_w, h_fc_w, 2048000, cudaMemcpyHostToDevice));
  float* h_fc_b = (float*)malloc(4000);
  if (!h_fc_b) { fprintf(stderr, "malloc failed for fc_b\n"); return 1; }
  { FILE* f = fopen("resnet18_test_fc_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_fc_b.bin\n"); return 1; }
    size_t nread = fread(h_fc_b, sizeof(float), 1000, f);
    fclose(f);
    if (nread != 1000) { fprintf(stderr, "Failed to read resnet18_test_fc_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc_b, h_fc_b, 4000, cudaMemcpyHostToDevice));
  float h_bn_eps;
  { FILE* f = fopen("resnet18_test_bn_eps.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_test_bn_eps.bin\n"); return 1; }
    fread(&h_bn_eps, sizeof(float), 1, f);
    fclose(f);
  }
  CUDA_CHECK(cudaMemcpyToSymbol(bn_eps, &h_bn_eps, sizeof(h_bn_eps)));
  CUDA_CHECK(cudaEventRecord(__io_stop));
  CUDA_CHECK(cudaEventSynchronize(__io_stop));
  {
    float __io_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&__io_ms, __io_start, __io_stop));
    printf("MEMTIME_H2D resnet18_test %.6f\n", __io_ms);
  }

  float* h_output = (float*)malloc(4000);
  if (!h_output) { fprintf(stderr, "malloc failed for output\n"); return 1; }

  cudaEvent_t __timer_start, __timer_stop;
  CUDA_CHECK(cudaEventCreate(&__timer_start));
  CUDA_CHECK(cudaEventCreate(&__timer_stop));
  for (int __w = 0; __w < 5; __w++) resnet18_test();
  CUDA_CHECK(cudaDeviceSynchronize());
  float __best_ms = 1e30f;
  float __all_ms[20];
  for (int __it = 0; __it < 20; __it++) {
    CUDA_CHECK(cudaEventRecord(__timer_start));
    resnet18_test();
    CUDA_CHECK(cudaEventRecord(__timer_stop));
    CUDA_CHECK(cudaEventSynchronize(__timer_stop));
    float __elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&__elapsed_ms, __timer_start, __timer_stop));
    __all_ms[__it] = __elapsed_ms;
    if (__elapsed_ms < __best_ms) __best_ms = __elapsed_ms;
  }
  for (int __i = 0; __i < 20; __i++)
    for (int __j = __i + 1; __j < 20; __j++)
      if (__all_ms[__j] < __all_ms[__i]) { float __t = __all_ms[__i]; __all_ms[__i] = __all_ms[__j]; __all_ms[__j] = __t; }
    printf("TIME resnet18_test %.6f\n", __best_ms);
    printf("TIMEMED resnet18_test %.6f\n", __all_ms[10]);
  CUDA_CHECK(cudaEventDestroy(__timer_start));
  CUDA_CHECK(cudaEventDestroy(__timer_stop));

  CUDA_CHECK(cudaEventRecord(__io_start));
  CUDA_CHECK(cudaMemcpy(h_output, d_output, 4000, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(__io_stop));
  CUDA_CHECK(cudaEventSynchronize(__io_stop));
  {
    float __d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&__d2h_ms, __io_start, __io_stop));
    printf("MEMTIME_D2H resnet18_test %.6f\n", __d2h_ms);
  }
  CUDA_CHECK(cudaEventDestroy(__io_start));
  CUDA_CHECK(cudaEventDestroy(__io_stop));

  FILE* fout = fopen("resnet18_test_output.bin", "wb");
  if (!fout) { fprintf(stderr, "Cannot open %s\n", "resnet18_test_output.bin"); return 1; }
  fwrite(h_output, sizeof(float), 1000, fout);
  fclose(fout);

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_conv1_w));
  CUDA_CHECK(cudaFree(d_conv1_b));
  CUDA_CHECK(cudaFree(d_bn1_mean));
  CUDA_CHECK(cudaFree(d_bn1_var));
  CUDA_CHECK(cudaFree(d_bn1_gamma));
  CUDA_CHECK(cudaFree(d_bn1_beta));
  CUDA_CHECK(cudaFree(d_l1_b1_w1));
  CUDA_CHECK(cudaFree(d_l1_b1_b1));
  CUDA_CHECK(cudaFree(d_l1_b1_w2));
  CUDA_CHECK(cudaFree(d_l1_b1_b2));
  CUDA_CHECK(cudaFree(d_l1_b1_bn1_mean));
  CUDA_CHECK(cudaFree(d_l1_b1_bn1_var));
  CUDA_CHECK(cudaFree(d_l1_b1_bn1_gamma));
  CUDA_CHECK(cudaFree(d_l1_b1_bn1_beta));
  CUDA_CHECK(cudaFree(d_l1_b1_bn2_mean));
  CUDA_CHECK(cudaFree(d_l1_b1_bn2_var));
  CUDA_CHECK(cudaFree(d_l1_b1_bn2_gamma));
  CUDA_CHECK(cudaFree(d_l1_b1_bn2_beta));
  CUDA_CHECK(cudaFree(d_l1_b2_w1));
  CUDA_CHECK(cudaFree(d_l1_b2_b1));
  CUDA_CHECK(cudaFree(d_l1_b2_w2));
  CUDA_CHECK(cudaFree(d_l1_b2_b2));
  CUDA_CHECK(cudaFree(d_l1_b2_bn1_mean));
  CUDA_CHECK(cudaFree(d_l1_b2_bn1_var));
  CUDA_CHECK(cudaFree(d_l1_b2_bn1_gamma));
  CUDA_CHECK(cudaFree(d_l1_b2_bn1_beta));
  CUDA_CHECK(cudaFree(d_l1_b2_bn2_mean));
  CUDA_CHECK(cudaFree(d_l1_b2_bn2_var));
  CUDA_CHECK(cudaFree(d_l1_b2_bn2_gamma));
  CUDA_CHECK(cudaFree(d_l1_b2_bn2_beta));
  CUDA_CHECK(cudaFree(d_l2_b1_w1));
  CUDA_CHECK(cudaFree(d_l2_b1_b1));
  CUDA_CHECK(cudaFree(d_l2_b1_w2));
  CUDA_CHECK(cudaFree(d_l2_b1_b2));
  CUDA_CHECK(cudaFree(d_l2_b1_bn1_mean));
  CUDA_CHECK(cudaFree(d_l2_b1_bn1_var));
  CUDA_CHECK(cudaFree(d_l2_b1_bn1_gamma));
  CUDA_CHECK(cudaFree(d_l2_b1_bn1_beta));
  CUDA_CHECK(cudaFree(d_l2_b1_bn2_mean));
  CUDA_CHECK(cudaFree(d_l2_b1_bn2_var));
  CUDA_CHECK(cudaFree(d_l2_b1_bn2_gamma));
  CUDA_CHECK(cudaFree(d_l2_b1_bn2_beta));
  CUDA_CHECK(cudaFree(d_l2_b1_shortcut_w));
  CUDA_CHECK(cudaFree(d_l2_b1_shortcut_b));
  CUDA_CHECK(cudaFree(d_l2_b1_shortcut_mean));
  CUDA_CHECK(cudaFree(d_l2_b1_shortcut_var));
  CUDA_CHECK(cudaFree(d_l2_b1_shortcut_gamma));
  CUDA_CHECK(cudaFree(d_l2_b1_shortcut_beta));
  CUDA_CHECK(cudaFree(d_l2_b2_w1));
  CUDA_CHECK(cudaFree(d_l2_b2_b1));
  CUDA_CHECK(cudaFree(d_l2_b2_w2));
  CUDA_CHECK(cudaFree(d_l2_b2_b2));
  CUDA_CHECK(cudaFree(d_l2_b2_bn1_mean));
  CUDA_CHECK(cudaFree(d_l2_b2_bn1_var));
  CUDA_CHECK(cudaFree(d_l2_b2_bn1_gamma));
  CUDA_CHECK(cudaFree(d_l2_b2_bn1_beta));
  CUDA_CHECK(cudaFree(d_l2_b2_bn2_mean));
  CUDA_CHECK(cudaFree(d_l2_b2_bn2_var));
  CUDA_CHECK(cudaFree(d_l2_b2_bn2_gamma));
  CUDA_CHECK(cudaFree(d_l2_b2_bn2_beta));
  CUDA_CHECK(cudaFree(d_l3_b1_w1));
  CUDA_CHECK(cudaFree(d_l3_b1_b1));
  CUDA_CHECK(cudaFree(d_l3_b1_w2));
  CUDA_CHECK(cudaFree(d_l3_b1_b2));
  CUDA_CHECK(cudaFree(d_l3_b1_bn1_mean));
  CUDA_CHECK(cudaFree(d_l3_b1_bn1_var));
  CUDA_CHECK(cudaFree(d_l3_b1_bn1_gamma));
  CUDA_CHECK(cudaFree(d_l3_b1_bn1_beta));
  CUDA_CHECK(cudaFree(d_l3_b1_bn2_mean));
  CUDA_CHECK(cudaFree(d_l3_b1_bn2_var));
  CUDA_CHECK(cudaFree(d_l3_b1_bn2_gamma));
  CUDA_CHECK(cudaFree(d_l3_b1_bn2_beta));
  CUDA_CHECK(cudaFree(d_l3_b1_shortcut_w));
  CUDA_CHECK(cudaFree(d_l3_b1_shortcut_b));
  CUDA_CHECK(cudaFree(d_l3_b1_shortcut_mean));
  CUDA_CHECK(cudaFree(d_l3_b1_shortcut_var));
  CUDA_CHECK(cudaFree(d_l3_b1_shortcut_gamma));
  CUDA_CHECK(cudaFree(d_l3_b1_shortcut_beta));
  CUDA_CHECK(cudaFree(d_l3_b2_w1));
  CUDA_CHECK(cudaFree(d_l3_b2_b1));
  CUDA_CHECK(cudaFree(d_l3_b2_w2));
  CUDA_CHECK(cudaFree(d_l3_b2_b2));
  CUDA_CHECK(cudaFree(d_l3_b2_bn1_mean));
  CUDA_CHECK(cudaFree(d_l3_b2_bn1_var));
  CUDA_CHECK(cudaFree(d_l3_b2_bn1_gamma));
  CUDA_CHECK(cudaFree(d_l3_b2_bn1_beta));
  CUDA_CHECK(cudaFree(d_l3_b2_bn2_mean));
  CUDA_CHECK(cudaFree(d_l3_b2_bn2_var));
  CUDA_CHECK(cudaFree(d_l3_b2_bn2_gamma));
  CUDA_CHECK(cudaFree(d_l3_b2_bn2_beta));
  CUDA_CHECK(cudaFree(d_l4_b1_w1));
  CUDA_CHECK(cudaFree(d_l4_b1_b1));
  CUDA_CHECK(cudaFree(d_l4_b1_w2));
  CUDA_CHECK(cudaFree(d_l4_b1_b2));
  CUDA_CHECK(cudaFree(d_l4_b1_bn1_mean));
  CUDA_CHECK(cudaFree(d_l4_b1_bn1_var));
  CUDA_CHECK(cudaFree(d_l4_b1_bn1_gamma));
  CUDA_CHECK(cudaFree(d_l4_b1_bn1_beta));
  CUDA_CHECK(cudaFree(d_l4_b1_bn2_mean));
  CUDA_CHECK(cudaFree(d_l4_b1_bn2_var));
  CUDA_CHECK(cudaFree(d_l4_b1_bn2_gamma));
  CUDA_CHECK(cudaFree(d_l4_b1_bn2_beta));
  CUDA_CHECK(cudaFree(d_l4_b1_shortcut_w));
  CUDA_CHECK(cudaFree(d_l4_b1_shortcut_b));
  CUDA_CHECK(cudaFree(d_l4_b1_shortcut_mean));
  CUDA_CHECK(cudaFree(d_l4_b1_shortcut_var));
  CUDA_CHECK(cudaFree(d_l4_b1_shortcut_gamma));
  CUDA_CHECK(cudaFree(d_l4_b1_shortcut_beta));
  CUDA_CHECK(cudaFree(d_l4_b2_w1));
  CUDA_CHECK(cudaFree(d_l4_b2_b1));
  CUDA_CHECK(cudaFree(d_l4_b2_w2));
  CUDA_CHECK(cudaFree(d_l4_b2_b2));
  CUDA_CHECK(cudaFree(d_l4_b2_bn1_mean));
  CUDA_CHECK(cudaFree(d_l4_b2_bn1_var));
  CUDA_CHECK(cudaFree(d_l4_b2_bn1_gamma));
  CUDA_CHECK(cudaFree(d_l4_b2_bn1_beta));
  CUDA_CHECK(cudaFree(d_l4_b2_bn2_mean));
  CUDA_CHECK(cudaFree(d_l4_b2_bn2_var));
  CUDA_CHECK(cudaFree(d_l4_b2_bn2_gamma));
  CUDA_CHECK(cudaFree(d_l4_b2_bn2_beta));
  CUDA_CHECK(cudaFree(d_fc_w));
  CUDA_CHECK(cudaFree(d_fc_b));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_relu1_out));
  CUDA_CHECK(cudaFree(d_pool_out));
  CUDA_CHECK(cudaFree(d_l1_b1_relu1_out));
  CUDA_CHECK(cudaFree(d_l1_b1_add_out));
  CUDA_CHECK(cudaFree(d_l1_b2_relu1_out));
  CUDA_CHECK(cudaFree(d_l1_b2_add_out));
  CUDA_CHECK(cudaFree(d_l2_b1_shortcut_out));
  CUDA_CHECK(cudaFree(d_l2_b1_relu1_out));
  CUDA_CHECK(cudaFree(d_l2_b1_add_out));
  CUDA_CHECK(cudaFree(d_l2_b2_relu1_out));
  CUDA_CHECK(cudaFree(d_l2_b2_add_out));
  CUDA_CHECK(cudaFree(d_l3_b1_shortcut_out));
  CUDA_CHECK(cudaFree(d_l3_b1_relu1_out));
  CUDA_CHECK(cudaFree(d_l3_b1_add_out));
  CUDA_CHECK(cudaFree(d_l3_b2_relu1_out));
  CUDA_CHECK(cudaFree(d_l3_b2_add_out));
  CUDA_CHECK(cudaFree(d_l4_b1_shortcut_out));
  CUDA_CHECK(cudaFree(d_l4_b1_relu1_out));
  CUDA_CHECK(cudaFree(d_l4_b1_add_out));
  CUDA_CHECK(cudaFree(d_l4_b2_relu1_out));
  CUDA_CHECK(cudaFree(d_l4_b2_add_out));
  free(h_input);
  free(h_conv1_w);
  free(h_conv1_b);
  free(h_bn1_mean);
  free(h_bn1_var);
  free(h_bn1_gamma);
  free(h_bn1_beta);
  free(h_l1_b1_w1);
  free(h_l1_b1_b1);
  free(h_l1_b1_w2);
  free(h_l1_b1_b2);
  free(h_l1_b1_bn1_mean);
  free(h_l1_b1_bn1_var);
  free(h_l1_b1_bn1_gamma);
  free(h_l1_b1_bn1_beta);
  free(h_l1_b1_bn2_mean);
  free(h_l1_b1_bn2_var);
  free(h_l1_b1_bn2_gamma);
  free(h_l1_b1_bn2_beta);
  free(h_l1_b2_w1);
  free(h_l1_b2_b1);
  free(h_l1_b2_w2);
  free(h_l1_b2_b2);
  free(h_l1_b2_bn1_mean);
  free(h_l1_b2_bn1_var);
  free(h_l1_b2_bn1_gamma);
  free(h_l1_b2_bn1_beta);
  free(h_l1_b2_bn2_mean);
  free(h_l1_b2_bn2_var);
  free(h_l1_b2_bn2_gamma);
  free(h_l1_b2_bn2_beta);
  free(h_l2_b1_w1);
  free(h_l2_b1_b1);
  free(h_l2_b1_w2);
  free(h_l2_b1_b2);
  free(h_l2_b1_bn1_mean);
  free(h_l2_b1_bn1_var);
  free(h_l2_b1_bn1_gamma);
  free(h_l2_b1_bn1_beta);
  free(h_l2_b1_bn2_mean);
  free(h_l2_b1_bn2_var);
  free(h_l2_b1_bn2_gamma);
  free(h_l2_b1_bn2_beta);
  free(h_l2_b1_shortcut_w);
  free(h_l2_b1_shortcut_b);
  free(h_l2_b1_shortcut_mean);
  free(h_l2_b1_shortcut_var);
  free(h_l2_b1_shortcut_gamma);
  free(h_l2_b1_shortcut_beta);
  free(h_l2_b2_w1);
  free(h_l2_b2_b1);
  free(h_l2_b2_w2);
  free(h_l2_b2_b2);
  free(h_l2_b2_bn1_mean);
  free(h_l2_b2_bn1_var);
  free(h_l2_b2_bn1_gamma);
  free(h_l2_b2_bn1_beta);
  free(h_l2_b2_bn2_mean);
  free(h_l2_b2_bn2_var);
  free(h_l2_b2_bn2_gamma);
  free(h_l2_b2_bn2_beta);
  free(h_l3_b1_w1);
  free(h_l3_b1_b1);
  free(h_l3_b1_w2);
  free(h_l3_b1_b2);
  free(h_l3_b1_bn1_mean);
  free(h_l3_b1_bn1_var);
  free(h_l3_b1_bn1_gamma);
  free(h_l3_b1_bn1_beta);
  free(h_l3_b1_bn2_mean);
  free(h_l3_b1_bn2_var);
  free(h_l3_b1_bn2_gamma);
  free(h_l3_b1_bn2_beta);
  free(h_l3_b1_shortcut_w);
  free(h_l3_b1_shortcut_b);
  free(h_l3_b1_shortcut_mean);
  free(h_l3_b1_shortcut_var);
  free(h_l3_b1_shortcut_gamma);
  free(h_l3_b1_shortcut_beta);
  free(h_l3_b2_w1);
  free(h_l3_b2_b1);
  free(h_l3_b2_w2);
  free(h_l3_b2_b2);
  free(h_l3_b2_bn1_mean);
  free(h_l3_b2_bn1_var);
  free(h_l3_b2_bn1_gamma);
  free(h_l3_b2_bn1_beta);
  free(h_l3_b2_bn2_mean);
  free(h_l3_b2_bn2_var);
  free(h_l3_b2_bn2_gamma);
  free(h_l3_b2_bn2_beta);
  free(h_l4_b1_w1);
  free(h_l4_b1_b1);
  free(h_l4_b1_w2);
  free(h_l4_b1_b2);
  free(h_l4_b1_bn1_mean);
  free(h_l4_b1_bn1_var);
  free(h_l4_b1_bn1_gamma);
  free(h_l4_b1_bn1_beta);
  free(h_l4_b1_bn2_mean);
  free(h_l4_b1_bn2_var);
  free(h_l4_b1_bn2_gamma);
  free(h_l4_b1_bn2_beta);
  free(h_l4_b1_shortcut_w);
  free(h_l4_b1_shortcut_b);
  free(h_l4_b1_shortcut_mean);
  free(h_l4_b1_shortcut_var);
  free(h_l4_b1_shortcut_gamma);
  free(h_l4_b1_shortcut_beta);
  free(h_l4_b2_w1);
  free(h_l4_b2_b1);
  free(h_l4_b2_w2);
  free(h_l4_b2_b2);
  free(h_l4_b2_bn1_mean);
  free(h_l4_b2_bn1_var);
  free(h_l4_b2_bn1_gamma);
  free(h_l4_b2_bn1_beta);
  free(h_l4_b2_bn2_mean);
  free(h_l4_b2_bn2_var);
  free(h_l4_b2_bn2_gamma);
  free(h_l4_b2_bn2_beta);
  free(h_fc_w);
  free(h_fc_b);
  free(h_output);
  return 0;
}
