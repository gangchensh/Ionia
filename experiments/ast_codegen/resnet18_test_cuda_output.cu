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
