#include <cuda_runtime.h>
#include <math.h>

__global__ void ast_conv1_relu_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 512;
  if (idx >= N) return;
  int i3 = idx % 8; idx /= 8;
  int i2 = idx % 8; idx /= 8;
  int i1 = idx % 8; idx /= 8;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 3; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 9 || pj < 1 || pj >= 9)
                    ? 0.0f
                    : input[i0][c][pi - 1][pj - 1];
        acc += v * conv1_w[i1][c][k1][k2];
      }
  float result = acc + conv1_b[i1];
  conv1_relu[i0][i1][i2][i3] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_conv1_pool()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 128;
  if (idx >= N) return;
  int i3 = idx % 4; idx /= 4;
  int i2 = idx % 4; idx /= 4;
  int i1 = idx % 8; idx /= 8;
  int i0 = idx;
  float best = conv1_relu[i0][i1][2 * i2][2 * i3];
  for (int k1 = 0; k1 < 2; k1 += 1)
    for (int k2 = 0; k2 < 2; k2 += 1) {
      float v = conv1_relu[i0][i1][2 * i2 + k1][2 * i3 + k2];
      if (v > best) best = v;
    }
  conv1_pool[i0][i1][i2][i3] = best;
}

__global__ void ast_conv2_relu_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 256;
  if (idx >= N) return;
  int i3 = idx % 4; idx /= 4;
  int i2 = idx % 4; idx /= 4;
  int i1 = idx % 16; idx /= 16;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 8; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 5 || pj < 1 || pj >= 5)
                    ? 0.0f
                    : conv1_pool[i0][c][pi - 1][pj - 1];
        acc += v * conv2_w[i1][c][k1][k2];
      }
  float result = acc + conv2_b[i1];
  conv2_relu[i0][i1][i2][i3] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_conv2_pool()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 64;
  if (idx >= N) return;
  int i3 = idx % 2; idx /= 2;
  int i2 = idx % 2; idx /= 2;
  int i1 = idx % 16; idx /= 16;
  int i0 = idx;
  float best = conv2_relu[i0][i1][2 * i2][2 * i3];
  for (int k1 = 0; k1 < 2; k1 += 1)
    for (int k2 = 0; k2 < 2; k2 += 1) {
      float v = conv2_relu[i0][i1][2 * i2 + k1][2 * i3 + k2];
      if (v > best) best = v;
    }
  conv2_pool[i0][i1][i2][i3] = best;
}

__global__ void ast_flat()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 64;
  if (idx >= N) return;
  int i1 = idx % 64; idx /= 64;
  int i0 = idx;
  int hw = 4;
  int c = i1 / hw;
  int rem = i1 % hw;
  int h0 = rem / 2;
  int w0 = rem % 2;
  flat[i0][i1] = conv2_pool[i0][c][h0][w0];
}

__global__ void ast_fc1()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 16;
  if (idx >= N) return;
  int i1 = idx % 16; idx /= 16;
  int i0 = idx;
  float acc = 0.0f;
  for (int i = 0; i < 64; i += 1)
    acc += flat[i0][i] * fc1_w[i1][i];
  fc1[i0][i1] = acc + fc1_b[i1];
}

__global__ void ast_fc2()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 10;
  if (idx >= N) return;
  int i1 = idx % 10; idx /= 10;
  int i0 = idx;
  float acc = 0.0f;
  for (int i = 0; i < 16; i += 1)
    acc += fc1[i0][i] * fc2_w[i1][i];
  fc2[i0][i1] = acc + fc2_b[i1];
}

__global__ void ast_output()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 10;
  if (idx >= N) return;
  int i1 = idx % 10; idx /= 10;
  int i0 = idx;
  float sum = 0.0f;
  for (int j = 0; j < 10; j += 1)
    sum += expf(fc2[i0][j]);
  output[i0][i1] = expf(fc2[i0][i1]) / sum;
}

void alexnet_test()
{
  ast_conv1_relu_fused<<<2, 256>>>();
  cudaDeviceSynchronize();
  ast_conv1_pool<<<1, 256>>>();
  cudaDeviceSynchronize();
  ast_conv2_relu_fused<<<1, 256>>>();
  cudaDeviceSynchronize();
  ast_conv2_pool<<<1, 256>>>();
  cudaDeviceSynchronize();
  ast_flat<<<1, 256>>>();
  cudaDeviceSynchronize();
  ast_fc1<<<1, 256>>>();
  cudaDeviceSynchronize();
  ast_fc2<<<1, 256>>>();
  cudaDeviceSynchronize();
  ast_output<<<1, 256>>>();
  cudaDeviceSynchronize();
}
