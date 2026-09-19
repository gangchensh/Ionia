#include <cuda_runtime.h>
#include <math.h>

__global__ void ast_relu1_1_out_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 3211264;
  if (idx >= N) return;
  int i3 = idx % 224; idx /= 224;
  int i2 = idx % 224; idx /= 224;
  int i1 = idx % 64; idx /= 64;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 3; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 225 || pj < 1 || pj >= 225)
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
  int N = 3211264;
  if (idx >= N) return;
  int i3 = idx % 224; idx /= 224;
  int i2 = idx % 224; idx /= 224;
  int i1 = idx % 64; idx /= 64;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 64; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 225 || pj < 1 || pj >= 225)
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
  int N = 802816;
  if (idx >= N) return;
  int i3 = idx % 112; idx /= 112;
  int i2 = idx % 112; idx /= 112;
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
  int N = 1605632;
  if (idx >= N) return;
  int i3 = idx % 112; idx /= 112;
  int i2 = idx % 112; idx /= 112;
  int i1 = idx % 128; idx /= 128;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 64; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 113 || pj < 1 || pj >= 113)
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
  int N = 1605632;
  if (idx >= N) return;
  int i3 = idx % 112; idx /= 112;
  int i2 = idx % 112; idx /= 112;
  int i1 = idx % 128; idx /= 128;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 128; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 113 || pj < 1 || pj >= 113)
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
  int N = 401408;
  if (idx >= N) return;
  int i3 = idx % 56; idx /= 56;
  int i2 = idx % 56; idx /= 56;
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
  int N = 802816;
  if (idx >= N) return;
  int i3 = idx % 56; idx /= 56;
  int i2 = idx % 56; idx /= 56;
  int i1 = idx % 256; idx /= 256;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 128; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 57 || pj < 1 || pj >= 57)
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
  int N = 802816;
  if (idx >= N) return;
  int i3 = idx % 56; idx /= 56;
  int i2 = idx % 56; idx /= 56;
  int i1 = idx % 256; idx /= 256;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 256; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 57 || pj < 1 || pj >= 57)
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
  int N = 802816;
  if (idx >= N) return;
  int i3 = idx % 56; idx /= 56;
  int i2 = idx % 56; idx /= 56;
  int i1 = idx % 256; idx /= 256;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 256; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 57 || pj < 1 || pj >= 57)
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
  int N = 200704;
  if (idx >= N) return;
  int i3 = idx % 28; idx /= 28;
  int i2 = idx % 28; idx /= 28;
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
  int N = 401408;
  if (idx >= N) return;
  int i3 = idx % 28; idx /= 28;
  int i2 = idx % 28; idx /= 28;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 256; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 29 || pj < 1 || pj >= 29)
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
  int N = 401408;
  if (idx >= N) return;
  int i3 = idx % 28; idx /= 28;
  int i2 = idx % 28; idx /= 28;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 512; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 29 || pj < 1 || pj >= 29)
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
  int N = 401408;
  if (idx >= N) return;
  int i3 = idx % 28; idx /= 28;
  int i2 = idx % 28; idx /= 28;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 512; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 29 || pj < 1 || pj >= 29)
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
  int N = 100352;
  if (idx >= N) return;
  int i3 = idx % 14; idx /= 14;
  int i2 = idx % 14; idx /= 14;
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
  int N = 100352;
  if (idx >= N) return;
  int i3 = idx % 14; idx /= 14;
  int i2 = idx % 14; idx /= 14;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 512; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 15 || pj < 1 || pj >= 15)
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
  int N = 100352;
  if (idx >= N) return;
  int i3 = idx % 14; idx /= 14;
  int i2 = idx % 14; idx /= 14;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 512; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 15 || pj < 1 || pj >= 15)
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
  int N = 100352;
  if (idx >= N) return;
  int i3 = idx % 14; idx /= 14;
  int i2 = idx % 14; idx /= 14;
  int i1 = idx % 512; idx /= 512;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 512; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 15 || pj < 1 || pj >= 15)
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
  int N = 25088;
  if (idx >= N) return;
  int i3 = idx % 7; idx /= 7;
  int i2 = idx % 7; idx /= 7;
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
  int N = 25088;
  if (idx >= N) return;
  int i1 = idx % 25088; idx /= 25088;
  int i0 = idx;
  int hw = 49;
  int c = i1 / hw;
  int rem = i1 % hw;
  int h0 = rem / 7;
  int w0 = rem % 7;
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
  for (int i = 0; i < 25088; i += 1)
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

void vgg16p_test()
{
  ast_relu1_1_out_fused<<<12544, 256>>>();
  cudaDeviceSynchronize();
  ast_relu1_2_out_fused<<<12544, 256>>>();
  cudaDeviceSynchronize();
  ast_pool1_out<<<3136, 256>>>();
  cudaDeviceSynchronize();
  ast_relu2_1_out_fused<<<6272, 256>>>();
  cudaDeviceSynchronize();
  ast_relu2_2_out_fused<<<6272, 256>>>();
  cudaDeviceSynchronize();
  ast_pool2_out<<<1568, 256>>>();
  cudaDeviceSynchronize();
  ast_relu3_1_out_fused<<<3136, 256>>>();
  cudaDeviceSynchronize();
  ast_relu3_2_out_fused<<<3136, 256>>>();
  cudaDeviceSynchronize();
  ast_relu3_3_out_fused<<<3136, 256>>>();
  cudaDeviceSynchronize();
  ast_pool3_out<<<784, 256>>>();
  cudaDeviceSynchronize();
  ast_relu4_1_out_fused<<<1568, 256>>>();
  cudaDeviceSynchronize();
  ast_relu4_2_out_fused<<<1568, 256>>>();
  cudaDeviceSynchronize();
  ast_relu4_3_out_fused<<<1568, 256>>>();
  cudaDeviceSynchronize();
  ast_pool4_out<<<392, 256>>>();
  cudaDeviceSynchronize();
  ast_relu5_1_out_fused<<<392, 256>>>();
  cudaDeviceSynchronize();
  ast_relu5_2_out_fused<<<392, 256>>>();
  cudaDeviceSynchronize();
  ast_relu5_3_out_fused<<<392, 256>>>();
  cudaDeviceSynchronize();
  ast_pool5_out<<<98, 256>>>();
  cudaDeviceSynchronize();
  ast_flatten_out<<<98, 256>>>();
  cudaDeviceSynchronize();
  ast_fc1_relu_out_fused<<<16, 256>>>();
  cudaDeviceSynchronize();
  ast_fc2_relu_out_fused<<<16, 256>>>();
  cudaDeviceSynchronize();
  ast_output<<<4, 256>>>();
  cudaDeviceSynchronize();
}
