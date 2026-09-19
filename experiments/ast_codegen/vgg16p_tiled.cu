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
__constant__ float (*fc1_w)[25088];
__constant__ float* fc1_b;
__constant__ float (*fc2_w)[4096];
__constant__ float* fc2_b;
__constant__ float (*fc3_w)[4096];
__constant__ float* fc3_b;
__constant__ float (*output)[1000];
__constant__ float (*relu1_1_out)[64][224][224];
__constant__ float (*relu1_2_out)[64][224][224];
__constant__ float (*pool1_out)[64][112][112];
__constant__ float (*relu2_1_out)[128][112][112];
__constant__ float (*relu2_2_out)[128][112][112];
__constant__ float (*pool2_out)[128][56][56];
__constant__ float (*relu3_1_out)[256][56][56];
__constant__ float (*relu3_2_out)[256][56][56];
__constant__ float (*relu3_3_out)[256][56][56];
__constant__ float (*pool3_out)[256][28][28];
__constant__ float (*relu4_1_out)[512][28][28];
__constant__ float (*relu4_2_out)[512][28][28];
__constant__ float (*relu4_3_out)[512][28][28];
__constant__ float (*pool4_out)[512][14][14];
__constant__ float (*relu5_1_out)[512][14][14];
__constant__ float (*relu5_2_out)[512][14][14];
__constant__ float (*relu5_3_out)[512][14][14];
__constant__ float (*pool5_out)[512][7][7];
__constant__ float (*flatten_out)[25088];
__constant__ float (*fc1_relu_out)[4096];
__constant__ float (*fc2_relu_out)[4096];
__constant__ float (*relu4_1_out_ws)[512][28][28];
__constant__ float (*relu4_2_out_ws)[512][28][28];
__constant__ float (*relu4_3_out_ws)[512][28][28];
__constant__ float (*relu5_1_out_ws)[512][14][14];
__constant__ float (*relu5_2_out_ws)[512][14][14];
__constant__ float (*relu5_3_out_ws)[512][14][14];

#include <cuda_runtime.h>
#include <math.h>

/*LAUNCH 7 7 8 16 16*/
__global__ void ast_relu1_1_out_fused()
{
  __shared__ float sh2[2][34][35];
  __shared__ float shw2[2][8][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int oy0 = blockIdx.y * 32 + ty * 2;
  int ox0 = blockIdx.x * 32 + tx * 2;
  int ih0 = blockIdx.y * 32 - 1;
  int iw0 = blockIdx.x * 32 - 1;
  int c0 = 0;
  int c1 = 3;
  int oc0 = blockIdx.z * 8;
  float acc[8][2][2];
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1)
    for (int dy = 0; dy < 2; dy += 1)
      for (int dx = 0; dx < 2; dx += 1)
        acc[cb][dy][dx] = 0.0f;
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[0][sy][sx] = (iy < 0 || iy >= 224 || ix < 0 || ix >= 224)
                     ? 0.0f
                     : input[0][c0][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[0][cb][t] = (oc0 + cb < 64) ? conv1_1_w[oc0 + cb][c0][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
  __syncthreads();
  for (int c = c0; c < c1; c += 1) {
    int cur = (c - c0) & 1;
    if (c + 1 < c1) {
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[cur ^ 1][sy][sx] = (iy < 0 || iy >= 224 || ix < 0 || ix >= 224)
                     ? 0.0f
                     : input[0][(c + 1)][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[cur ^ 1][cb][t] = (oc0 + cb < 64) ? conv1_1_w[oc0 + cb][(c + 1)][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
    }
    float win[4][4];
    #pragma unroll
    for (int wy = 0; wy < 4; wy += 1)
      #pragma unroll
      for (int wx = 0; wx < 4; wx += 1)
        win[wy][wx] = sh2[cur][ty * 2 + wy][tx * 2 + wx];
    #pragma unroll
    for (int cb = 0; cb < 8; cb += 1) {
      if (oc0 + cb < 64) {
        #pragma unroll
        for (int k1 = 0; k1 < 3; k1 += 1)
          #pragma unroll
          for (int k2 = 0; k2 < 3; k2 += 1) {
            float wv = shw2[cur][cb][k1 * 3 + k2];
            acc[cb][0][0] += win[0 + k1][0 + k2] * wv;
            acc[cb][0][1] += win[0 + k1][1 + k2] * wv;
            acc[cb][1][0] += win[1 + k1][0 + k2] * wv;
            acc[cb][1][1] += win[1 + k1][1 + k2] * wv;
          }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 64) {
      if (oy0 < 224 && ox0 < 224) {
        float result = acc[cb][0][0] + conv1_1_b[oc];
        float fused_result = result;
        relu1_1_out[0][oc][oy0][ox0] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 < 224 && ox0 + 1 < 224) {
        float result = acc[cb][0][1] + conv1_1_b[oc];
        float fused_result = result;
        relu1_1_out[0][oc][oy0][ox0 + 1] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 + 1 < 224 && ox0 < 224) {
        float result = acc[cb][1][0] + conv1_1_b[oc];
        float fused_result = result;
        relu1_1_out[0][oc][oy0 + 1][ox0] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 + 1 < 224 && ox0 + 1 < 224) {
        float result = acc[cb][1][1] + conv1_1_b[oc];
        float fused_result = result;
        relu1_1_out[0][oc][oy0 + 1][ox0 + 1] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
    }
  }
}

/*LAUNCH 7 7 8 16 16*/
__global__ void ast_relu1_2_out_fused()
{
  __shared__ float sh2[2][34][35];
  __shared__ float shw2[2][8][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int oy0 = blockIdx.y * 32 + ty * 2;
  int ox0 = blockIdx.x * 32 + tx * 2;
  int ih0 = blockIdx.y * 32 - 1;
  int iw0 = blockIdx.x * 32 - 1;
  int c0 = 0;
  int c1 = 64;
  int oc0 = blockIdx.z * 8;
  float acc[8][2][2];
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1)
    for (int dy = 0; dy < 2; dy += 1)
      for (int dx = 0; dx < 2; dx += 1)
        acc[cb][dy][dx] = 0.0f;
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[0][sy][sx] = (iy < 0 || iy >= 224 || ix < 0 || ix >= 224)
                     ? 0.0f
                     : relu1_1_out[0][c0][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[0][cb][t] = (oc0 + cb < 64) ? conv1_2_w[oc0 + cb][c0][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
  __syncthreads();
  for (int c = c0; c < c1; c += 1) {
    int cur = (c - c0) & 1;
    if (c + 1 < c1) {
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[cur ^ 1][sy][sx] = (iy < 0 || iy >= 224 || ix < 0 || ix >= 224)
                     ? 0.0f
                     : relu1_1_out[0][(c + 1)][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[cur ^ 1][cb][t] = (oc0 + cb < 64) ? conv1_2_w[oc0 + cb][(c + 1)][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
    }
    float win[4][4];
    #pragma unroll
    for (int wy = 0; wy < 4; wy += 1)
      #pragma unroll
      for (int wx = 0; wx < 4; wx += 1)
        win[wy][wx] = sh2[cur][ty * 2 + wy][tx * 2 + wx];
    #pragma unroll
    for (int cb = 0; cb < 8; cb += 1) {
      if (oc0 + cb < 64) {
        #pragma unroll
        for (int k1 = 0; k1 < 3; k1 += 1)
          #pragma unroll
          for (int k2 = 0; k2 < 3; k2 += 1) {
            float wv = shw2[cur][cb][k1 * 3 + k2];
            acc[cb][0][0] += win[0 + k1][0 + k2] * wv;
            acc[cb][0][1] += win[0 + k1][1 + k2] * wv;
            acc[cb][1][0] += win[1 + k1][0 + k2] * wv;
            acc[cb][1][1] += win[1 + k1][1 + k2] * wv;
          }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 64) {
      if (oy0 < 224 && ox0 < 224) {
        float result = acc[cb][0][0] + conv1_2_b[oc];
        float fused_result = result;
        relu1_2_out[0][oc][oy0][ox0] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 < 224 && ox0 + 1 < 224) {
        float result = acc[cb][0][1] + conv1_2_b[oc];
        float fused_result = result;
        relu1_2_out[0][oc][oy0][ox0 + 1] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 + 1 < 224 && ox0 < 224) {
        float result = acc[cb][1][0] + conv1_2_b[oc];
        float fused_result = result;
        relu1_2_out[0][oc][oy0 + 1][ox0] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 + 1 < 224 && ox0 + 1 < 224) {
        float result = acc[cb][1][1] + conv1_2_b[oc];
        float fused_result = result;
        relu1_2_out[0][oc][oy0 + 1][ox0 + 1] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
    }
  }
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

/*LAUNCH 4 4 16 16 16*/
__global__ void ast_relu2_1_out_fused()
{
  __shared__ float sh2[2][34][35];
  __shared__ float shw2[2][8][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int oy0 = blockIdx.y * 32 + ty * 2;
  int ox0 = blockIdx.x * 32 + tx * 2;
  int ih0 = blockIdx.y * 32 - 1;
  int iw0 = blockIdx.x * 32 - 1;
  int c0 = 0;
  int c1 = 64;
  int oc0 = blockIdx.z * 8;
  float acc[8][2][2];
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1)
    for (int dy = 0; dy < 2; dy += 1)
      for (int dx = 0; dx < 2; dx += 1)
        acc[cb][dy][dx] = 0.0f;
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[0][sy][sx] = (iy < 0 || iy >= 112 || ix < 0 || ix >= 112)
                     ? 0.0f
                     : relu1_2_out[0][c0][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[0][cb][t] = (oc0 + cb < 128) ? conv2_1_w[oc0 + cb][c0][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
  __syncthreads();
  for (int c = c0; c < c1; c += 1) {
    int cur = (c - c0) & 1;
    if (c + 1 < c1) {
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[cur ^ 1][sy][sx] = (iy < 0 || iy >= 112 || ix < 0 || ix >= 112)
                     ? 0.0f
                     : relu1_2_out[0][(c + 1)][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[cur ^ 1][cb][t] = (oc0 + cb < 128) ? conv2_1_w[oc0 + cb][(c + 1)][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
    }
    float win[4][4];
    #pragma unroll
    for (int wy = 0; wy < 4; wy += 1)
      #pragma unroll
      for (int wx = 0; wx < 4; wx += 1)
        win[wy][wx] = sh2[cur][ty * 2 + wy][tx * 2 + wx];
    #pragma unroll
    for (int cb = 0; cb < 8; cb += 1) {
      if (oc0 + cb < 128) {
        #pragma unroll
        for (int k1 = 0; k1 < 3; k1 += 1)
          #pragma unroll
          for (int k2 = 0; k2 < 3; k2 += 1) {
            float wv = shw2[cur][cb][k1 * 3 + k2];
            acc[cb][0][0] += win[0 + k1][0 + k2] * wv;
            acc[cb][0][1] += win[0 + k1][1 + k2] * wv;
            acc[cb][1][0] += win[1 + k1][0 + k2] * wv;
            acc[cb][1][1] += win[1 + k1][1 + k2] * wv;
          }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 128) {
      if (oy0 < 112 && ox0 < 112) {
        float result = acc[cb][0][0] + conv2_1_b[oc];
        float fused_result = result;
        relu2_1_out[0][oc][oy0][ox0] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 < 112 && ox0 + 1 < 112) {
        float result = acc[cb][0][1] + conv2_1_b[oc];
        float fused_result = result;
        relu2_1_out[0][oc][oy0][ox0 + 1] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 + 1 < 112 && ox0 < 112) {
        float result = acc[cb][1][0] + conv2_1_b[oc];
        float fused_result = result;
        relu2_1_out[0][oc][oy0 + 1][ox0] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 + 1 < 112 && ox0 + 1 < 112) {
        float result = acc[cb][1][1] + conv2_1_b[oc];
        float fused_result = result;
        relu2_1_out[0][oc][oy0 + 1][ox0 + 1] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
    }
  }
}

/*LAUNCH 4 4 16 16 16*/
__global__ void ast_relu2_2_out_fused()
{
  __shared__ float sh2[2][34][35];
  __shared__ float shw2[2][8][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int oy0 = blockIdx.y * 32 + ty * 2;
  int ox0 = blockIdx.x * 32 + tx * 2;
  int ih0 = blockIdx.y * 32 - 1;
  int iw0 = blockIdx.x * 32 - 1;
  int c0 = 0;
  int c1 = 128;
  int oc0 = blockIdx.z * 8;
  float acc[8][2][2];
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1)
    for (int dy = 0; dy < 2; dy += 1)
      for (int dx = 0; dx < 2; dx += 1)
        acc[cb][dy][dx] = 0.0f;
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[0][sy][sx] = (iy < 0 || iy >= 112 || ix < 0 || ix >= 112)
                     ? 0.0f
                     : relu2_1_out[0][c0][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[0][cb][t] = (oc0 + cb < 128) ? conv2_2_w[oc0 + cb][c0][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
  __syncthreads();
  for (int c = c0; c < c1; c += 1) {
    int cur = (c - c0) & 1;
    if (c + 1 < c1) {
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[cur ^ 1][sy][sx] = (iy < 0 || iy >= 112 || ix < 0 || ix >= 112)
                     ? 0.0f
                     : relu2_1_out[0][(c + 1)][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[cur ^ 1][cb][t] = (oc0 + cb < 128) ? conv2_2_w[oc0 + cb][(c + 1)][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
    }
    float win[4][4];
    #pragma unroll
    for (int wy = 0; wy < 4; wy += 1)
      #pragma unroll
      for (int wx = 0; wx < 4; wx += 1)
        win[wy][wx] = sh2[cur][ty * 2 + wy][tx * 2 + wx];
    #pragma unroll
    for (int cb = 0; cb < 8; cb += 1) {
      if (oc0 + cb < 128) {
        #pragma unroll
        for (int k1 = 0; k1 < 3; k1 += 1)
          #pragma unroll
          for (int k2 = 0; k2 < 3; k2 += 1) {
            float wv = shw2[cur][cb][k1 * 3 + k2];
            acc[cb][0][0] += win[0 + k1][0 + k2] * wv;
            acc[cb][0][1] += win[0 + k1][1 + k2] * wv;
            acc[cb][1][0] += win[1 + k1][0 + k2] * wv;
            acc[cb][1][1] += win[1 + k1][1 + k2] * wv;
          }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 128) {
      if (oy0 < 112 && ox0 < 112) {
        float result = acc[cb][0][0] + conv2_2_b[oc];
        float fused_result = result;
        relu2_2_out[0][oc][oy0][ox0] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 < 112 && ox0 + 1 < 112) {
        float result = acc[cb][0][1] + conv2_2_b[oc];
        float fused_result = result;
        relu2_2_out[0][oc][oy0][ox0 + 1] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 + 1 < 112 && ox0 < 112) {
        float result = acc[cb][1][0] + conv2_2_b[oc];
        float fused_result = result;
        relu2_2_out[0][oc][oy0 + 1][ox0] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 + 1 < 112 && ox0 + 1 < 112) {
        float result = acc[cb][1][1] + conv2_2_b[oc];
        float fused_result = result;
        relu2_2_out[0][oc][oy0 + 1][ox0 + 1] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
    }
  }
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

/*LAUNCH 2 2 32 16 16*/
__global__ void ast_relu3_1_out_fused()
{
  __shared__ float sh2[2][34][35];
  __shared__ float shw2[2][8][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int oy0 = blockIdx.y * 32 + ty * 2;
  int ox0 = blockIdx.x * 32 + tx * 2;
  int ih0 = blockIdx.y * 32 - 1;
  int iw0 = blockIdx.x * 32 - 1;
  int c0 = 0;
  int c1 = 128;
  int oc0 = blockIdx.z * 8;
  float acc[8][2][2];
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1)
    for (int dy = 0; dy < 2; dy += 1)
      for (int dx = 0; dx < 2; dx += 1)
        acc[cb][dy][dx] = 0.0f;
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[0][sy][sx] = (iy < 0 || iy >= 56 || ix < 0 || ix >= 56)
                     ? 0.0f
                     : relu2_2_out[0][c0][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[0][cb][t] = (oc0 + cb < 256) ? conv3_1_w[oc0 + cb][c0][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
  __syncthreads();
  for (int c = c0; c < c1; c += 1) {
    int cur = (c - c0) & 1;
    if (c + 1 < c1) {
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[cur ^ 1][sy][sx] = (iy < 0 || iy >= 56 || ix < 0 || ix >= 56)
                     ? 0.0f
                     : relu2_2_out[0][(c + 1)][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[cur ^ 1][cb][t] = (oc0 + cb < 256) ? conv3_1_w[oc0 + cb][(c + 1)][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
    }
    float win[4][4];
    #pragma unroll
    for (int wy = 0; wy < 4; wy += 1)
      #pragma unroll
      for (int wx = 0; wx < 4; wx += 1)
        win[wy][wx] = sh2[cur][ty * 2 + wy][tx * 2 + wx];
    #pragma unroll
    for (int cb = 0; cb < 8; cb += 1) {
      if (oc0 + cb < 256) {
        #pragma unroll
        for (int k1 = 0; k1 < 3; k1 += 1)
          #pragma unroll
          for (int k2 = 0; k2 < 3; k2 += 1) {
            float wv = shw2[cur][cb][k1 * 3 + k2];
            acc[cb][0][0] += win[0 + k1][0 + k2] * wv;
            acc[cb][0][1] += win[0 + k1][1 + k2] * wv;
            acc[cb][1][0] += win[1 + k1][0 + k2] * wv;
            acc[cb][1][1] += win[1 + k1][1 + k2] * wv;
          }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 256) {
      if (oy0 < 56 && ox0 < 56) {
        float result = acc[cb][0][0] + conv3_1_b[oc];
        float fused_result = result;
        relu3_1_out[0][oc][oy0][ox0] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 < 56 && ox0 + 1 < 56) {
        float result = acc[cb][0][1] + conv3_1_b[oc];
        float fused_result = result;
        relu3_1_out[0][oc][oy0][ox0 + 1] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 + 1 < 56 && ox0 < 56) {
        float result = acc[cb][1][0] + conv3_1_b[oc];
        float fused_result = result;
        relu3_1_out[0][oc][oy0 + 1][ox0] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 + 1 < 56 && ox0 + 1 < 56) {
        float result = acc[cb][1][1] + conv3_1_b[oc];
        float fused_result = result;
        relu3_1_out[0][oc][oy0 + 1][ox0 + 1] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
    }
  }
}

/*LAUNCH 2 2 32 16 16*/
__global__ void ast_relu3_2_out_fused()
{
  __shared__ float sh2[2][34][35];
  __shared__ float shw2[2][8][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int oy0 = blockIdx.y * 32 + ty * 2;
  int ox0 = blockIdx.x * 32 + tx * 2;
  int ih0 = blockIdx.y * 32 - 1;
  int iw0 = blockIdx.x * 32 - 1;
  int c0 = 0;
  int c1 = 256;
  int oc0 = blockIdx.z * 8;
  float acc[8][2][2];
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1)
    for (int dy = 0; dy < 2; dy += 1)
      for (int dx = 0; dx < 2; dx += 1)
        acc[cb][dy][dx] = 0.0f;
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[0][sy][sx] = (iy < 0 || iy >= 56 || ix < 0 || ix >= 56)
                     ? 0.0f
                     : relu3_1_out[0][c0][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[0][cb][t] = (oc0 + cb < 256) ? conv3_2_w[oc0 + cb][c0][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
  __syncthreads();
  for (int c = c0; c < c1; c += 1) {
    int cur = (c - c0) & 1;
    if (c + 1 < c1) {
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[cur ^ 1][sy][sx] = (iy < 0 || iy >= 56 || ix < 0 || ix >= 56)
                     ? 0.0f
                     : relu3_1_out[0][(c + 1)][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[cur ^ 1][cb][t] = (oc0 + cb < 256) ? conv3_2_w[oc0 + cb][(c + 1)][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
    }
    float win[4][4];
    #pragma unroll
    for (int wy = 0; wy < 4; wy += 1)
      #pragma unroll
      for (int wx = 0; wx < 4; wx += 1)
        win[wy][wx] = sh2[cur][ty * 2 + wy][tx * 2 + wx];
    #pragma unroll
    for (int cb = 0; cb < 8; cb += 1) {
      if (oc0 + cb < 256) {
        #pragma unroll
        for (int k1 = 0; k1 < 3; k1 += 1)
          #pragma unroll
          for (int k2 = 0; k2 < 3; k2 += 1) {
            float wv = shw2[cur][cb][k1 * 3 + k2];
            acc[cb][0][0] += win[0 + k1][0 + k2] * wv;
            acc[cb][0][1] += win[0 + k1][1 + k2] * wv;
            acc[cb][1][0] += win[1 + k1][0 + k2] * wv;
            acc[cb][1][1] += win[1 + k1][1 + k2] * wv;
          }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 256) {
      if (oy0 < 56 && ox0 < 56) {
        float result = acc[cb][0][0] + conv3_2_b[oc];
        float fused_result = result;
        relu3_2_out[0][oc][oy0][ox0] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 < 56 && ox0 + 1 < 56) {
        float result = acc[cb][0][1] + conv3_2_b[oc];
        float fused_result = result;
        relu3_2_out[0][oc][oy0][ox0 + 1] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 + 1 < 56 && ox0 < 56) {
        float result = acc[cb][1][0] + conv3_2_b[oc];
        float fused_result = result;
        relu3_2_out[0][oc][oy0 + 1][ox0] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 + 1 < 56 && ox0 + 1 < 56) {
        float result = acc[cb][1][1] + conv3_2_b[oc];
        float fused_result = result;
        relu3_2_out[0][oc][oy0 + 1][ox0 + 1] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
    }
  }
}

/*LAUNCH 2 2 32 16 16*/
__global__ void ast_relu3_3_out_fused()
{
  __shared__ float sh2[2][34][35];
  __shared__ float shw2[2][8][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int oy0 = blockIdx.y * 32 + ty * 2;
  int ox0 = blockIdx.x * 32 + tx * 2;
  int ih0 = blockIdx.y * 32 - 1;
  int iw0 = blockIdx.x * 32 - 1;
  int c0 = 0;
  int c1 = 256;
  int oc0 = blockIdx.z * 8;
  float acc[8][2][2];
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1)
    for (int dy = 0; dy < 2; dy += 1)
      for (int dx = 0; dx < 2; dx += 1)
        acc[cb][dy][dx] = 0.0f;
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[0][sy][sx] = (iy < 0 || iy >= 56 || ix < 0 || ix >= 56)
                     ? 0.0f
                     : relu3_2_out[0][c0][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[0][cb][t] = (oc0 + cb < 256) ? conv3_3_w[oc0 + cb][c0][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
  __syncthreads();
  for (int c = c0; c < c1; c += 1) {
    int cur = (c - c0) & 1;
    if (c + 1 < c1) {
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[cur ^ 1][sy][sx] = (iy < 0 || iy >= 56 || ix < 0 || ix >= 56)
                     ? 0.0f
                     : relu3_2_out[0][(c + 1)][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[cur ^ 1][cb][t] = (oc0 + cb < 256) ? conv3_3_w[oc0 + cb][(c + 1)][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
    }
    float win[4][4];
    #pragma unroll
    for (int wy = 0; wy < 4; wy += 1)
      #pragma unroll
      for (int wx = 0; wx < 4; wx += 1)
        win[wy][wx] = sh2[cur][ty * 2 + wy][tx * 2 + wx];
    #pragma unroll
    for (int cb = 0; cb < 8; cb += 1) {
      if (oc0 + cb < 256) {
        #pragma unroll
        for (int k1 = 0; k1 < 3; k1 += 1)
          #pragma unroll
          for (int k2 = 0; k2 < 3; k2 += 1) {
            float wv = shw2[cur][cb][k1 * 3 + k2];
            acc[cb][0][0] += win[0 + k1][0 + k2] * wv;
            acc[cb][0][1] += win[0 + k1][1 + k2] * wv;
            acc[cb][1][0] += win[1 + k1][0 + k2] * wv;
            acc[cb][1][1] += win[1 + k1][1 + k2] * wv;
          }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 256) {
      if (oy0 < 56 && ox0 < 56) {
        float result = acc[cb][0][0] + conv3_3_b[oc];
        float fused_result = result;
        relu3_3_out[0][oc][oy0][ox0] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 < 56 && ox0 + 1 < 56) {
        float result = acc[cb][0][1] + conv3_3_b[oc];
        float fused_result = result;
        relu3_3_out[0][oc][oy0][ox0 + 1] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 + 1 < 56 && ox0 < 56) {
        float result = acc[cb][1][0] + conv3_3_b[oc];
        float fused_result = result;
        relu3_3_out[0][oc][oy0 + 1][ox0] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 + 1 < 56 && ox0 + 1 < 56) {
        float result = acc[cb][1][1] + conv3_3_b[oc];
        float fused_result = result;
        relu3_3_out[0][oc][oy0 + 1][ox0 + 1] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
    }
  }
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

/*LAUNCH 1 1 128 16 16*/
__global__ void ast_relu4_1_out_fused()
{
  __shared__ float sh2[2][34][35];
  __shared__ float shw2[2][8][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int oy0 = blockIdx.y * 32 + ty * 2;
  int ox0 = blockIdx.x * 32 + tx * 2;
  int ih0 = blockIdx.y * 32 - 1;
  int iw0 = blockIdx.x * 32 - 1;
  int c0 = 0;
  int c1 = 256;
  int sc = blockIdx.z / 64;
  int oc0 = (blockIdx.z - sc * 64) * 8;
  c0 = sc * 128;
  c1 = c0 + 128;
  if (c1 > 256) c1 = 256;
  float acc[8][2][2];
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1)
    for (int dy = 0; dy < 2; dy += 1)
      for (int dx = 0; dx < 2; dx += 1)
        acc[cb][dy][dx] = 0.0f;
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[0][sy][sx] = (iy < 0 || iy >= 28 || ix < 0 || ix >= 28)
                     ? 0.0f
                     : relu3_3_out[0][c0][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[0][cb][t] = (oc0 + cb < 512) ? conv4_1_w[oc0 + cb][c0][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
  __syncthreads();
  for (int c = c0; c < c1; c += 1) {
    int cur = (c - c0) & 1;
    if (c + 1 < c1) {
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[cur ^ 1][sy][sx] = (iy < 0 || iy >= 28 || ix < 0 || ix >= 28)
                     ? 0.0f
                     : relu3_3_out[0][(c + 1)][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[cur ^ 1][cb][t] = (oc0 + cb < 512) ? conv4_1_w[oc0 + cb][(c + 1)][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
    }
    float win[4][4];
    #pragma unroll
    for (int wy = 0; wy < 4; wy += 1)
      #pragma unroll
      for (int wx = 0; wx < 4; wx += 1)
        win[wy][wx] = sh2[cur][ty * 2 + wy][tx * 2 + wx];
    #pragma unroll
    for (int cb = 0; cb < 8; cb += 1) {
      if (oc0 + cb < 512) {
        #pragma unroll
        for (int k1 = 0; k1 < 3; k1 += 1)
          #pragma unroll
          for (int k2 = 0; k2 < 3; k2 += 1) {
            float wv = shw2[cur][cb][k1 * 3 + k2];
            acc[cb][0][0] += win[0 + k1][0 + k2] * wv;
            acc[cb][0][1] += win[0 + k1][1 + k2] * wv;
            acc[cb][1][0] += win[1 + k1][0 + k2] * wv;
            acc[cb][1][1] += win[1 + k1][1 + k2] * wv;
          }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 512) {
      if (oy0 < 28 && ox0 < 28) {
        relu4_1_out_ws[sc][oc][oy0][ox0] = acc[cb][0][0];
      }
      if (oy0 < 28 && ox0 + 1 < 28) {
        relu4_1_out_ws[sc][oc][oy0][ox0 + 1] = acc[cb][0][1];
      }
      if (oy0 + 1 < 28 && ox0 < 28) {
        relu4_1_out_ws[sc][oc][oy0 + 1][ox0] = acc[cb][1][0];
      }
      if (oy0 + 1 < 28 && ox0 + 1 < 28) {
        relu4_1_out_ws[sc][oc][oy0 + 1][ox0 + 1] = acc[cb][1][1];
      }
    }
  }
}

/*LAUNCH2 1568 1 1 256 1*/
__global__ void ast_relu4_1_out_fused_red()
{
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= 401408) return;
  int ow = idx % 28;
  int t0 = idx / 28;
  int oh = t0 % 28;
  int oc = t0 / 28;
  float acc = relu4_1_out_ws[0][oc][oh][ow];
  acc += relu4_1_out_ws[1][oc][oh][ow];
  float result = acc + conv4_1_b[oc];
  float fused_result = result;
  relu4_1_out[0][oc][oh][ow] = (fused_result <= 0.0f) ? 0.0f : fused_result;
}

/*LAUNCH 1 1 128 16 16*/
__global__ void ast_relu4_2_out_fused()
{
  __shared__ float sh2[2][34][35];
  __shared__ float shw2[2][8][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int oy0 = blockIdx.y * 32 + ty * 2;
  int ox0 = blockIdx.x * 32 + tx * 2;
  int ih0 = blockIdx.y * 32 - 1;
  int iw0 = blockIdx.x * 32 - 1;
  int c0 = 0;
  int c1 = 512;
  int sc = blockIdx.z / 64;
  int oc0 = (blockIdx.z - sc * 64) * 8;
  c0 = sc * 256;
  c1 = c0 + 256;
  if (c1 > 512) c1 = 512;
  float acc[8][2][2];
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1)
    for (int dy = 0; dy < 2; dy += 1)
      for (int dx = 0; dx < 2; dx += 1)
        acc[cb][dy][dx] = 0.0f;
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[0][sy][sx] = (iy < 0 || iy >= 28 || ix < 0 || ix >= 28)
                     ? 0.0f
                     : relu4_1_out[0][c0][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[0][cb][t] = (oc0 + cb < 512) ? conv4_2_w[oc0 + cb][c0][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
  __syncthreads();
  for (int c = c0; c < c1; c += 1) {
    int cur = (c - c0) & 1;
    if (c + 1 < c1) {
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[cur ^ 1][sy][sx] = (iy < 0 || iy >= 28 || ix < 0 || ix >= 28)
                     ? 0.0f
                     : relu4_1_out[0][(c + 1)][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[cur ^ 1][cb][t] = (oc0 + cb < 512) ? conv4_2_w[oc0 + cb][(c + 1)][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
    }
    float win[4][4];
    #pragma unroll
    for (int wy = 0; wy < 4; wy += 1)
      #pragma unroll
      for (int wx = 0; wx < 4; wx += 1)
        win[wy][wx] = sh2[cur][ty * 2 + wy][tx * 2 + wx];
    #pragma unroll
    for (int cb = 0; cb < 8; cb += 1) {
      if (oc0 + cb < 512) {
        #pragma unroll
        for (int k1 = 0; k1 < 3; k1 += 1)
          #pragma unroll
          for (int k2 = 0; k2 < 3; k2 += 1) {
            float wv = shw2[cur][cb][k1 * 3 + k2];
            acc[cb][0][0] += win[0 + k1][0 + k2] * wv;
            acc[cb][0][1] += win[0 + k1][1 + k2] * wv;
            acc[cb][1][0] += win[1 + k1][0 + k2] * wv;
            acc[cb][1][1] += win[1 + k1][1 + k2] * wv;
          }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 512) {
      if (oy0 < 28 && ox0 < 28) {
        relu4_2_out_ws[sc][oc][oy0][ox0] = acc[cb][0][0];
      }
      if (oy0 < 28 && ox0 + 1 < 28) {
        relu4_2_out_ws[sc][oc][oy0][ox0 + 1] = acc[cb][0][1];
      }
      if (oy0 + 1 < 28 && ox0 < 28) {
        relu4_2_out_ws[sc][oc][oy0 + 1][ox0] = acc[cb][1][0];
      }
      if (oy0 + 1 < 28 && ox0 + 1 < 28) {
        relu4_2_out_ws[sc][oc][oy0 + 1][ox0 + 1] = acc[cb][1][1];
      }
    }
  }
}

/*LAUNCH2 1568 1 1 256 1*/
__global__ void ast_relu4_2_out_fused_red()
{
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= 401408) return;
  int ow = idx % 28;
  int t0 = idx / 28;
  int oh = t0 % 28;
  int oc = t0 / 28;
  float acc = relu4_2_out_ws[0][oc][oh][ow];
  acc += relu4_2_out_ws[1][oc][oh][ow];
  float result = acc + conv4_2_b[oc];
  float fused_result = result;
  relu4_2_out[0][oc][oh][ow] = (fused_result <= 0.0f) ? 0.0f : fused_result;
}

/*LAUNCH 1 1 128 16 16*/
__global__ void ast_relu4_3_out_fused()
{
  __shared__ float sh2[2][34][35];
  __shared__ float shw2[2][8][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int oy0 = blockIdx.y * 32 + ty * 2;
  int ox0 = blockIdx.x * 32 + tx * 2;
  int ih0 = blockIdx.y * 32 - 1;
  int iw0 = blockIdx.x * 32 - 1;
  int c0 = 0;
  int c1 = 512;
  int sc = blockIdx.z / 64;
  int oc0 = (blockIdx.z - sc * 64) * 8;
  c0 = sc * 256;
  c1 = c0 + 256;
  if (c1 > 512) c1 = 512;
  float acc[8][2][2];
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1)
    for (int dy = 0; dy < 2; dy += 1)
      for (int dx = 0; dx < 2; dx += 1)
        acc[cb][dy][dx] = 0.0f;
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[0][sy][sx] = (iy < 0 || iy >= 28 || ix < 0 || ix >= 28)
                     ? 0.0f
                     : relu4_2_out[0][c0][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[0][cb][t] = (oc0 + cb < 512) ? conv4_3_w[oc0 + cb][c0][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
  __syncthreads();
  for (int c = c0; c < c1; c += 1) {
    int cur = (c - c0) & 1;
    if (c + 1 < c1) {
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[cur ^ 1][sy][sx] = (iy < 0 || iy >= 28 || ix < 0 || ix >= 28)
                     ? 0.0f
                     : relu4_2_out[0][(c + 1)][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 72) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[cur ^ 1][cb][t] = (oc0 + cb < 512) ? conv4_3_w[oc0 + cb][(c + 1)][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
    }
    float win[4][4];
    #pragma unroll
    for (int wy = 0; wy < 4; wy += 1)
      #pragma unroll
      for (int wx = 0; wx < 4; wx += 1)
        win[wy][wx] = sh2[cur][ty * 2 + wy][tx * 2 + wx];
    #pragma unroll
    for (int cb = 0; cb < 8; cb += 1) {
      if (oc0 + cb < 512) {
        #pragma unroll
        for (int k1 = 0; k1 < 3; k1 += 1)
          #pragma unroll
          for (int k2 = 0; k2 < 3; k2 += 1) {
            float wv = shw2[cur][cb][k1 * 3 + k2];
            acc[cb][0][0] += win[0 + k1][0 + k2] * wv;
            acc[cb][0][1] += win[0 + k1][1 + k2] * wv;
            acc[cb][1][0] += win[1 + k1][0 + k2] * wv;
            acc[cb][1][1] += win[1 + k1][1 + k2] * wv;
          }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int cb = 0; cb < 8; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 512) {
      if (oy0 < 28 && ox0 < 28) {
        relu4_3_out_ws[sc][oc][oy0][ox0] = acc[cb][0][0];
      }
      if (oy0 < 28 && ox0 + 1 < 28) {
        relu4_3_out_ws[sc][oc][oy0][ox0 + 1] = acc[cb][0][1];
      }
      if (oy0 + 1 < 28 && ox0 < 28) {
        relu4_3_out_ws[sc][oc][oy0 + 1][ox0] = acc[cb][1][0];
      }
      if (oy0 + 1 < 28 && ox0 + 1 < 28) {
        relu4_3_out_ws[sc][oc][oy0 + 1][ox0 + 1] = acc[cb][1][1];
      }
    }
  }
}

/*LAUNCH2 1568 1 1 256 1*/
__global__ void ast_relu4_3_out_fused_red()
{
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= 401408) return;
  int ow = idx % 28;
  int t0 = idx / 28;
  int oh = t0 % 28;
  int oc = t0 / 28;
  float acc = relu4_3_out_ws[0][oc][oh][ow];
  acc += relu4_3_out_ws[1][oc][oh][ow];
  float result = acc + conv4_3_b[oc];
  float fused_result = result;
  relu4_3_out[0][oc][oh][ow] = (fused_result <= 0.0f) ? 0.0f : fused_result;
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

/*LAUNCH 1 1 512 16 16*/
__global__ void ast_relu5_1_out_fused()
{
  __shared__ float sh[18][18];
  __shared__ float shw[4][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ow = blockIdx.x * 16 + tx;
  int oh = blockIdx.y * 16 + ty;
  int iw0 = blockIdx.x * 16 - 1;
  int ih0 = blockIdx.y * 16 - 1;
  int sc = blockIdx.z / 128;
  int oc0 = (blockIdx.z - sc * 128) * 4;
  int c0 = sc * 128;
  int c1 = c0 + 128;
  if (c1 > 512) c1 = 512;
  float acc[4];
  #pragma unroll
  for (int cb = 0; cb < 4; cb += 1) acc[cb] = 0.0f;
  for (int c = c0; c < c1; c += 1) {
    for (int sy = ty; sy < 18; sy += 16)
      for (int sx = tx; sx < 18; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh[sy][sx] = (iy < 0 || iy >= 14 || ix < 0 || ix >= 14)
                     ? 0.0f
                     : relu4_3_out[0][c][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 36) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw[cb][t] = (oc0 + cb < 512) ? conv5_1_w[oc0 + cb][c][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
    __syncthreads();
    int by0 = ty * 1;
    int bx0 = tx * 1;
    #pragma unroll
    for (int cb = 0; cb < 4; cb += 1) {
      if (oc0 + cb < 512) {
        #pragma unroll
        for (int k1 = 0; k1 < 3; k1 += 1)
          #pragma unroll
          for (int k2 = 0; k2 < 3; k2 += 1)
            acc[cb] += sh[by0 + k1][bx0 + k2] * shw[cb][k1 * 3 + k2];
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int cb = 0; cb < 4; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 512 && oh < 14 && ow < 14)
      relu5_1_out_ws[sc][oc][oh][ow] = acc[cb];
  }
}

/*LAUNCH2 392 1 1 256 1*/
__global__ void ast_relu5_1_out_fused_red()
{
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= 100352) return;
  int ow = idx % 14;
  int t0 = idx / 14;
  int oh = t0 % 14;
  int oc = t0 / 14;
  float acc = relu5_1_out_ws[0][oc][oh][ow];
  acc += relu5_1_out_ws[1][oc][oh][ow];
  acc += relu5_1_out_ws[2][oc][oh][ow];
  acc += relu5_1_out_ws[3][oc][oh][ow];
  float result = acc + conv5_1_b[oc];
  float fused_result = result;
  relu5_1_out[0][oc][oh][ow] = (fused_result <= 0.0f) ? 0.0f : fused_result;
}

/*LAUNCH 1 1 512 16 16*/
__global__ void ast_relu5_2_out_fused()
{
  __shared__ float sh[18][18];
  __shared__ float shw[4][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ow = blockIdx.x * 16 + tx;
  int oh = blockIdx.y * 16 + ty;
  int iw0 = blockIdx.x * 16 - 1;
  int ih0 = blockIdx.y * 16 - 1;
  int sc = blockIdx.z / 128;
  int oc0 = (blockIdx.z - sc * 128) * 4;
  int c0 = sc * 128;
  int c1 = c0 + 128;
  if (c1 > 512) c1 = 512;
  float acc[4];
  #pragma unroll
  for (int cb = 0; cb < 4; cb += 1) acc[cb] = 0.0f;
  for (int c = c0; c < c1; c += 1) {
    for (int sy = ty; sy < 18; sy += 16)
      for (int sx = tx; sx < 18; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh[sy][sx] = (iy < 0 || iy >= 14 || ix < 0 || ix >= 14)
                     ? 0.0f
                     : relu5_1_out[0][c][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 36) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw[cb][t] = (oc0 + cb < 512) ? conv5_2_w[oc0 + cb][c][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
    __syncthreads();
    int by0 = ty * 1;
    int bx0 = tx * 1;
    #pragma unroll
    for (int cb = 0; cb < 4; cb += 1) {
      if (oc0 + cb < 512) {
        #pragma unroll
        for (int k1 = 0; k1 < 3; k1 += 1)
          #pragma unroll
          for (int k2 = 0; k2 < 3; k2 += 1)
            acc[cb] += sh[by0 + k1][bx0 + k2] * shw[cb][k1 * 3 + k2];
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int cb = 0; cb < 4; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 512 && oh < 14 && ow < 14)
      relu5_2_out_ws[sc][oc][oh][ow] = acc[cb];
  }
}

/*LAUNCH2 392 1 1 256 1*/
__global__ void ast_relu5_2_out_fused_red()
{
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= 100352) return;
  int ow = idx % 14;
  int t0 = idx / 14;
  int oh = t0 % 14;
  int oc = t0 / 14;
  float acc = relu5_2_out_ws[0][oc][oh][ow];
  acc += relu5_2_out_ws[1][oc][oh][ow];
  acc += relu5_2_out_ws[2][oc][oh][ow];
  acc += relu5_2_out_ws[3][oc][oh][ow];
  float result = acc + conv5_2_b[oc];
  float fused_result = result;
  relu5_2_out[0][oc][oh][ow] = (fused_result <= 0.0f) ? 0.0f : fused_result;
}

/*LAUNCH 1 1 512 16 16*/
__global__ void ast_relu5_3_out_fused()
{
  __shared__ float sh[18][18];
  __shared__ float shw[4][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ow = blockIdx.x * 16 + tx;
  int oh = blockIdx.y * 16 + ty;
  int iw0 = blockIdx.x * 16 - 1;
  int ih0 = blockIdx.y * 16 - 1;
  int sc = blockIdx.z / 128;
  int oc0 = (blockIdx.z - sc * 128) * 4;
  int c0 = sc * 128;
  int c1 = c0 + 128;
  if (c1 > 512) c1 = 512;
  float acc[4];
  #pragma unroll
  for (int cb = 0; cb < 4; cb += 1) acc[cb] = 0.0f;
  for (int c = c0; c < c1; c += 1) {
    for (int sy = ty; sy < 18; sy += 16)
      for (int sx = tx; sx < 18; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh[sy][sx] = (iy < 0 || iy >= 14 || ix < 0 || ix >= 14)
                     ? 0.0f
                     : relu5_2_out[0][c][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 36) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw[cb][t] = (oc0 + cb < 512) ? conv5_3_w[oc0 + cb][c][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
    __syncthreads();
    int by0 = ty * 1;
    int bx0 = tx * 1;
    #pragma unroll
    for (int cb = 0; cb < 4; cb += 1) {
      if (oc0 + cb < 512) {
        #pragma unroll
        for (int k1 = 0; k1 < 3; k1 += 1)
          #pragma unroll
          for (int k2 = 0; k2 < 3; k2 += 1)
            acc[cb] += sh[by0 + k1][bx0 + k2] * shw[cb][k1 * 3 + k2];
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int cb = 0; cb < 4; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 512 && oh < 14 && ow < 14)
      relu5_3_out_ws[sc][oc][oh][ow] = acc[cb];
  }
}

/*LAUNCH2 392 1 1 256 1*/
__global__ void ast_relu5_3_out_fused_red()
{
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= 100352) return;
  int ow = idx % 14;
  int t0 = idx / 14;
  int oh = t0 % 14;
  int oc = t0 / 14;
  float acc = relu5_3_out_ws[0][oc][oh][ow];
  acc += relu5_3_out_ws[1][oc][oh][ow];
  acc += relu5_3_out_ws[2][oc][oh][ow];
  acc += relu5_3_out_ws[3][oc][oh][ow];
  float result = acc + conv5_3_b[oc];
  float fused_result = result;
  relu5_3_out[0][oc][oh][ow] = (fused_result <= 0.0f) ? 0.0f : fused_result;
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

/*LAUNCH 4096 1 1 256 1*/
__global__ void ast_fc1_relu_out_fused()
{
  __shared__ float red[256];
  int o = blockIdx.x;
  int tid = threadIdx.x;
  float acc = 0.0f;
  for (int i = tid; i < 25088; i += 256)
    acc += flatten_out[0][i] * fc1_w[o][i];
  red[tid] = acc;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid == 0) {
    float result = red[0] + fc1_b[o];
    fc1_relu_out[0][o] = (result <= 0.0f) ? 0.0f : result;
  }
}

/*LAUNCH 4096 1 1 256 1*/
__global__ void ast_fc2_relu_out_fused()
{
  __shared__ float red[256];
  int o = blockIdx.x;
  int tid = threadIdx.x;
  float acc = 0.0f;
  for (int i = tid; i < 4096; i += 256)
    acc += fc1_relu_out[0][i] * fc2_w[o][i];
  red[tid] = acc;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid == 0) {
    float result = red[0] + fc2_b[o];
    fc2_relu_out[0][o] = (result <= 0.0f) ? 0.0f : result;
  }
}

/*LAUNCH 1000 1 1 256 1*/
__global__ void ast_output()
{
  __shared__ float red[256];
  int o = blockIdx.x;
  int tid = threadIdx.x;
  float acc = 0.0f;
  for (int i = tid; i < 4096; i += 256)
    acc += fc2_relu_out[0][i] * fc3_w[o][i];
  red[tid] = acc;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid == 0) {
    float result = red[0] + fc3_b[o];
    output[0][o] = result;
  }
}

void vgg16p_tiled()
{
  { dim3 g(7,7,8); dim3 b(16,16,1); ast_relu1_1_out_fused<<<g, b>>>(); }
  { dim3 g(7,7,8); dim3 b(16,16,1); ast_relu1_2_out_fused<<<g, b>>>(); }
  ast_pool1_out<<<3136, 256>>>();
  { dim3 g(4,4,16); dim3 b(16,16,1); ast_relu2_1_out_fused<<<g, b>>>(); }
  { dim3 g(4,4,16); dim3 b(16,16,1); ast_relu2_2_out_fused<<<g, b>>>(); }
  ast_pool2_out<<<1568, 256>>>();
  { dim3 g(2,2,32); dim3 b(16,16,1); ast_relu3_1_out_fused<<<g, b>>>(); }
  { dim3 g(2,2,32); dim3 b(16,16,1); ast_relu3_2_out_fused<<<g, b>>>(); }
  { dim3 g(2,2,32); dim3 b(16,16,1); ast_relu3_3_out_fused<<<g, b>>>(); }
  ast_pool3_out<<<784, 256>>>();
  { dim3 g(1,1,128); dim3 b(16,16,1); ast_relu4_1_out_fused<<<g, b>>>(); }
  { dim3 g(1568,1,1); dim3 b(256,1,1); ast_relu4_1_out_fused_red<<<g, b>>>(); }
  { dim3 g(1,1,128); dim3 b(16,16,1); ast_relu4_2_out_fused<<<g, b>>>(); }
  { dim3 g(1568,1,1); dim3 b(256,1,1); ast_relu4_2_out_fused_red<<<g, b>>>(); }
  { dim3 g(1,1,128); dim3 b(16,16,1); ast_relu4_3_out_fused<<<g, b>>>(); }
  { dim3 g(1568,1,1); dim3 b(256,1,1); ast_relu4_3_out_fused_red<<<g, b>>>(); }
  ast_pool4_out<<<392, 256>>>();
  { dim3 g(1,1,512); dim3 b(16,16,1); ast_relu5_1_out_fused<<<g, b>>>(); }
  { dim3 g(392,1,1); dim3 b(256,1,1); ast_relu5_1_out_fused_red<<<g, b>>>(); }
  { dim3 g(1,1,512); dim3 b(16,16,1); ast_relu5_2_out_fused<<<g, b>>>(); }
  { dim3 g(392,1,1); dim3 b(256,1,1); ast_relu5_2_out_fused_red<<<g, b>>>(); }
  { dim3 g(1,1,512); dim3 b(16,16,1); ast_relu5_3_out_fused<<<g, b>>>(); }
  { dim3 g(392,1,1); dim3 b(256,1,1); ast_relu5_3_out_fused_red<<<g, b>>>(); }
  ast_pool5_out<<<98, 256>>>();
  ast_flatten_out<<<98, 256>>>();
  { dim3 g(4096,1,1); dim3 b(256,1,1); ast_fc1_relu_out_fused<<<g, b>>>(); }
  { dim3 g(4096,1,1); dim3 b(256,1,1); ast_fc2_relu_out_fused<<<g, b>>>(); }
  { dim3 g(1000,1,1); dim3 b(256,1,1); ast_output<<<g, b>>>(); }
  cudaDeviceSynchronize();
}
void vgg16p_tiled_kt()
{
  cudaEvent_t eva, evb;
  cudaEventCreate(&eva);
  cudaEventCreate(&evb);
  float kt[22];
  cudaEventRecord(eva);
  { dim3 g(7,7,8); dim3 b(16,16,1); ast_relu1_1_out_fused<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[0], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(7,7,8); dim3 b(16,16,1); ast_relu1_2_out_fused<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[1], eva, evb);
  cudaEventRecord(eva);
  ast_pool1_out<<<3136, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[2], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(4,4,16); dim3 b(16,16,1); ast_relu2_1_out_fused<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[3], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(4,4,16); dim3 b(16,16,1); ast_relu2_2_out_fused<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[4], eva, evb);
  cudaEventRecord(eva);
  ast_pool2_out<<<1568, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[5], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(2,2,32); dim3 b(16,16,1); ast_relu3_1_out_fused<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[6], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(2,2,32); dim3 b(16,16,1); ast_relu3_2_out_fused<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[7], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(2,2,32); dim3 b(16,16,1); ast_relu3_3_out_fused<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[8], eva, evb);
  cudaEventRecord(eva);
  ast_pool3_out<<<784, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[9], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(1,1,128); dim3 b(16,16,1); ast_relu4_1_out_fused<<<g, b>>>(); }
  { dim3 g(1568,1,1); dim3 b(256,1,1); ast_relu4_1_out_fused_red<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[10], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(1,1,128); dim3 b(16,16,1); ast_relu4_2_out_fused<<<g, b>>>(); }
  { dim3 g(1568,1,1); dim3 b(256,1,1); ast_relu4_2_out_fused_red<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[11], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(1,1,128); dim3 b(16,16,1); ast_relu4_3_out_fused<<<g, b>>>(); }
  { dim3 g(1568,1,1); dim3 b(256,1,1); ast_relu4_3_out_fused_red<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[12], eva, evb);
  cudaEventRecord(eva);
  ast_pool4_out<<<392, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[13], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(1,1,512); dim3 b(16,16,1); ast_relu5_1_out_fused<<<g, b>>>(); }
  { dim3 g(392,1,1); dim3 b(256,1,1); ast_relu5_1_out_fused_red<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[14], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(1,1,512); dim3 b(16,16,1); ast_relu5_2_out_fused<<<g, b>>>(); }
  { dim3 g(392,1,1); dim3 b(256,1,1); ast_relu5_2_out_fused_red<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[15], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(1,1,512); dim3 b(16,16,1); ast_relu5_3_out_fused<<<g, b>>>(); }
  { dim3 g(392,1,1); dim3 b(256,1,1); ast_relu5_3_out_fused_red<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[16], eva, evb);
  cudaEventRecord(eva);
  ast_pool5_out<<<98, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[17], eva, evb);
  cudaEventRecord(eva);
  ast_flatten_out<<<98, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[18], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(4096,1,1); dim3 b(256,1,1); ast_fc1_relu_out_fused<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[19], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(4096,1,1); dim3 b(256,1,1); ast_fc2_relu_out_fused<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[20], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(1000,1,1); dim3 b(256,1,1); ast_output<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[21], eva, evb);
  printf("KTIME 0 ast_relu1_1_out_fused %.6f\n", kt[0]);
  printf("KTIME 1 ast_relu1_2_out_fused %.6f\n", kt[1]);
  printf("KTIME 2 ast_pool1_out %.6f\n", kt[2]);
  printf("KTIME 3 ast_relu2_1_out_fused %.6f\n", kt[3]);
  printf("KTIME 4 ast_relu2_2_out_fused %.6f\n", kt[4]);
  printf("KTIME 5 ast_pool2_out %.6f\n", kt[5]);
  printf("KTIME 6 ast_relu3_1_out_fused %.6f\n", kt[6]);
  printf("KTIME 7 ast_relu3_2_out_fused %.6f\n", kt[7]);
  printf("KTIME 8 ast_relu3_3_out_fused %.6f\n", kt[8]);
  printf("KTIME 9 ast_pool3_out %.6f\n", kt[9]);
  printf("KTIME 10 ast_relu4_1_out_fused %.6f\n", kt[10]);
  printf("KTIME 11 ast_relu4_2_out_fused %.6f\n", kt[11]);
  printf("KTIME 12 ast_relu4_3_out_fused %.6f\n", kt[12]);
  printf("KTIME 13 ast_pool4_out %.6f\n", kt[13]);
  printf("KTIME 14 ast_relu5_1_out_fused %.6f\n", kt[14]);
  printf("KTIME 15 ast_relu5_2_out_fused %.6f\n", kt[15]);
  printf("KTIME 16 ast_relu5_3_out_fused %.6f\n", kt[16]);
  printf("KTIME 17 ast_pool5_out %.6f\n", kt[17]);
  printf("KTIME 18 ast_flatten_out %.6f\n", kt[18]);
  printf("KTIME 19 ast_fc1_relu_out_fused %.6f\n", kt[19]);
  printf("KTIME 20 ast_fc2_relu_out_fused %.6f\n", kt[20]);
  printf("KTIME 21 ast_output %.6f\n", kt[21]);
  cudaEventDestroy(eva);
  cudaEventDestroy(evb);
}


int main(int argc, char** argv)
{
  size_t input_bytes = 602112;
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
  size_t fc1_w_bytes = 411041792;
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
  size_t relu1_1_out_bytes = 12845056;
  float* d_relu1_1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu1_1_out, relu1_1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu1_1_out, &d_relu1_1_out, sizeof(d_relu1_1_out)));
  size_t relu1_2_out_bytes = 12845056;
  float* d_relu1_2_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu1_2_out, relu1_2_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu1_2_out, &d_relu1_2_out, sizeof(d_relu1_2_out)));
  size_t pool1_out_bytes = 3211264;
  float* d_pool1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_pool1_out, pool1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(pool1_out, &d_pool1_out, sizeof(d_pool1_out)));
  size_t relu2_1_out_bytes = 6422528;
  float* d_relu2_1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu2_1_out, relu2_1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu2_1_out, &d_relu2_1_out, sizeof(d_relu2_1_out)));
  size_t relu2_2_out_bytes = 6422528;
  float* d_relu2_2_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu2_2_out, relu2_2_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu2_2_out, &d_relu2_2_out, sizeof(d_relu2_2_out)));
  size_t pool2_out_bytes = 1605632;
  float* d_pool2_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_pool2_out, pool2_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(pool2_out, &d_pool2_out, sizeof(d_pool2_out)));
  size_t relu3_1_out_bytes = 3211264;
  float* d_relu3_1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu3_1_out, relu3_1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu3_1_out, &d_relu3_1_out, sizeof(d_relu3_1_out)));
  size_t relu3_2_out_bytes = 3211264;
  float* d_relu3_2_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu3_2_out, relu3_2_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu3_2_out, &d_relu3_2_out, sizeof(d_relu3_2_out)));
  size_t relu3_3_out_bytes = 3211264;
  float* d_relu3_3_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu3_3_out, relu3_3_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu3_3_out, &d_relu3_3_out, sizeof(d_relu3_3_out)));
  size_t pool3_out_bytes = 802816;
  float* d_pool3_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_pool3_out, pool3_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(pool3_out, &d_pool3_out, sizeof(d_pool3_out)));
  size_t relu4_1_out_bytes = 1605632;
  float* d_relu4_1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu4_1_out, relu4_1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu4_1_out, &d_relu4_1_out, sizeof(d_relu4_1_out)));
  size_t relu4_2_out_bytes = 1605632;
  float* d_relu4_2_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu4_2_out, relu4_2_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu4_2_out, &d_relu4_2_out, sizeof(d_relu4_2_out)));
  size_t relu4_3_out_bytes = 1605632;
  float* d_relu4_3_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu4_3_out, relu4_3_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu4_3_out, &d_relu4_3_out, sizeof(d_relu4_3_out)));
  size_t pool4_out_bytes = 401408;
  float* d_pool4_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_pool4_out, pool4_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(pool4_out, &d_pool4_out, sizeof(d_pool4_out)));
  size_t relu5_1_out_bytes = 401408;
  float* d_relu5_1_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu5_1_out, relu5_1_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu5_1_out, &d_relu5_1_out, sizeof(d_relu5_1_out)));
  size_t relu5_2_out_bytes = 401408;
  float* d_relu5_2_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu5_2_out, relu5_2_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu5_2_out, &d_relu5_2_out, sizeof(d_relu5_2_out)));
  size_t relu5_3_out_bytes = 401408;
  float* d_relu5_3_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu5_3_out, relu5_3_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu5_3_out, &d_relu5_3_out, sizeof(d_relu5_3_out)));
  size_t pool5_out_bytes = 100352;
  float* d_pool5_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_pool5_out, pool5_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(pool5_out, &d_pool5_out, sizeof(d_pool5_out)));
  size_t flatten_out_bytes = 100352;
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
  size_t relu4_1_out_ws_bytes = 3211264;
  float* d_relu4_1_out_ws = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu4_1_out_ws, relu4_1_out_ws_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu4_1_out_ws, &d_relu4_1_out_ws, sizeof(d_relu4_1_out_ws)));
  size_t relu4_2_out_ws_bytes = 3211264;
  float* d_relu4_2_out_ws = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu4_2_out_ws, relu4_2_out_ws_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu4_2_out_ws, &d_relu4_2_out_ws, sizeof(d_relu4_2_out_ws)));
  size_t relu4_3_out_ws_bytes = 3211264;
  float* d_relu4_3_out_ws = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu4_3_out_ws, relu4_3_out_ws_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu4_3_out_ws, &d_relu4_3_out_ws, sizeof(d_relu4_3_out_ws)));
  size_t relu5_1_out_ws_bytes = 1605632;
  float* d_relu5_1_out_ws = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu5_1_out_ws, relu5_1_out_ws_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu5_1_out_ws, &d_relu5_1_out_ws, sizeof(d_relu5_1_out_ws)));
  size_t relu5_2_out_ws_bytes = 1605632;
  float* d_relu5_2_out_ws = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu5_2_out_ws, relu5_2_out_ws_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu5_2_out_ws, &d_relu5_2_out_ws, sizeof(d_relu5_2_out_ws)));
  size_t relu5_3_out_ws_bytes = 1605632;
  float* d_relu5_3_out_ws = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_relu5_3_out_ws, relu5_3_out_ws_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(relu5_3_out_ws, &d_relu5_3_out_ws, sizeof(d_relu5_3_out_ws)));

  cudaEvent_t __io_start, __io_stop;
  CUDA_CHECK(cudaEventCreate(&__io_start));
  CUDA_CHECK(cudaEventCreate(&__io_stop));
  CUDA_CHECK(cudaEventRecord(__io_start));
  float* h_input = (float*)malloc(602112);
  if (!h_input) { fprintf(stderr, "malloc failed for input\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_input.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_input.bin\n"); return 1; }
    size_t nread = fread(h_input, sizeof(float), 150528, f);
    fclose(f);
    if (nread != 150528) { fprintf(stderr, "Failed to read vgg16p_tiled_input.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_input, h_input, 602112, cudaMemcpyHostToDevice));
  float* h_conv1_1_w = (float*)malloc(6912);
  if (!h_conv1_1_w) { fprintf(stderr, "malloc failed for conv1_1_w\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv1_1_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv1_1_w.bin\n"); return 1; }
    size_t nread = fread(h_conv1_1_w, sizeof(float), 1728, f);
    fclose(f);
    if (nread != 1728) { fprintf(stderr, "Failed to read vgg16p_tiled_conv1_1_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv1_1_w, h_conv1_1_w, 6912, cudaMemcpyHostToDevice));
  float* h_conv1_1_b = (float*)malloc(256);
  if (!h_conv1_1_b) { fprintf(stderr, "malloc failed for conv1_1_b\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv1_1_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv1_1_b.bin\n"); return 1; }
    size_t nread = fread(h_conv1_1_b, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read vgg16p_tiled_conv1_1_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv1_1_b, h_conv1_1_b, 256, cudaMemcpyHostToDevice));
  float* h_conv1_2_w = (float*)malloc(147456);
  if (!h_conv1_2_w) { fprintf(stderr, "malloc failed for conv1_2_w\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv1_2_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv1_2_w.bin\n"); return 1; }
    size_t nread = fread(h_conv1_2_w, sizeof(float), 36864, f);
    fclose(f);
    if (nread != 36864) { fprintf(stderr, "Failed to read vgg16p_tiled_conv1_2_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv1_2_w, h_conv1_2_w, 147456, cudaMemcpyHostToDevice));
  float* h_conv1_2_b = (float*)malloc(256);
  if (!h_conv1_2_b) { fprintf(stderr, "malloc failed for conv1_2_b\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv1_2_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv1_2_b.bin\n"); return 1; }
    size_t nread = fread(h_conv1_2_b, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read vgg16p_tiled_conv1_2_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv1_2_b, h_conv1_2_b, 256, cudaMemcpyHostToDevice));
  float* h_conv2_1_w = (float*)malloc(294912);
  if (!h_conv2_1_w) { fprintf(stderr, "malloc failed for conv2_1_w\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv2_1_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv2_1_w.bin\n"); return 1; }
    size_t nread = fread(h_conv2_1_w, sizeof(float), 73728, f);
    fclose(f);
    if (nread != 73728) { fprintf(stderr, "Failed to read vgg16p_tiled_conv2_1_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv2_1_w, h_conv2_1_w, 294912, cudaMemcpyHostToDevice));
  float* h_conv2_1_b = (float*)malloc(512);
  if (!h_conv2_1_b) { fprintf(stderr, "malloc failed for conv2_1_b\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv2_1_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv2_1_b.bin\n"); return 1; }
    size_t nread = fread(h_conv2_1_b, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read vgg16p_tiled_conv2_1_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv2_1_b, h_conv2_1_b, 512, cudaMemcpyHostToDevice));
  float* h_conv2_2_w = (float*)malloc(589824);
  if (!h_conv2_2_w) { fprintf(stderr, "malloc failed for conv2_2_w\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv2_2_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv2_2_w.bin\n"); return 1; }
    size_t nread = fread(h_conv2_2_w, sizeof(float), 147456, f);
    fclose(f);
    if (nread != 147456) { fprintf(stderr, "Failed to read vgg16p_tiled_conv2_2_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv2_2_w, h_conv2_2_w, 589824, cudaMemcpyHostToDevice));
  float* h_conv2_2_b = (float*)malloc(512);
  if (!h_conv2_2_b) { fprintf(stderr, "malloc failed for conv2_2_b\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv2_2_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv2_2_b.bin\n"); return 1; }
    size_t nread = fread(h_conv2_2_b, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read vgg16p_tiled_conv2_2_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv2_2_b, h_conv2_2_b, 512, cudaMemcpyHostToDevice));
  float* h_conv3_1_w = (float*)malloc(1179648);
  if (!h_conv3_1_w) { fprintf(stderr, "malloc failed for conv3_1_w\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv3_1_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv3_1_w.bin\n"); return 1; }
    size_t nread = fread(h_conv3_1_w, sizeof(float), 294912, f);
    fclose(f);
    if (nread != 294912) { fprintf(stderr, "Failed to read vgg16p_tiled_conv3_1_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv3_1_w, h_conv3_1_w, 1179648, cudaMemcpyHostToDevice));
  float* h_conv3_1_b = (float*)malloc(1024);
  if (!h_conv3_1_b) { fprintf(stderr, "malloc failed for conv3_1_b\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv3_1_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv3_1_b.bin\n"); return 1; }
    size_t nread = fread(h_conv3_1_b, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read vgg16p_tiled_conv3_1_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv3_1_b, h_conv3_1_b, 1024, cudaMemcpyHostToDevice));
  float* h_conv3_2_w = (float*)malloc(2359296);
  if (!h_conv3_2_w) { fprintf(stderr, "malloc failed for conv3_2_w\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv3_2_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv3_2_w.bin\n"); return 1; }
    size_t nread = fread(h_conv3_2_w, sizeof(float), 589824, f);
    fclose(f);
    if (nread != 589824) { fprintf(stderr, "Failed to read vgg16p_tiled_conv3_2_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv3_2_w, h_conv3_2_w, 2359296, cudaMemcpyHostToDevice));
  float* h_conv3_2_b = (float*)malloc(1024);
  if (!h_conv3_2_b) { fprintf(stderr, "malloc failed for conv3_2_b\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv3_2_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv3_2_b.bin\n"); return 1; }
    size_t nread = fread(h_conv3_2_b, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read vgg16p_tiled_conv3_2_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv3_2_b, h_conv3_2_b, 1024, cudaMemcpyHostToDevice));
  float* h_conv3_3_w = (float*)malloc(2359296);
  if (!h_conv3_3_w) { fprintf(stderr, "malloc failed for conv3_3_w\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv3_3_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv3_3_w.bin\n"); return 1; }
    size_t nread = fread(h_conv3_3_w, sizeof(float), 589824, f);
    fclose(f);
    if (nread != 589824) { fprintf(stderr, "Failed to read vgg16p_tiled_conv3_3_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv3_3_w, h_conv3_3_w, 2359296, cudaMemcpyHostToDevice));
  float* h_conv3_3_b = (float*)malloc(1024);
  if (!h_conv3_3_b) { fprintf(stderr, "malloc failed for conv3_3_b\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv3_3_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv3_3_b.bin\n"); return 1; }
    size_t nread = fread(h_conv3_3_b, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read vgg16p_tiled_conv3_3_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv3_3_b, h_conv3_3_b, 1024, cudaMemcpyHostToDevice));
  float* h_conv4_1_w = (float*)malloc(4718592);
  if (!h_conv4_1_w) { fprintf(stderr, "malloc failed for conv4_1_w\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv4_1_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv4_1_w.bin\n"); return 1; }
    size_t nread = fread(h_conv4_1_w, sizeof(float), 1179648, f);
    fclose(f);
    if (nread != 1179648) { fprintf(stderr, "Failed to read vgg16p_tiled_conv4_1_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv4_1_w, h_conv4_1_w, 4718592, cudaMemcpyHostToDevice));
  float* h_conv4_1_b = (float*)malloc(2048);
  if (!h_conv4_1_b) { fprintf(stderr, "malloc failed for conv4_1_b\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv4_1_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv4_1_b.bin\n"); return 1; }
    size_t nread = fread(h_conv4_1_b, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read vgg16p_tiled_conv4_1_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv4_1_b, h_conv4_1_b, 2048, cudaMemcpyHostToDevice));
  float* h_conv4_2_w = (float*)malloc(9437184);
  if (!h_conv4_2_w) { fprintf(stderr, "malloc failed for conv4_2_w\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv4_2_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv4_2_w.bin\n"); return 1; }
    size_t nread = fread(h_conv4_2_w, sizeof(float), 2359296, f);
    fclose(f);
    if (nread != 2359296) { fprintf(stderr, "Failed to read vgg16p_tiled_conv4_2_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv4_2_w, h_conv4_2_w, 9437184, cudaMemcpyHostToDevice));
  float* h_conv4_2_b = (float*)malloc(2048);
  if (!h_conv4_2_b) { fprintf(stderr, "malloc failed for conv4_2_b\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv4_2_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv4_2_b.bin\n"); return 1; }
    size_t nread = fread(h_conv4_2_b, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read vgg16p_tiled_conv4_2_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv4_2_b, h_conv4_2_b, 2048, cudaMemcpyHostToDevice));
  float* h_conv4_3_w = (float*)malloc(9437184);
  if (!h_conv4_3_w) { fprintf(stderr, "malloc failed for conv4_3_w\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv4_3_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv4_3_w.bin\n"); return 1; }
    size_t nread = fread(h_conv4_3_w, sizeof(float), 2359296, f);
    fclose(f);
    if (nread != 2359296) { fprintf(stderr, "Failed to read vgg16p_tiled_conv4_3_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv4_3_w, h_conv4_3_w, 9437184, cudaMemcpyHostToDevice));
  float* h_conv4_3_b = (float*)malloc(2048);
  if (!h_conv4_3_b) { fprintf(stderr, "malloc failed for conv4_3_b\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv4_3_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv4_3_b.bin\n"); return 1; }
    size_t nread = fread(h_conv4_3_b, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read vgg16p_tiled_conv4_3_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv4_3_b, h_conv4_3_b, 2048, cudaMemcpyHostToDevice));
  float* h_conv5_1_w = (float*)malloc(9437184);
  if (!h_conv5_1_w) { fprintf(stderr, "malloc failed for conv5_1_w\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv5_1_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv5_1_w.bin\n"); return 1; }
    size_t nread = fread(h_conv5_1_w, sizeof(float), 2359296, f);
    fclose(f);
    if (nread != 2359296) { fprintf(stderr, "Failed to read vgg16p_tiled_conv5_1_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv5_1_w, h_conv5_1_w, 9437184, cudaMemcpyHostToDevice));
  float* h_conv5_1_b = (float*)malloc(2048);
  if (!h_conv5_1_b) { fprintf(stderr, "malloc failed for conv5_1_b\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv5_1_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv5_1_b.bin\n"); return 1; }
    size_t nread = fread(h_conv5_1_b, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read vgg16p_tiled_conv5_1_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv5_1_b, h_conv5_1_b, 2048, cudaMemcpyHostToDevice));
  float* h_conv5_2_w = (float*)malloc(9437184);
  if (!h_conv5_2_w) { fprintf(stderr, "malloc failed for conv5_2_w\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv5_2_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv5_2_w.bin\n"); return 1; }
    size_t nread = fread(h_conv5_2_w, sizeof(float), 2359296, f);
    fclose(f);
    if (nread != 2359296) { fprintf(stderr, "Failed to read vgg16p_tiled_conv5_2_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv5_2_w, h_conv5_2_w, 9437184, cudaMemcpyHostToDevice));
  float* h_conv5_2_b = (float*)malloc(2048);
  if (!h_conv5_2_b) { fprintf(stderr, "malloc failed for conv5_2_b\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv5_2_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv5_2_b.bin\n"); return 1; }
    size_t nread = fread(h_conv5_2_b, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read vgg16p_tiled_conv5_2_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv5_2_b, h_conv5_2_b, 2048, cudaMemcpyHostToDevice));
  float* h_conv5_3_w = (float*)malloc(9437184);
  if (!h_conv5_3_w) { fprintf(stderr, "malloc failed for conv5_3_w\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv5_3_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv5_3_w.bin\n"); return 1; }
    size_t nread = fread(h_conv5_3_w, sizeof(float), 2359296, f);
    fclose(f);
    if (nread != 2359296) { fprintf(stderr, "Failed to read vgg16p_tiled_conv5_3_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv5_3_w, h_conv5_3_w, 9437184, cudaMemcpyHostToDevice));
  float* h_conv5_3_b = (float*)malloc(2048);
  if (!h_conv5_3_b) { fprintf(stderr, "malloc failed for conv5_3_b\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_conv5_3_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_conv5_3_b.bin\n"); return 1; }
    size_t nread = fread(h_conv5_3_b, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read vgg16p_tiled_conv5_3_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv5_3_b, h_conv5_3_b, 2048, cudaMemcpyHostToDevice));
  float* h_fc1_w = (float*)malloc(411041792);
  if (!h_fc1_w) { fprintf(stderr, "malloc failed for fc1_w\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_fc1_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_fc1_w.bin\n"); return 1; }
    size_t nread = fread(h_fc1_w, sizeof(float), 102760448, f);
    fclose(f);
    if (nread != 102760448) { fprintf(stderr, "Failed to read vgg16p_tiled_fc1_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc1_w, h_fc1_w, 411041792, cudaMemcpyHostToDevice));
  float* h_fc1_b = (float*)malloc(16384);
  if (!h_fc1_b) { fprintf(stderr, "malloc failed for fc1_b\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_fc1_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_fc1_b.bin\n"); return 1; }
    size_t nread = fread(h_fc1_b, sizeof(float), 4096, f);
    fclose(f);
    if (nread != 4096) { fprintf(stderr, "Failed to read vgg16p_tiled_fc1_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc1_b, h_fc1_b, 16384, cudaMemcpyHostToDevice));
  float* h_fc2_w = (float*)malloc(67108864);
  if (!h_fc2_w) { fprintf(stderr, "malloc failed for fc2_w\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_fc2_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_fc2_w.bin\n"); return 1; }
    size_t nread = fread(h_fc2_w, sizeof(float), 16777216, f);
    fclose(f);
    if (nread != 16777216) { fprintf(stderr, "Failed to read vgg16p_tiled_fc2_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc2_w, h_fc2_w, 67108864, cudaMemcpyHostToDevice));
  float* h_fc2_b = (float*)malloc(16384);
  if (!h_fc2_b) { fprintf(stderr, "malloc failed for fc2_b\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_fc2_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_fc2_b.bin\n"); return 1; }
    size_t nread = fread(h_fc2_b, sizeof(float), 4096, f);
    fclose(f);
    if (nread != 4096) { fprintf(stderr, "Failed to read vgg16p_tiled_fc2_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc2_b, h_fc2_b, 16384, cudaMemcpyHostToDevice));
  float* h_fc3_w = (float*)malloc(16384000);
  if (!h_fc3_w) { fprintf(stderr, "malloc failed for fc3_w\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_fc3_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_fc3_w.bin\n"); return 1; }
    size_t nread = fread(h_fc3_w, sizeof(float), 4096000, f);
    fclose(f);
    if (nread != 4096000) { fprintf(stderr, "Failed to read vgg16p_tiled_fc3_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc3_w, h_fc3_w, 16384000, cudaMemcpyHostToDevice));
  float* h_fc3_b = (float*)malloc(4000);
  if (!h_fc3_b) { fprintf(stderr, "malloc failed for fc3_b\n"); return 1; }
  { FILE* f = fopen("vgg16p_tiled_fc3_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open vgg16p_tiled_fc3_b.bin\n"); return 1; }
    size_t nread = fread(h_fc3_b, sizeof(float), 1000, f);
    fclose(f);
    if (nread != 1000) { fprintf(stderr, "Failed to read vgg16p_tiled_fc3_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc3_b, h_fc3_b, 4000, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(__io_stop));
  CUDA_CHECK(cudaEventSynchronize(__io_stop));
  {
    float __io_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&__io_ms, __io_start, __io_stop));
    printf("MEMTIME_H2D vgg16p_tiled %.6f\n", __io_ms);
  }

  float* h_output = (float*)malloc(4000);
  if (!h_output) { fprintf(stderr, "malloc failed for output\n"); return 1; }

  cudaEvent_t __timer_start, __timer_stop;
  CUDA_CHECK(cudaEventCreate(&__timer_start));
  CUDA_CHECK(cudaEventCreate(&__timer_stop));
  CUDA_CHECK(cudaEventRecord(__timer_start));
  vgg16p_tiled();
  CUDA_CHECK(cudaEventRecord(__timer_stop));
  CUDA_CHECK(cudaEventSynchronize(__timer_stop));
  {
    float __elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&__elapsed_ms, __timer_start, __timer_stop));
    printf("TIME vgg16p_tiled %.6f\n", __elapsed_ms);
  }
  CUDA_CHECK(cudaEventDestroy(__timer_start));
  CUDA_CHECK(cudaEventDestroy(__timer_stop));

  vgg16p_tiled_kt();

  CUDA_CHECK(cudaEventRecord(__io_start));
  CUDA_CHECK(cudaMemcpy(h_output, d_output, 4000, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(__io_stop));
  CUDA_CHECK(cudaEventSynchronize(__io_stop));
  {
    float __d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&__d2h_ms, __io_start, __io_stop));
    printf("MEMTIME_D2H vgg16p_tiled %.6f\n", __d2h_ms);
  }
  CUDA_CHECK(cudaEventDestroy(__io_start));
  CUDA_CHECK(cudaEventDestroy(__io_stop));

  FILE* fout = fopen("vgg16p_tiled_output.bin", "wb");
  if (!fout) { fprintf(stderr, "Cannot open %s\n", "vgg16p_tiled_output.bin"); return 1; }
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
  CUDA_CHECK(cudaFree(d_relu4_1_out_ws));
  CUDA_CHECK(cudaFree(d_relu4_2_out_ws));
  CUDA_CHECK(cudaFree(d_relu4_3_out_ws));
  CUDA_CHECK(cudaFree(d_relu5_1_out_ws));
  CUDA_CHECK(cudaFree(d_relu5_2_out_ws));
  CUDA_CHECK(cudaFree(d_relu5_3_out_ws));
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
