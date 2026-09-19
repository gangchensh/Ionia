#include <cuda_runtime.h>
#include <math.h>

/*LAUNCH 7 7 16 16 16*/
__global__ void ast0_relu1_out_fused()
{
  __shared__ float sh[37][37];
  __shared__ float shw[4][49];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ow = blockIdx.x * 16 + tx;
  int oh = blockIdx.y * 16 + ty;
  int iw0 = blockIdx.x * 32 - 3;
  int ih0 = blockIdx.y * 32 - 3;
  int oc0 = blockIdx.z * 4;
  float acc[4];
  #pragma unroll
  for (int cb = 0; cb < 4; cb += 1) acc[cb] = 0.0f;
  for (int c = 0; c < 3; c += 1) {
    for (int sy = ty; sy < 37; sy += 16)
      for (int sx = tx; sx < 37; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh[sy][sx] = (iy < 0 || iy >= 224 || ix < 0 || ix >= 224)
                     ? 0.0f
                     : input[0][c][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 196) {
        int cb = tid / 49;
        int t = tid - cb * 49;
        shw[cb][t] = (oc0 + cb < 64) ? conv1_w[oc0 + cb][c][t / 7][t - (t / 7) * 7] : 0.0f;
      }
    }
    __syncthreads();
    int by0 = ty * 2;
    int bx0 = tx * 2;
    #pragma unroll
    for (int cb = 0; cb < 4; cb += 1) {
      if (oc0 + cb < 64) {
        #pragma unroll
        for (int k1 = 0; k1 < 7; k1 += 1)
          #pragma unroll
          for (int k2 = 0; k2 < 7; k2 += 1)
            acc[cb] += sh[by0 + k1][bx0 + k2] * shw[cb][k1 * 7 + k2];
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int cb = 0; cb < 4; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 64 && oh < 112 && ow < 112) {
      float result = acc[cb] + conv1_b[oc];
      float fused_result = (((result - bn1_mean[oc]) / sqrtf((bn1_var[oc] + bn_eps))) * bn1_gamma[oc] + bn1_beta[oc]);
      relu1_out[0][oc][oh][ow] = (fused_result <= 0.0f) ? 0.0f : fused_result;
    }
  }
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

/*LAUNCH 2 2 32 16 16*/
__global__ void ast2_l1_b1_relu1_out_fused()
{
  __shared__ float sh2[2][34][35];
  __shared__ float shw2[2][2][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int oy0 = blockIdx.y * 32 + ty * 2;
  int ox0 = blockIdx.x * 32 + tx * 2;
  int ih0 = blockIdx.y * 32 - 1;
  int iw0 = blockIdx.x * 32 - 1;
  int c0 = 0;
  int c1 = 64;
  int oc0 = blockIdx.z * 2;
  float acc[2][2][2];
  #pragma unroll
  for (int cb = 0; cb < 2; cb += 1)
    for (int dy = 0; dy < 2; dy += 1)
      for (int dx = 0; dx < 2; dx += 1)
        acc[cb][dy][dx] = 0.0f;
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[0][sy][sx] = (iy < 0 || iy >= 56 || ix < 0 || ix >= 56)
                     ? 0.0f
                     : pool_out[0][c0][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 18) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[0][cb][t] = (oc0 + cb < 64) ? l1_b1_w1[oc0 + cb][c0][t / 3][t - (t / 3) * 3] : 0.0f;
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
                     : pool_out[0][(c + 1)][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 18) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[cur ^ 1][cb][t] = (oc0 + cb < 64) ? l1_b1_w1[oc0 + cb][(c + 1)][t / 3][t - (t / 3) * 3] : 0.0f;
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
    for (int cb = 0; cb < 2; cb += 1) {
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
  for (int cb = 0; cb < 2; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 64) {
      if (oy0 < 56 && ox0 < 56) {
        float result = acc[cb][0][0] + l1_b1_b1[oc];
        float fused_result = (((result - l1_b1_bn1_mean[oc]) / sqrtf((l1_b1_bn1_var[oc] + bn_eps))) * l1_b1_bn1_gamma[oc] + l1_b1_bn1_beta[oc]);
        l1_b1_relu1_out[0][oc][oy0][ox0] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 < 56 && ox0 + 1 < 56) {
        float result = acc[cb][0][1] + l1_b1_b1[oc];
        float fused_result = (((result - l1_b1_bn1_mean[oc]) / sqrtf((l1_b1_bn1_var[oc] + bn_eps))) * l1_b1_bn1_gamma[oc] + l1_b1_bn1_beta[oc]);
        l1_b1_relu1_out[0][oc][oy0][ox0 + 1] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 + 1 < 56 && ox0 < 56) {
        float result = acc[cb][1][0] + l1_b1_b1[oc];
        float fused_result = (((result - l1_b1_bn1_mean[oc]) / sqrtf((l1_b1_bn1_var[oc] + bn_eps))) * l1_b1_bn1_gamma[oc] + l1_b1_bn1_beta[oc]);
        l1_b1_relu1_out[0][oc][oy0 + 1][ox0] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 + 1 < 56 && ox0 + 1 < 56) {
        float result = acc[cb][1][1] + l1_b1_b1[oc];
        float fused_result = (((result - l1_b1_bn1_mean[oc]) / sqrtf((l1_b1_bn1_var[oc] + bn_eps))) * l1_b1_bn1_gamma[oc] + l1_b1_bn1_beta[oc]);
        l1_b1_relu1_out[0][oc][oy0 + 1][ox0 + 1] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
    }
  }
}

/*LAUNCH 2 2 32 16 16*/
__global__ void ast3_l1_b1_add_out_fused()
{
  __shared__ float sh2[2][34][35];
  __shared__ float shw2[2][2][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int oy0 = blockIdx.y * 32 + ty * 2;
  int ox0 = blockIdx.x * 32 + tx * 2;
  int ih0 = blockIdx.y * 32 - 1;
  int iw0 = blockIdx.x * 32 - 1;
  int c0 = 0;
  int c1 = 64;
  int oc0 = blockIdx.z * 2;
  float acc[2][2][2];
  #pragma unroll
  for (int cb = 0; cb < 2; cb += 1)
    for (int dy = 0; dy < 2; dy += 1)
      for (int dx = 0; dx < 2; dx += 1)
        acc[cb][dy][dx] = 0.0f;
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[0][sy][sx] = (iy < 0 || iy >= 56 || ix < 0 || ix >= 56)
                     ? 0.0f
                     : l1_b1_relu1_out[0][c0][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 18) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[0][cb][t] = (oc0 + cb < 64) ? l1_b1_w2[oc0 + cb][c0][t / 3][t - (t / 3) * 3] : 0.0f;
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
                     : l1_b1_relu1_out[0][(c + 1)][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 18) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[cur ^ 1][cb][t] = (oc0 + cb < 64) ? l1_b1_w2[oc0 + cb][(c + 1)][t / 3][t - (t / 3) * 3] : 0.0f;
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
    for (int cb = 0; cb < 2; cb += 1) {
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
  for (int cb = 0; cb < 2; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 64) {
      if (oy0 < 56 && ox0 < 56) {
        float result = acc[cb][0][0] + l1_b1_b2[oc];
        l1_b1_add_out[0][oc][oy0][ox0] = ((((result - l1_b1_bn2_mean[oc]) / sqrtf((l1_b1_bn2_var[oc] + bn_eps))) * l1_b1_bn2_gamma[oc] + l1_b1_bn2_beta[oc]) + pool_out[0][oc][oy0][ox0]);
      }
      if (oy0 < 56 && ox0 + 1 < 56) {
        float result = acc[cb][0][1] + l1_b1_b2[oc];
        l1_b1_add_out[0][oc][oy0][ox0 + 1] = ((((result - l1_b1_bn2_mean[oc]) / sqrtf((l1_b1_bn2_var[oc] + bn_eps))) * l1_b1_bn2_gamma[oc] + l1_b1_bn2_beta[oc]) + pool_out[0][oc][oy0][ox0 + 1]);
      }
      if (oy0 + 1 < 56 && ox0 < 56) {
        float result = acc[cb][1][0] + l1_b1_b2[oc];
        l1_b1_add_out[0][oc][oy0 + 1][ox0] = ((((result - l1_b1_bn2_mean[oc]) / sqrtf((l1_b1_bn2_var[oc] + bn_eps))) * l1_b1_bn2_gamma[oc] + l1_b1_bn2_beta[oc]) + pool_out[0][oc][oy0 + 1][ox0]);
      }
      if (oy0 + 1 < 56 && ox0 + 1 < 56) {
        float result = acc[cb][1][1] + l1_b1_b2[oc];
        l1_b1_add_out[0][oc][oy0 + 1][ox0 + 1] = ((((result - l1_b1_bn2_mean[oc]) / sqrtf((l1_b1_bn2_var[oc] + bn_eps))) * l1_b1_bn2_gamma[oc] + l1_b1_bn2_beta[oc]) + pool_out[0][oc][oy0 + 1][ox0 + 1]);
      }
    }
  }
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

/*LAUNCH 2 2 32 16 16*/
__global__ void ast5_l1_b2_relu1_out_fused()
{
  __shared__ float sh2[2][34][35];
  __shared__ float shw2[2][2][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int oy0 = blockIdx.y * 32 + ty * 2;
  int ox0 = blockIdx.x * 32 + tx * 2;
  int ih0 = blockIdx.y * 32 - 1;
  int iw0 = blockIdx.x * 32 - 1;
  int c0 = 0;
  int c1 = 64;
  int oc0 = blockIdx.z * 2;
  float acc[2][2][2];
  #pragma unroll
  for (int cb = 0; cb < 2; cb += 1)
    for (int dy = 0; dy < 2; dy += 1)
      for (int dx = 0; dx < 2; dx += 1)
        acc[cb][dy][dx] = 0.0f;
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[0][sy][sx] = (iy < 0 || iy >= 56 || ix < 0 || ix >= 56)
                     ? 0.0f
                     : l1_b1_add_out[0][c0][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 18) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[0][cb][t] = (oc0 + cb < 64) ? l1_b2_w1[oc0 + cb][c0][t / 3][t - (t / 3) * 3] : 0.0f;
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
                     : l1_b1_add_out[0][(c + 1)][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 18) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[cur ^ 1][cb][t] = (oc0 + cb < 64) ? l1_b2_w1[oc0 + cb][(c + 1)][t / 3][t - (t / 3) * 3] : 0.0f;
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
    for (int cb = 0; cb < 2; cb += 1) {
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
  for (int cb = 0; cb < 2; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 64) {
      if (oy0 < 56 && ox0 < 56) {
        float result = acc[cb][0][0] + l1_b2_b1[oc];
        float fused_result = (((result - l1_b2_bn1_mean[oc]) / sqrtf((l1_b2_bn1_var[oc] + bn_eps))) * l1_b2_bn1_gamma[oc] + l1_b2_bn1_beta[oc]);
        l1_b2_relu1_out[0][oc][oy0][ox0] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 < 56 && ox0 + 1 < 56) {
        float result = acc[cb][0][1] + l1_b2_b1[oc];
        float fused_result = (((result - l1_b2_bn1_mean[oc]) / sqrtf((l1_b2_bn1_var[oc] + bn_eps))) * l1_b2_bn1_gamma[oc] + l1_b2_bn1_beta[oc]);
        l1_b2_relu1_out[0][oc][oy0][ox0 + 1] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 + 1 < 56 && ox0 < 56) {
        float result = acc[cb][1][0] + l1_b2_b1[oc];
        float fused_result = (((result - l1_b2_bn1_mean[oc]) / sqrtf((l1_b2_bn1_var[oc] + bn_eps))) * l1_b2_bn1_gamma[oc] + l1_b2_bn1_beta[oc]);
        l1_b2_relu1_out[0][oc][oy0 + 1][ox0] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
      if (oy0 + 1 < 56 && ox0 + 1 < 56) {
        float result = acc[cb][1][1] + l1_b2_b1[oc];
        float fused_result = (((result - l1_b2_bn1_mean[oc]) / sqrtf((l1_b2_bn1_var[oc] + bn_eps))) * l1_b2_bn1_gamma[oc] + l1_b2_bn1_beta[oc]);
        l1_b2_relu1_out[0][oc][oy0 + 1][ox0 + 1] = (fused_result <= 0.0f) ? 0.0f : fused_result;
      }
    }
  }
}

/*LAUNCH 2 2 32 16 16*/
__global__ void ast6_l1_b2_add_out_fused()
{
  __shared__ float sh2[2][34][35];
  __shared__ float shw2[2][2][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int oy0 = blockIdx.y * 32 + ty * 2;
  int ox0 = blockIdx.x * 32 + tx * 2;
  int ih0 = blockIdx.y * 32 - 1;
  int iw0 = blockIdx.x * 32 - 1;
  int c0 = 0;
  int c1 = 64;
  int oc0 = blockIdx.z * 2;
  float acc[2][2][2];
  #pragma unroll
  for (int cb = 0; cb < 2; cb += 1)
    for (int dy = 0; dy < 2; dy += 1)
      for (int dx = 0; dx < 2; dx += 1)
        acc[cb][dy][dx] = 0.0f;
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[0][sy][sx] = (iy < 0 || iy >= 56 || ix < 0 || ix >= 56)
                     ? 0.0f
                     : l1_b2_relu1_out[0][c0][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 18) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[0][cb][t] = (oc0 + cb < 64) ? l1_b2_w2[oc0 + cb][c0][t / 3][t - (t / 3) * 3] : 0.0f;
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
                     : l1_b2_relu1_out[0][(c + 1)][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 18) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[cur ^ 1][cb][t] = (oc0 + cb < 64) ? l1_b2_w2[oc0 + cb][(c + 1)][t / 3][t - (t / 3) * 3] : 0.0f;
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
    for (int cb = 0; cb < 2; cb += 1) {
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
  for (int cb = 0; cb < 2; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 64) {
      if (oy0 < 56 && ox0 < 56) {
        float result = acc[cb][0][0] + l1_b2_b2[oc];
        l1_b2_add_out[0][oc][oy0][ox0] = ((((result - l1_b2_bn2_mean[oc]) / sqrtf((l1_b2_bn2_var[oc] + bn_eps))) * l1_b2_bn2_gamma[oc] + l1_b2_bn2_beta[oc]) + l1_b1_add_out[0][oc][oy0][ox0]);
      }
      if (oy0 < 56 && ox0 + 1 < 56) {
        float result = acc[cb][0][1] + l1_b2_b2[oc];
        l1_b2_add_out[0][oc][oy0][ox0 + 1] = ((((result - l1_b2_bn2_mean[oc]) / sqrtf((l1_b2_bn2_var[oc] + bn_eps))) * l1_b2_bn2_gamma[oc] + l1_b2_bn2_beta[oc]) + l1_b1_add_out[0][oc][oy0][ox0 + 1]);
      }
      if (oy0 + 1 < 56 && ox0 < 56) {
        float result = acc[cb][1][0] + l1_b2_b2[oc];
        l1_b2_add_out[0][oc][oy0 + 1][ox0] = ((((result - l1_b2_bn2_mean[oc]) / sqrtf((l1_b2_bn2_var[oc] + bn_eps))) * l1_b2_bn2_gamma[oc] + l1_b2_bn2_beta[oc]) + l1_b1_add_out[0][oc][oy0 + 1][ox0]);
      }
      if (oy0 + 1 < 56 && ox0 + 1 < 56) {
        float result = acc[cb][1][1] + l1_b2_b2[oc];
        l1_b2_add_out[0][oc][oy0 + 1][ox0 + 1] = ((((result - l1_b2_bn2_mean[oc]) / sqrtf((l1_b2_bn2_var[oc] + bn_eps))) * l1_b2_bn2_gamma[oc] + l1_b2_bn2_beta[oc]) + l1_b1_add_out[0][oc][oy0 + 1][ox0 + 1]);
      }
    }
  }
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

/*LAUNCH 2 2 64 16 16*/
__global__ void ast10_l2_b1_relu1_out_fused()
{
  __shared__ float sh[33][33];
  __shared__ float shw[4][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ow = blockIdx.x * 16 + tx;
  int oh = blockIdx.y * 16 + ty;
  int iw0 = blockIdx.x * 32 - 1;
  int ih0 = blockIdx.y * 32 - 1;
  int sc = blockIdx.z / 32;
  int oc0 = (blockIdx.z - sc * 32) * 4;
  int c0 = sc * 32;
  int c1 = c0 + 32;
  if (c1 > 64) c1 = 64;
  float acc[4];
  #pragma unroll
  for (int cb = 0; cb < 4; cb += 1) acc[cb] = 0.0f;
  for (int c = c0; c < c1; c += 1) {
    for (int sy = ty; sy < 33; sy += 16)
      for (int sx = tx; sx < 33; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh[sy][sx] = (iy < 0 || iy >= 56 || ix < 0 || ix >= 56)
                     ? 0.0f
                     : l1_b2_add_out[0][c][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 36) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw[cb][t] = (oc0 + cb < 128) ? l2_b1_w1[oc0 + cb][c][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
    __syncthreads();
    int by0 = ty * 2;
    int bx0 = tx * 2;
    #pragma unroll
    for (int cb = 0; cb < 4; cb += 1) {
      if (oc0 + cb < 128) {
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
    if (oc < 128 && oh < 28 && ow < 28)
      l2_b1_relu1_out_ws[sc][oc][oh][ow] = acc[cb];
  }
}

/*LAUNCH2 392 1 1 256 1*/
__global__ void ast10_l2_b1_relu1_out_fused_red()
{
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= 100352) return;
  int ow = idx % 28;
  int t0 = idx / 28;
  int oh = t0 % 28;
  int oc = t0 / 28;
  float acc = l2_b1_relu1_out_ws[0][oc][oh][ow];
  acc += l2_b1_relu1_out_ws[1][oc][oh][ow];
  float result = acc + l2_b1_b1[oc];
  float fused_result = (((result - l2_b1_bn1_mean[oc]) / sqrtf((l2_b1_bn1_var[oc] + bn_eps))) * l2_b1_bn1_gamma[oc] + l2_b1_bn1_beta[oc]);
  l2_b1_relu1_out[0][oc][oh][ow] = (fused_result <= 0.0f) ? 0.0f : fused_result;
}

/*LAUNCH 1 1 128 16 16*/
__global__ void ast11_l2_b1_add_out_fused()
{
  __shared__ float sh2[2][34][35];
  __shared__ float shw2[2][2][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int oy0 = blockIdx.y * 32 + ty * 2;
  int ox0 = blockIdx.x * 32 + tx * 2;
  int ih0 = blockIdx.y * 32 - 1;
  int iw0 = blockIdx.x * 32 - 1;
  int c0 = 0;
  int c1 = 128;
  int sc = blockIdx.z / 64;
  int oc0 = (blockIdx.z - sc * 64) * 2;
  c0 = sc * 64;
  c1 = c0 + 64;
  if (c1 > 128) c1 = 128;
  float acc[2][2][2];
  #pragma unroll
  for (int cb = 0; cb < 2; cb += 1)
    for (int dy = 0; dy < 2; dy += 1)
      for (int dx = 0; dx < 2; dx += 1)
        acc[cb][dy][dx] = 0.0f;
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[0][sy][sx] = (iy < 0 || iy >= 28 || ix < 0 || ix >= 28)
                     ? 0.0f
                     : l2_b1_relu1_out[0][c0][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 18) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[0][cb][t] = (oc0 + cb < 128) ? l2_b1_w2[oc0 + cb][c0][t / 3][t - (t / 3) * 3] : 0.0f;
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
                     : l2_b1_relu1_out[0][(c + 1)][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 18) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[cur ^ 1][cb][t] = (oc0 + cb < 128) ? l2_b1_w2[oc0 + cb][(c + 1)][t / 3][t - (t / 3) * 3] : 0.0f;
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
    for (int cb = 0; cb < 2; cb += 1) {
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
  for (int cb = 0; cb < 2; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 128) {
      if (oy0 < 28 && ox0 < 28) {
        l2_b1_add_out_ws[sc][oc][oy0][ox0] = acc[cb][0][0];
      }
      if (oy0 < 28 && ox0 + 1 < 28) {
        l2_b1_add_out_ws[sc][oc][oy0][ox0 + 1] = acc[cb][0][1];
      }
      if (oy0 + 1 < 28 && ox0 < 28) {
        l2_b1_add_out_ws[sc][oc][oy0 + 1][ox0] = acc[cb][1][0];
      }
      if (oy0 + 1 < 28 && ox0 + 1 < 28) {
        l2_b1_add_out_ws[sc][oc][oy0 + 1][ox0 + 1] = acc[cb][1][1];
      }
    }
  }
}

/*LAUNCH2 392 1 1 256 1*/
__global__ void ast11_l2_b1_add_out_fused_red()
{
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= 100352) return;
  int ow = idx % 28;
  int t0 = idx / 28;
  int oh = t0 % 28;
  int oc = t0 / 28;
  float acc = l2_b1_add_out_ws[0][oc][oh][ow];
  acc += l2_b1_add_out_ws[1][oc][oh][ow];
  float result = acc + l2_b1_b2[oc];
  l2_b1_add_out[0][oc][oh][ow] = ((((result - l2_b1_bn2_mean[oc]) / sqrtf((l2_b1_bn2_var[oc] + bn_eps))) * l2_b1_bn2_gamma[oc] + l2_b1_bn2_beta[oc]) + l2_b1_shortcut_out[0][oc][oh][ow]);
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

/*LAUNCH 1 1 128 16 16*/
__global__ void ast13_l2_b2_relu1_out_fused()
{
  __shared__ float sh2[2][34][35];
  __shared__ float shw2[2][2][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int oy0 = blockIdx.y * 32 + ty * 2;
  int ox0 = blockIdx.x * 32 + tx * 2;
  int ih0 = blockIdx.y * 32 - 1;
  int iw0 = blockIdx.x * 32 - 1;
  int c0 = 0;
  int c1 = 128;
  int sc = blockIdx.z / 64;
  int oc0 = (blockIdx.z - sc * 64) * 2;
  c0 = sc * 64;
  c1 = c0 + 64;
  if (c1 > 128) c1 = 128;
  float acc[2][2][2];
  #pragma unroll
  for (int cb = 0; cb < 2; cb += 1)
    for (int dy = 0; dy < 2; dy += 1)
      for (int dx = 0; dx < 2; dx += 1)
        acc[cb][dy][dx] = 0.0f;
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[0][sy][sx] = (iy < 0 || iy >= 28 || ix < 0 || ix >= 28)
                     ? 0.0f
                     : l2_b1_add_out[0][c0][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 18) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[0][cb][t] = (oc0 + cb < 128) ? l2_b2_w1[oc0 + cb][c0][t / 3][t - (t / 3) * 3] : 0.0f;
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
                     : l2_b1_add_out[0][(c + 1)][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 18) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[cur ^ 1][cb][t] = (oc0 + cb < 128) ? l2_b2_w1[oc0 + cb][(c + 1)][t / 3][t - (t / 3) * 3] : 0.0f;
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
    for (int cb = 0; cb < 2; cb += 1) {
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
  for (int cb = 0; cb < 2; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 128) {
      if (oy0 < 28 && ox0 < 28) {
        l2_b2_relu1_out_ws[sc][oc][oy0][ox0] = acc[cb][0][0];
      }
      if (oy0 < 28 && ox0 + 1 < 28) {
        l2_b2_relu1_out_ws[sc][oc][oy0][ox0 + 1] = acc[cb][0][1];
      }
      if (oy0 + 1 < 28 && ox0 < 28) {
        l2_b2_relu1_out_ws[sc][oc][oy0 + 1][ox0] = acc[cb][1][0];
      }
      if (oy0 + 1 < 28 && ox0 + 1 < 28) {
        l2_b2_relu1_out_ws[sc][oc][oy0 + 1][ox0 + 1] = acc[cb][1][1];
      }
    }
  }
}

/*LAUNCH2 392 1 1 256 1*/
__global__ void ast13_l2_b2_relu1_out_fused_red()
{
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= 100352) return;
  int ow = idx % 28;
  int t0 = idx / 28;
  int oh = t0 % 28;
  int oc = t0 / 28;
  float acc = l2_b2_relu1_out_ws[0][oc][oh][ow];
  acc += l2_b2_relu1_out_ws[1][oc][oh][ow];
  float result = acc + l2_b2_b1[oc];
  float fused_result = (((result - l2_b2_bn1_mean[oc]) / sqrtf((l2_b2_bn1_var[oc] + bn_eps))) * l2_b2_bn1_gamma[oc] + l2_b2_bn1_beta[oc]);
  l2_b2_relu1_out[0][oc][oh][ow] = (fused_result <= 0.0f) ? 0.0f : fused_result;
}

/*LAUNCH 1 1 128 16 16*/
__global__ void ast14_l2_b2_add_out_fused()
{
  __shared__ float sh2[2][34][35];
  __shared__ float shw2[2][2][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int oy0 = blockIdx.y * 32 + ty * 2;
  int ox0 = blockIdx.x * 32 + tx * 2;
  int ih0 = blockIdx.y * 32 - 1;
  int iw0 = blockIdx.x * 32 - 1;
  int c0 = 0;
  int c1 = 128;
  int sc = blockIdx.z / 64;
  int oc0 = (blockIdx.z - sc * 64) * 2;
  c0 = sc * 64;
  c1 = c0 + 64;
  if (c1 > 128) c1 = 128;
  float acc[2][2][2];
  #pragma unroll
  for (int cb = 0; cb < 2; cb += 1)
    for (int dy = 0; dy < 2; dy += 1)
      for (int dx = 0; dx < 2; dx += 1)
        acc[cb][dy][dx] = 0.0f;
    for (int sy = ty; sy < 34; sy += 16)
      for (int sx = tx; sx < 34; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh2[0][sy][sx] = (iy < 0 || iy >= 28 || ix < 0 || ix >= 28)
                     ? 0.0f
                     : l2_b2_relu1_out[0][c0][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 18) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[0][cb][t] = (oc0 + cb < 128) ? l2_b2_w2[oc0 + cb][c0][t / 3][t - (t / 3) * 3] : 0.0f;
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
                     : l2_b2_relu1_out[0][(c + 1)][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 18) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw2[cur ^ 1][cb][t] = (oc0 + cb < 128) ? l2_b2_w2[oc0 + cb][(c + 1)][t / 3][t - (t / 3) * 3] : 0.0f;
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
    for (int cb = 0; cb < 2; cb += 1) {
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
  for (int cb = 0; cb < 2; cb += 1) {
    int oc = oc0 + cb;
    if (oc < 128) {
      if (oy0 < 28 && ox0 < 28) {
        l2_b2_add_out_ws[sc][oc][oy0][ox0] = acc[cb][0][0];
      }
      if (oy0 < 28 && ox0 + 1 < 28) {
        l2_b2_add_out_ws[sc][oc][oy0][ox0 + 1] = acc[cb][0][1];
      }
      if (oy0 + 1 < 28 && ox0 < 28) {
        l2_b2_add_out_ws[sc][oc][oy0 + 1][ox0] = acc[cb][1][0];
      }
      if (oy0 + 1 < 28 && ox0 + 1 < 28) {
        l2_b2_add_out_ws[sc][oc][oy0 + 1][ox0 + 1] = acc[cb][1][1];
      }
    }
  }
}

/*LAUNCH2 392 1 1 256 1*/
__global__ void ast14_l2_b2_add_out_fused_red()
{
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= 100352) return;
  int ow = idx % 28;
  int t0 = idx / 28;
  int oh = t0 % 28;
  int oc = t0 / 28;
  float acc = l2_b2_add_out_ws[0][oc][oh][ow];
  acc += l2_b2_add_out_ws[1][oc][oh][ow];
  float result = acc + l2_b2_b2[oc];
  l2_b2_add_out[0][oc][oh][ow] = ((((result - l2_b2_bn2_mean[oc]) / sqrtf((l2_b2_bn2_var[oc] + bn_eps))) * l2_b2_bn2_gamma[oc] + l2_b2_bn2_beta[oc]) + l2_b1_add_out[0][oc][oh][ow]);
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

/*LAUNCH 1 1 256 16 16*/
__global__ void ast18_l3_b1_relu1_out_fused()
{
  __shared__ float sh[33][33];
  __shared__ float shw[4][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ow = blockIdx.x * 16 + tx;
  int oh = blockIdx.y * 16 + ty;
  int iw0 = blockIdx.x * 32 - 1;
  int ih0 = blockIdx.y * 32 - 1;
  int sc = blockIdx.z / 64;
  int oc0 = (blockIdx.z - sc * 64) * 4;
  int c0 = sc * 32;
  int c1 = c0 + 32;
  if (c1 > 128) c1 = 128;
  float acc[4];
  #pragma unroll
  for (int cb = 0; cb < 4; cb += 1) acc[cb] = 0.0f;
  for (int c = c0; c < c1; c += 1) {
    for (int sy = ty; sy < 33; sy += 16)
      for (int sx = tx; sx < 33; sx += 16) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh[sy][sx] = (iy < 0 || iy >= 28 || ix < 0 || ix >= 28)
                     ? 0.0f
                     : l2_b2_add_out[0][c][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 36) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw[cb][t] = (oc0 + cb < 256) ? l3_b1_w1[oc0 + cb][c][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
    __syncthreads();
    int by0 = ty * 2;
    int bx0 = tx * 2;
    #pragma unroll
    for (int cb = 0; cb < 4; cb += 1) {
      if (oc0 + cb < 256) {
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
    if (oc < 256 && oh < 14 && ow < 14)
      l3_b1_relu1_out_ws[sc][oc][oh][ow] = acc[cb];
  }
}

/*LAUNCH2 196 1 1 256 1*/
__global__ void ast18_l3_b1_relu1_out_fused_red()
{
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= 50176) return;
  int ow = idx % 14;
  int t0 = idx / 14;
  int oh = t0 % 14;
  int oc = t0 / 14;
  float acc = l3_b1_relu1_out_ws[0][oc][oh][ow];
  acc += l3_b1_relu1_out_ws[1][oc][oh][ow];
  acc += l3_b1_relu1_out_ws[2][oc][oh][ow];
  acc += l3_b1_relu1_out_ws[3][oc][oh][ow];
  float result = acc + l3_b1_b1[oc];
  float fused_result = (((result - l3_b1_bn1_mean[oc]) / sqrtf((l3_b1_bn1_var[oc] + bn_eps))) * l3_b1_bn1_gamma[oc] + l3_b1_bn1_beta[oc]);
  l3_b1_relu1_out[0][oc][oh][ow] = (fused_result <= 0.0f) ? 0.0f : fused_result;
}

/*LAUNCH 1 1 256 16 16*/
__global__ void ast19_l3_b1_add_out_fused()
{
  __shared__ float sh[18][18];
  __shared__ float shw[4][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ow = blockIdx.x * 16 + tx;
  int oh = blockIdx.y * 16 + ty;
  int iw0 = blockIdx.x * 16 - 1;
  int ih0 = blockIdx.y * 16 - 1;
  int sc = blockIdx.z / 64;
  int oc0 = (blockIdx.z - sc * 64) * 4;
  int c0 = sc * 64;
  int c1 = c0 + 64;
  if (c1 > 256) c1 = 256;
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
                     : l3_b1_relu1_out[0][c][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 36) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw[cb][t] = (oc0 + cb < 256) ? l3_b1_w2[oc0 + cb][c][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
    __syncthreads();
    int by0 = ty * 1;
    int bx0 = tx * 1;
    #pragma unroll
    for (int cb = 0; cb < 4; cb += 1) {
      if (oc0 + cb < 256) {
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
    if (oc < 256 && oh < 14 && ow < 14)
      l3_b1_add_out_ws[sc][oc][oh][ow] = acc[cb];
  }
}

/*LAUNCH2 196 1 1 256 1*/
__global__ void ast19_l3_b1_add_out_fused_red()
{
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= 50176) return;
  int ow = idx % 14;
  int t0 = idx / 14;
  int oh = t0 % 14;
  int oc = t0 / 14;
  float acc = l3_b1_add_out_ws[0][oc][oh][ow];
  acc += l3_b1_add_out_ws[1][oc][oh][ow];
  acc += l3_b1_add_out_ws[2][oc][oh][ow];
  acc += l3_b1_add_out_ws[3][oc][oh][ow];
  float result = acc + l3_b1_b2[oc];
  l3_b1_add_out[0][oc][oh][ow] = ((((result - l3_b1_bn2_mean[oc]) / sqrtf((l3_b1_bn2_var[oc] + bn_eps))) * l3_b1_bn2_gamma[oc] + l3_b1_bn2_beta[oc]) + l3_b1_shortcut_out[0][oc][oh][ow]);
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

/*LAUNCH 1 1 256 16 16*/
__global__ void ast21_l3_b2_relu1_out_fused()
{
  __shared__ float sh[18][18];
  __shared__ float shw[4][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ow = blockIdx.x * 16 + tx;
  int oh = blockIdx.y * 16 + ty;
  int iw0 = blockIdx.x * 16 - 1;
  int ih0 = blockIdx.y * 16 - 1;
  int sc = blockIdx.z / 64;
  int oc0 = (blockIdx.z - sc * 64) * 4;
  int c0 = sc * 64;
  int c1 = c0 + 64;
  if (c1 > 256) c1 = 256;
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
                     : l3_b1_add_out[0][c][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 36) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw[cb][t] = (oc0 + cb < 256) ? l3_b2_w1[oc0 + cb][c][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
    __syncthreads();
    int by0 = ty * 1;
    int bx0 = tx * 1;
    #pragma unroll
    for (int cb = 0; cb < 4; cb += 1) {
      if (oc0 + cb < 256) {
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
    if (oc < 256 && oh < 14 && ow < 14)
      l3_b2_relu1_out_ws[sc][oc][oh][ow] = acc[cb];
  }
}

/*LAUNCH2 196 1 1 256 1*/
__global__ void ast21_l3_b2_relu1_out_fused_red()
{
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= 50176) return;
  int ow = idx % 14;
  int t0 = idx / 14;
  int oh = t0 % 14;
  int oc = t0 / 14;
  float acc = l3_b2_relu1_out_ws[0][oc][oh][ow];
  acc += l3_b2_relu1_out_ws[1][oc][oh][ow];
  acc += l3_b2_relu1_out_ws[2][oc][oh][ow];
  acc += l3_b2_relu1_out_ws[3][oc][oh][ow];
  float result = acc + l3_b2_b1[oc];
  float fused_result = (((result - l3_b2_bn1_mean[oc]) / sqrtf((l3_b2_bn1_var[oc] + bn_eps))) * l3_b2_bn1_gamma[oc] + l3_b2_bn1_beta[oc]);
  l3_b2_relu1_out[0][oc][oh][ow] = (fused_result <= 0.0f) ? 0.0f : fused_result;
}

/*LAUNCH 1 1 256 16 16*/
__global__ void ast22_l3_b2_add_out_fused()
{
  __shared__ float sh[18][18];
  __shared__ float shw[4][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ow = blockIdx.x * 16 + tx;
  int oh = blockIdx.y * 16 + ty;
  int iw0 = blockIdx.x * 16 - 1;
  int ih0 = blockIdx.y * 16 - 1;
  int sc = blockIdx.z / 64;
  int oc0 = (blockIdx.z - sc * 64) * 4;
  int c0 = sc * 64;
  int c1 = c0 + 64;
  if (c1 > 256) c1 = 256;
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
                     : l3_b2_relu1_out[0][c][iy][ix];
      }
    {
      int tid = ty * 16 + tx;
      if (tid < 36) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw[cb][t] = (oc0 + cb < 256) ? l3_b2_w2[oc0 + cb][c][t / 3][t - (t / 3) * 3] : 0.0f;
      }
    }
    __syncthreads();
    int by0 = ty * 1;
    int bx0 = tx * 1;
    #pragma unroll
    for (int cb = 0; cb < 4; cb += 1) {
      if (oc0 + cb < 256) {
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
    if (oc < 256 && oh < 14 && ow < 14)
      l3_b2_add_out_ws[sc][oc][oh][ow] = acc[cb];
  }
}

/*LAUNCH2 196 1 1 256 1*/
__global__ void ast22_l3_b2_add_out_fused_red()
{
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= 50176) return;
  int ow = idx % 14;
  int t0 = idx / 14;
  int oh = t0 % 14;
  int oc = t0 / 14;
  float acc = l3_b2_add_out_ws[0][oc][oh][ow];
  acc += l3_b2_add_out_ws[1][oc][oh][ow];
  acc += l3_b2_add_out_ws[2][oc][oh][ow];
  acc += l3_b2_add_out_ws[3][oc][oh][ow];
  float result = acc + l3_b2_b2[oc];
  l3_b2_add_out[0][oc][oh][ow] = ((((result - l3_b2_bn2_mean[oc]) / sqrtf((l3_b2_bn2_var[oc] + bn_eps))) * l3_b2_bn2_gamma[oc] + l3_b2_bn2_beta[oc]) + l3_b1_add_out[0][oc][oh][ow]);
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

/*LAUNCH 1 1 256 8 8*/
__global__ void ast27_l4_b1_add_out_fused()
{
  __shared__ float sh[10][10];
  __shared__ float shw[4][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ow = blockIdx.x * 8 + tx;
  int oh = blockIdx.y * 8 + ty;
  int iw0 = blockIdx.x * 8 - 1;
  int ih0 = blockIdx.y * 8 - 1;
  int sc = blockIdx.z / 128;
  int oc0 = (blockIdx.z - sc * 128) * 4;
  int c0 = sc * 256;
  int c1 = c0 + 256;
  if (c1 > 512) c1 = 512;
  float acc[4];
  #pragma unroll
  for (int cb = 0; cb < 4; cb += 1) acc[cb] = 0.0f;
  for (int c = c0; c < c1; c += 1) {
    for (int sy = ty; sy < 10; sy += 8)
      for (int sx = tx; sx < 10; sx += 8) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh[sy][sx] = (iy < 0 || iy >= 7 || ix < 0 || ix >= 7)
                     ? 0.0f
                     : l4_b1_relu1_out[0][c][iy][ix];
      }
    {
      int tid = ty * 8 + tx;
      if (tid < 36) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw[cb][t] = (oc0 + cb < 512) ? l4_b1_w2[oc0 + cb][c][t / 3][t - (t / 3) * 3] : 0.0f;
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
    if (oc < 512 && oh < 7 && ow < 7)
      l4_b1_add_out_ws[sc][oc][oh][ow] = acc[cb];
  }
}

/*LAUNCH2 98 1 1 256 1*/
__global__ void ast27_l4_b1_add_out_fused_red()
{
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= 25088) return;
  int ow = idx % 7;
  int t0 = idx / 7;
  int oh = t0 % 7;
  int oc = t0 / 7;
  float acc = l4_b1_add_out_ws[0][oc][oh][ow];
  acc += l4_b1_add_out_ws[1][oc][oh][ow];
  float result = acc + l4_b1_b2[oc];
  l4_b1_add_out[0][oc][oh][ow] = ((((result - l4_b1_bn2_mean[oc]) / sqrtf((l4_b1_bn2_var[oc] + bn_eps))) * l4_b1_bn2_gamma[oc] + l4_b1_bn2_beta[oc]) + l4_b1_shortcut_out[0][oc][oh][ow]);
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

/*LAUNCH 1 1 256 8 8*/
__global__ void ast29_l4_b2_relu1_out_fused()
{
  __shared__ float sh[10][10];
  __shared__ float shw[4][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ow = blockIdx.x * 8 + tx;
  int oh = blockIdx.y * 8 + ty;
  int iw0 = blockIdx.x * 8 - 1;
  int ih0 = blockIdx.y * 8 - 1;
  int sc = blockIdx.z / 128;
  int oc0 = (blockIdx.z - sc * 128) * 4;
  int c0 = sc * 256;
  int c1 = c0 + 256;
  if (c1 > 512) c1 = 512;
  float acc[4];
  #pragma unroll
  for (int cb = 0; cb < 4; cb += 1) acc[cb] = 0.0f;
  for (int c = c0; c < c1; c += 1) {
    for (int sy = ty; sy < 10; sy += 8)
      for (int sx = tx; sx < 10; sx += 8) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh[sy][sx] = (iy < 0 || iy >= 7 || ix < 0 || ix >= 7)
                     ? 0.0f
                     : l4_b1_add_out[0][c][iy][ix];
      }
    {
      int tid = ty * 8 + tx;
      if (tid < 36) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw[cb][t] = (oc0 + cb < 512) ? l4_b2_w1[oc0 + cb][c][t / 3][t - (t / 3) * 3] : 0.0f;
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
    if (oc < 512 && oh < 7 && ow < 7)
      l4_b2_relu1_out_ws[sc][oc][oh][ow] = acc[cb];
  }
}

/*LAUNCH2 98 1 1 256 1*/
__global__ void ast29_l4_b2_relu1_out_fused_red()
{
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= 25088) return;
  int ow = idx % 7;
  int t0 = idx / 7;
  int oh = t0 % 7;
  int oc = t0 / 7;
  float acc = l4_b2_relu1_out_ws[0][oc][oh][ow];
  acc += l4_b2_relu1_out_ws[1][oc][oh][ow];
  float result = acc + l4_b2_b1[oc];
  float fused_result = (((result - l4_b2_bn1_mean[oc]) / sqrtf((l4_b2_bn1_var[oc] + bn_eps))) * l4_b2_bn1_gamma[oc] + l4_b2_bn1_beta[oc]);
  l4_b2_relu1_out[0][oc][oh][ow] = (fused_result <= 0.0f) ? 0.0f : fused_result;
}

/*LAUNCH 1 1 256 8 8*/
__global__ void ast30_l4_b2_add_out_fused()
{
  __shared__ float sh[10][10];
  __shared__ float shw[4][9];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int ow = blockIdx.x * 8 + tx;
  int oh = blockIdx.y * 8 + ty;
  int iw0 = blockIdx.x * 8 - 1;
  int ih0 = blockIdx.y * 8 - 1;
  int sc = blockIdx.z / 128;
  int oc0 = (blockIdx.z - sc * 128) * 4;
  int c0 = sc * 256;
  int c1 = c0 + 256;
  if (c1 > 512) c1 = 512;
  float acc[4];
  #pragma unroll
  for (int cb = 0; cb < 4; cb += 1) acc[cb] = 0.0f;
  for (int c = c0; c < c1; c += 1) {
    for (int sy = ty; sy < 10; sy += 8)
      for (int sx = tx; sx < 10; sx += 8) {
        int iy = ih0 + sy;
        int ix = iw0 + sx;
        sh[sy][sx] = (iy < 0 || iy >= 7 || ix < 0 || ix >= 7)
                     ? 0.0f
                     : l4_b2_relu1_out[0][c][iy][ix];
      }
    {
      int tid = ty * 8 + tx;
      if (tid < 36) {
        int cb = tid / 9;
        int t = tid - cb * 9;
        shw[cb][t] = (oc0 + cb < 512) ? l4_b2_w2[oc0 + cb][c][t / 3][t - (t / 3) * 3] : 0.0f;
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
    if (oc < 512 && oh < 7 && ow < 7)
      l4_b2_add_out_ws[sc][oc][oh][ow] = acc[cb];
  }
}

/*LAUNCH2 98 1 1 256 1*/
__global__ void ast30_l4_b2_add_out_fused_red()
{
  int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= 25088) return;
  int ow = idx % 7;
  int t0 = idx / 7;
  int oh = t0 % 7;
  int oc = t0 / 7;
  float acc = l4_b2_add_out_ws[0][oc][oh][ow];
  acc += l4_b2_add_out_ws[1][oc][oh][ow];
  float result = acc + l4_b2_b2[oc];
  l4_b2_add_out[0][oc][oh][ow] = ((((result - l4_b2_bn2_mean[oc]) / sqrtf((l4_b2_bn2_var[oc] + bn_eps))) * l4_b2_bn2_gamma[oc] + l4_b2_bn2_beta[oc]) + l4_b1_add_out[0][oc][oh][ow]);
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

/*LAUNCH 1000 1 1 256 1*/
__global__ void ast32_output_fused()
{
  __shared__ float red[256];
  int o = blockIdx.x;
  int tid = threadIdx.x;
  float acc = 0.0f;
  for (int c = tid; c < 512; c += 256) {
    float sum = 0.0f;
    for (int ii = 0; ii < 7; ii += 1)
      for (int jj = 0; jj < 7; jj += 1)
        sum += l4_b2_add_out[0][c][ii][jj];
    float avg = sum / 49.0f;
    acc += avg * fc_w[o][c];
  }
  red[tid] = acc;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    output[0][o] = red[0] + fc_b[o];
}

void resnet18_tiled()
{
  { dim3 g(7,7,16); dim3 b(16,16,1); ast0_relu1_out_fused<<<g, b>>>(); }
  ast1_pool_out<<<784, 256>>>();
  { dim3 g(2,2,32); dim3 b(16,16,1); ast2_l1_b1_relu1_out_fused<<<g, b>>>(); }
  { dim3 g(2,2,32); dim3 b(16,16,1); ast3_l1_b1_add_out_fused<<<g, b>>>(); }
  ast4_l1_b1_add_out_fused<<<784, 256>>>();
  { dim3 g(2,2,32); dim3 b(16,16,1); ast5_l1_b2_relu1_out_fused<<<g, b>>>(); }
  { dim3 g(2,2,32); dim3 b(16,16,1); ast6_l1_b2_add_out_fused<<<g, b>>>(); }
  ast7_l1_b2_add_out_fused<<<784, 256>>>();
  ast8_l2_b1_shortcut_out<<<392, 256>>>();
  ast9_l2_b1_shortcut_out_fused<<<392, 256>>>();
  { dim3 g(2,2,64); dim3 b(16,16,1); ast10_l2_b1_relu1_out_fused<<<g, b>>>(); }
  { dim3 g(392,1,1); dim3 b(256,1,1); ast10_l2_b1_relu1_out_fused_red<<<g, b>>>(); }
  { dim3 g(1,1,128); dim3 b(16,16,1); ast11_l2_b1_add_out_fused<<<g, b>>>(); }
  { dim3 g(392,1,1); dim3 b(256,1,1); ast11_l2_b1_add_out_fused_red<<<g, b>>>(); }
  ast12_l2_b1_add_out_fused<<<392, 256>>>();
  { dim3 g(1,1,128); dim3 b(16,16,1); ast13_l2_b2_relu1_out_fused<<<g, b>>>(); }
  { dim3 g(392,1,1); dim3 b(256,1,1); ast13_l2_b2_relu1_out_fused_red<<<g, b>>>(); }
  { dim3 g(1,1,128); dim3 b(16,16,1); ast14_l2_b2_add_out_fused<<<g, b>>>(); }
  { dim3 g(392,1,1); dim3 b(256,1,1); ast14_l2_b2_add_out_fused_red<<<g, b>>>(); }
  ast15_l2_b2_add_out_fused<<<392, 256>>>();
  ast16_l3_b1_shortcut_out<<<196, 256>>>();
  ast17_l3_b1_shortcut_out_fused<<<196, 256>>>();
  { dim3 g(1,1,256); dim3 b(16,16,1); ast18_l3_b1_relu1_out_fused<<<g, b>>>(); }
  { dim3 g(196,1,1); dim3 b(256,1,1); ast18_l3_b1_relu1_out_fused_red<<<g, b>>>(); }
  { dim3 g(1,1,256); dim3 b(16,16,1); ast19_l3_b1_add_out_fused<<<g, b>>>(); }
  { dim3 g(196,1,1); dim3 b(256,1,1); ast19_l3_b1_add_out_fused_red<<<g, b>>>(); }
  ast20_l3_b1_add_out_fused<<<196, 256>>>();
  { dim3 g(1,1,256); dim3 b(16,16,1); ast21_l3_b2_relu1_out_fused<<<g, b>>>(); }
  { dim3 g(196,1,1); dim3 b(256,1,1); ast21_l3_b2_relu1_out_fused_red<<<g, b>>>(); }
  { dim3 g(1,1,256); dim3 b(16,16,1); ast22_l3_b2_add_out_fused<<<g, b>>>(); }
  { dim3 g(196,1,1); dim3 b(256,1,1); ast22_l3_b2_add_out_fused_red<<<g, b>>>(); }
  ast23_l3_b2_add_out_fused<<<196, 256>>>();
  ast24_l4_b1_shortcut_out<<<98, 256>>>();
  ast25_l4_b1_shortcut_out_fused<<<98, 256>>>();
  ast26_l4_b1_relu1_out_fused<<<98, 256>>>();
  { dim3 g(1,1,256); dim3 b(8,8,1); ast27_l4_b1_add_out_fused<<<g, b>>>(); }
  { dim3 g(98,1,1); dim3 b(256,1,1); ast27_l4_b1_add_out_fused_red<<<g, b>>>(); }
  ast28_l4_b1_add_out_fused<<<98, 256>>>();
  { dim3 g(1,1,256); dim3 b(8,8,1); ast29_l4_b2_relu1_out_fused<<<g, b>>>(); }
  { dim3 g(98,1,1); dim3 b(256,1,1); ast29_l4_b2_relu1_out_fused_red<<<g, b>>>(); }
  { dim3 g(1,1,256); dim3 b(8,8,1); ast30_l4_b2_add_out_fused<<<g, b>>>(); }
  { dim3 g(98,1,1); dim3 b(256,1,1); ast30_l4_b2_add_out_fused_red<<<g, b>>>(); }
  ast31_l4_b2_add_out_fused<<<98, 256>>>();
  { dim3 g(1000,1,1); dim3 b(256,1,1); ast32_output_fused<<<g, b>>>(); }
  cudaDeviceSynchronize();
}
void resnet18_tiled_kt()
{
  cudaEvent_t eva, evb;
  cudaEventCreate(&eva);
  cudaEventCreate(&evb);
  float kt[33];
  cudaEventRecord(eva);
  { dim3 g(7,7,16); dim3 b(16,16,1); ast0_relu1_out_fused<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[0], eva, evb);
  cudaEventRecord(eva);
  ast1_pool_out<<<784, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[1], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(2,2,32); dim3 b(16,16,1); ast2_l1_b1_relu1_out_fused<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[2], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(2,2,32); dim3 b(16,16,1); ast3_l1_b1_add_out_fused<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[3], eva, evb);
  cudaEventRecord(eva);
  ast4_l1_b1_add_out_fused<<<784, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[4], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(2,2,32); dim3 b(16,16,1); ast5_l1_b2_relu1_out_fused<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[5], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(2,2,32); dim3 b(16,16,1); ast6_l1_b2_add_out_fused<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[6], eva, evb);
  cudaEventRecord(eva);
  ast7_l1_b2_add_out_fused<<<784, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[7], eva, evb);
  cudaEventRecord(eva);
  ast8_l2_b1_shortcut_out<<<392, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[8], eva, evb);
  cudaEventRecord(eva);
  ast9_l2_b1_shortcut_out_fused<<<392, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[9], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(2,2,64); dim3 b(16,16,1); ast10_l2_b1_relu1_out_fused<<<g, b>>>(); }
  { dim3 g(392,1,1); dim3 b(256,1,1); ast10_l2_b1_relu1_out_fused_red<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[10], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(1,1,128); dim3 b(16,16,1); ast11_l2_b1_add_out_fused<<<g, b>>>(); }
  { dim3 g(392,1,1); dim3 b(256,1,1); ast11_l2_b1_add_out_fused_red<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[11], eva, evb);
  cudaEventRecord(eva);
  ast12_l2_b1_add_out_fused<<<392, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[12], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(1,1,128); dim3 b(16,16,1); ast13_l2_b2_relu1_out_fused<<<g, b>>>(); }
  { dim3 g(392,1,1); dim3 b(256,1,1); ast13_l2_b2_relu1_out_fused_red<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[13], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(1,1,128); dim3 b(16,16,1); ast14_l2_b2_add_out_fused<<<g, b>>>(); }
  { dim3 g(392,1,1); dim3 b(256,1,1); ast14_l2_b2_add_out_fused_red<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[14], eva, evb);
  cudaEventRecord(eva);
  ast15_l2_b2_add_out_fused<<<392, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[15], eva, evb);
  cudaEventRecord(eva);
  ast16_l3_b1_shortcut_out<<<196, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[16], eva, evb);
  cudaEventRecord(eva);
  ast17_l3_b1_shortcut_out_fused<<<196, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[17], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(1,1,256); dim3 b(16,16,1); ast18_l3_b1_relu1_out_fused<<<g, b>>>(); }
  { dim3 g(196,1,1); dim3 b(256,1,1); ast18_l3_b1_relu1_out_fused_red<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[18], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(1,1,256); dim3 b(16,16,1); ast19_l3_b1_add_out_fused<<<g, b>>>(); }
  { dim3 g(196,1,1); dim3 b(256,1,1); ast19_l3_b1_add_out_fused_red<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[19], eva, evb);
  cudaEventRecord(eva);
  ast20_l3_b1_add_out_fused<<<196, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[20], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(1,1,256); dim3 b(16,16,1); ast21_l3_b2_relu1_out_fused<<<g, b>>>(); }
  { dim3 g(196,1,1); dim3 b(256,1,1); ast21_l3_b2_relu1_out_fused_red<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[21], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(1,1,256); dim3 b(16,16,1); ast22_l3_b2_add_out_fused<<<g, b>>>(); }
  { dim3 g(196,1,1); dim3 b(256,1,1); ast22_l3_b2_add_out_fused_red<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[22], eva, evb);
  cudaEventRecord(eva);
  ast23_l3_b2_add_out_fused<<<196, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[23], eva, evb);
  cudaEventRecord(eva);
  ast24_l4_b1_shortcut_out<<<98, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[24], eva, evb);
  cudaEventRecord(eva);
  ast25_l4_b1_shortcut_out_fused<<<98, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[25], eva, evb);
  cudaEventRecord(eva);
  ast26_l4_b1_relu1_out_fused<<<98, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[26], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(1,1,256); dim3 b(8,8,1); ast27_l4_b1_add_out_fused<<<g, b>>>(); }
  { dim3 g(98,1,1); dim3 b(256,1,1); ast27_l4_b1_add_out_fused_red<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[27], eva, evb);
  cudaEventRecord(eva);
  ast28_l4_b1_add_out_fused<<<98, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[28], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(1,1,256); dim3 b(8,8,1); ast29_l4_b2_relu1_out_fused<<<g, b>>>(); }
  { dim3 g(98,1,1); dim3 b(256,1,1); ast29_l4_b2_relu1_out_fused_red<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[29], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(1,1,256); dim3 b(8,8,1); ast30_l4_b2_add_out_fused<<<g, b>>>(); }
  { dim3 g(98,1,1); dim3 b(256,1,1); ast30_l4_b2_add_out_fused_red<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[30], eva, evb);
  cudaEventRecord(eva);
  ast31_l4_b2_add_out_fused<<<98, 256>>>();
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[31], eva, evb);
  cudaEventRecord(eva);
  { dim3 g(1000,1,1); dim3 b(256,1,1); ast32_output_fused<<<g, b>>>(); }
  cudaEventRecord(evb);
  cudaEventSynchronize(evb);
  cudaEventElapsedTime(&kt[32], eva, evb);
  printf("KTIME 0 ast0_relu1_out_fused %.6f\n", kt[0]);
  printf("KTIME 1 ast1_pool_out %.6f\n", kt[1]);
  printf("KTIME 2 ast2_l1_b1_relu1_out_fused %.6f\n", kt[2]);
  printf("KTIME 3 ast3_l1_b1_add_out_fused %.6f\n", kt[3]);
  printf("KTIME 4 ast4_l1_b1_add_out_fused %.6f\n", kt[4]);
  printf("KTIME 5 ast5_l1_b2_relu1_out_fused %.6f\n", kt[5]);
  printf("KTIME 6 ast6_l1_b2_add_out_fused %.6f\n", kt[6]);
  printf("KTIME 7 ast7_l1_b2_add_out_fused %.6f\n", kt[7]);
  printf("KTIME 8 ast8_l2_b1_shortcut_out %.6f\n", kt[8]);
  printf("KTIME 9 ast9_l2_b1_shortcut_out_fused %.6f\n", kt[9]);
  printf("KTIME 10 ast10_l2_b1_relu1_out_fused %.6f\n", kt[10]);
  printf("KTIME 11 ast11_l2_b1_add_out_fused %.6f\n", kt[11]);
  printf("KTIME 12 ast12_l2_b1_add_out_fused %.6f\n", kt[12]);
  printf("KTIME 13 ast13_l2_b2_relu1_out_fused %.6f\n", kt[13]);
  printf("KTIME 14 ast14_l2_b2_add_out_fused %.6f\n", kt[14]);
  printf("KTIME 15 ast15_l2_b2_add_out_fused %.6f\n", kt[15]);
  printf("KTIME 16 ast16_l3_b1_shortcut_out %.6f\n", kt[16]);
  printf("KTIME 17 ast17_l3_b1_shortcut_out_fused %.6f\n", kt[17]);
  printf("KTIME 18 ast18_l3_b1_relu1_out_fused %.6f\n", kt[18]);
  printf("KTIME 19 ast19_l3_b1_add_out_fused %.6f\n", kt[19]);
  printf("KTIME 20 ast20_l3_b1_add_out_fused %.6f\n", kt[20]);
  printf("KTIME 21 ast21_l3_b2_relu1_out_fused %.6f\n", kt[21]);
  printf("KTIME 22 ast22_l3_b2_add_out_fused %.6f\n", kt[22]);
  printf("KTIME 23 ast23_l3_b2_add_out_fused %.6f\n", kt[23]);
  printf("KTIME 24 ast24_l4_b1_shortcut_out %.6f\n", kt[24]);
  printf("KTIME 25 ast25_l4_b1_shortcut_out_fused %.6f\n", kt[25]);
  printf("KTIME 26 ast26_l4_b1_relu1_out_fused %.6f\n", kt[26]);
  printf("KTIME 27 ast27_l4_b1_add_out_fused %.6f\n", kt[27]);
  printf("KTIME 28 ast28_l4_b1_add_out_fused %.6f\n", kt[28]);
  printf("KTIME 29 ast29_l4_b2_relu1_out_fused %.6f\n", kt[29]);
  printf("KTIME 30 ast30_l4_b2_add_out_fused %.6f\n", kt[30]);
  printf("KTIME 31 ast31_l4_b2_add_out_fused %.6f\n", kt[31]);
  printf("KTIME 32 ast32_output_fused %.6f\n", kt[32]);
  cudaEventDestroy(eva);
  cudaEventDestroy(evb);
}
