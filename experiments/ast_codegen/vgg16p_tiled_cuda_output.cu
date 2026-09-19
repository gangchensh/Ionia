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
