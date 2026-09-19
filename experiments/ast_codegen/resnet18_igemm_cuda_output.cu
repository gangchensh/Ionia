
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define IG_BM 64
#define IG_BN 64
#define IG_BK 32
#define IG_NTHR 128

/* Implicit-GEMM 3x3/1x1 conv, FP32 CUDA core FMA.
 * X: HWC input [HI*WI][CI] (WI==HI, square maps).  Wt: [CO][TAPS][CI]
 * r-major.  P: partial output, block z writes P + z*outstride in CHW
 * ([CO][HO*HO]) layout.  Template constants let the compiler fold all
 * address math; CI is a power of two and CI%IG_BK==0 so every k-tile lies
 * within a single tap. */
template <int CI, int CO, int HI, int HO, int STRIDE, int TAPS>
__device__ __forceinline__ void ig_conv_dev(const float* __restrict__ X,
                                            const float* __restrict__ Wt,
                                            float* __restrict__ P,
                                            int kchunk, long outstride) {
  constexpr int N = HO * HO;
  constexpr int K = CI * TAPS;
  constexpr int PAD = (TAPS == 9) ? 1 : 0;
  __shared__ float As[IG_BK][IG_BM + 4];
  __shared__ float Bs[IG_BK][IG_BN + 4];
  const int m0 = blockIdx.y * IG_BM;
  const int n0 = blockIdx.x * IG_BN;
  const int tid = threadIdx.x;
  const int tx = tid & 15; /* n dim: 16 threads x 4 cols = 64 */
  const int ty = tid >> 4; /* m dim: 8 threads x 8 rows = 64 */
  const int kbeg = blockIdx.z * kchunk;
  const int kend = min(K, kbeg + kchunk);
  float* Out = P + (long)blockIdx.z * outstride;

  /* per-block smem LUTs for the spatial decode of n */
  __shared__ int oh_lut[IG_BN], ow_lut[IG_BN];
  __shared__ bool n_valid[IG_BN];
  if (tid < IG_BN) {
    int gn = n0 + tid;
    n_valid[tid] = gn < N;
    oh_lut[tid] = n_valid[tid] ? gn / HO : 0;
    ow_lut[tid] = n_valid[tid] ? gn % HO : 0;
  }
  __syncthreads();

  float c[8][4];
#pragma unroll
  for (int i = 0; i < 8; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) c[i][j] = 0.f;

  for (int k0 = kbeg; k0 < kend; k0 += IG_BK) {
    int r = k0 / CI, ci0 = k0 - r * CI;
    int kh = 0, kw = 0;
    if (TAPS == 9) { /* r/3, r%3 via multiply-shift (exact for r<9) */
      kh = (r * 11) >> 5;
      kw = r - 3 * kh;
    }
    for (int idx = tid; idx < IG_BM * (IG_BK / 4); idx += IG_NTHR) {
      int m = idx >> 3, q = idx & 7;
      float4 g = *(const float4*)&Wt[(long)(m0 + m) * K + k0 + q * 4];
      As[q * 4 + 0][m] = g.x;
      As[q * 4 + 1][m] = g.y;
      As[q * 4 + 2][m] = g.z;
      As[q * 4 + 3][m] = g.w;
    }
    for (int idx = tid; idx < IG_BN * (IG_BK / 4); idx += IG_NTHR) {
      int n = idx >> 3, q = idx & 7;
      int ih = oh_lut[n] * STRIDE + kh - PAD;
      int iw = ow_lut[n] * STRIDE + kw - PAD;
      float4 g = make_float4(0.f, 0.f, 0.f, 0.f);
      if (n_valid[n] && ih >= 0 && ih < HI && iw >= 0 && iw < HI)
        g = *(const float4*)&X[(long)(ih * HI + iw) * CI + ci0 + q * 4];
      Bs[q * 4 + 0][n] = g.x;
      Bs[q * 4 + 1][n] = g.y;
      Bs[q * 4 + 2][n] = g.z;
      Bs[q * 4 + 3][n] = g.w;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < IG_BK; k++) {
      float4 a0 = *(const float4*)&As[k][ty * 8];
      float4 a1 = *(const float4*)&As[k][ty * 8 + 4];
      float4 b0 = *(const float4*)&Bs[k][tx * 4];
      float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      float bv[4] = {b0.x, b0.y, b0.z, b0.w};
#pragma unroll
      for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) c[i][j] = fmaf(av[i], bv[j], c[i][j]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 8; i++)
#pragma unroll
    for (int j = 0; j < 4; j++)
      if (n_valid[tx * 4 + j])
        Out[(long)(m0 + ty * 8 + i) * N + n0 + tx * 4 + j] = c[i][j];
}

/* Ordered split-K reduce + fused epilogue: v = bias + sum_z P[z];
 * BN (eval); optional residual add (residual is HWC); optional ReLU.
 * Output HWC (out[n*CO+m]) or CHW (out[m*N+n], for the last conv). */
template <int CO, int N, int Z, bool RELU, bool HAS_RES, bool OUT_CHW>
__device__ __forceinline__ void ig_reduce_dev(const float* __restrict__ P,
                                              const float* __restrict__ bias,
                                              const float* __restrict__ mean,
                                              const float* __restrict__ var,
                                              const float* __restrict__ gamma,
                                              const float* __restrict__ beta,
                                              const float* __restrict__ res,
                                              float* __restrict__ out,
                                              float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= CO * N) return;
  int n = idx % N, m = idx / N;
  float v = bias[m];
#pragma unroll
  for (int z = 0; z < Z; z++) v += P[(long)z * CO * N + idx];
  v = (v - mean[m]) / sqrtf(var[m] + eps) * gamma[m] + beta[m];
  if (HAS_RES) v += res[(long)n * CO + m];
  if (RELU) v = (v <= 0.0f) ? 0.0f : v;
  out[OUT_CHW ? idx : (long)n * CO + m] = v;
}

/* CHW -> HWC transpose (stem pool output -> igemm input). */
__device__ __forceinline__ void ig_chw2hwc_dev(const float* in, float* out,
                                               int C, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= C * N) return;
  int n = idx % N, c = idx / N;
  out[(long)n * C + c] = in[idx];
}

/* One-time weight permutation [CO][CI][9] -> [CO][9][CI] (r-major). */
__device__ __forceinline__ void ig_wperm_dev(const float* w, float* pw,
                                             int CO, int CI) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= CO * CI * 9) return;
  int r = i % 9, t = i / 9, ci = t % CI, co = t / CI;
  pw[(long)co * 9 * CI + r * CI + ci] = w[i];
}
#define IG_WPERM(NAME, SRC, OFF, CO, CI) \
  __global__ void NAME() { \
    ig_wperm_dev((const float*)&SRC[0][0][0][0], ig_wp + OFF, CO, CI); \
  }

/* Instrumentation helper for resnet18_igemm_kt(): per-kernel min-of-30
 * (3 warmup) with CUDA events.  All kernels are idempotent (no in-place
 * accumulation), so repeated launches are safe. */
#define KT_ONE(I, NAME, ...) \
  do { \
    float best = 1e30f; \
    for (int r = 0; r < 33; r++) { \
      cudaEventRecord(__kta); \
      __VA_ARGS__; \
      cudaEventRecord(__ktb); \
      cudaEventSynchronize(__ktb); \
      float ms; cudaEventElapsedTime(&ms, __kta, __ktb); \
      if (r >= 3 && ms < best) best = ms; \
    } \
    printf("KMIN %d %s %.6f\n", I, NAME, best); \
  } while (0)

#define IG_CONV(NAME, X, W, CI, CO, HI, HO, ST, TAPS, KCHUNK) \
  __global__ void NAME() { \
    ig_conv_dev<CI, CO, HI, HO, ST, TAPS>(X, W, ig_ws, KCHUNK, \
                                          (long)(CO) * (HO) * (HO)); \
  }

#define IG_RED(NAME, CO, N, Z, RELU, HASRES, OUTCHW, BIAS, MEAN, VAR, GAMMA, \
               BETA, RES, OUT) \
  __global__ void NAME() { \
    ig_reduce_dev<CO, N, Z, RELU, HASRES, OUTCHW>(ig_ws, BIAS, MEAN, VAR, \
                                                  GAMMA, BETA, RES, OUT, \
                                                  bn_eps); \
  }
IG_WPERM(wperm_l1_b1_w1_k, l1_b1_w1, 0, 64, 64)
IG_WPERM(wperm_l1_b1_w2_k, l1_b1_w2, 36864, 64, 64)
IG_WPERM(wperm_l1_b2_w1_k, l1_b2_w1, 73728, 64, 64)
IG_WPERM(wperm_l1_b2_w2_k, l1_b2_w2, 110592, 64, 64)
IG_WPERM(wperm_l2_b1_w1_k, l2_b1_w1, 147456, 128, 64)
IG_WPERM(wperm_l2_b1_w2_k, l2_b1_w2, 221184, 128, 128)
IG_WPERM(wperm_l2_b2_w1_k, l2_b2_w1, 368640, 128, 128)
IG_WPERM(wperm_l2_b2_w2_k, l2_b2_w2, 516096, 128, 128)
IG_WPERM(wperm_l3_b1_w1_k, l3_b1_w1, 663552, 256, 128)
IG_WPERM(wperm_l3_b1_w2_k, l3_b1_w2, 958464, 256, 256)
IG_WPERM(wperm_l3_b2_w1_k, l3_b2_w1, 1548288, 256, 256)
IG_WPERM(wperm_l3_b2_w2_k, l3_b2_w2, 2138112, 256, 256)
IG_WPERM(wperm_l4_b1_w1_k, l4_b1_w1, 2727936, 512, 256)
IG_WPERM(wperm_l4_b1_w2_k, l4_b1_w2, 3907584, 512, 512)
IG_WPERM(wperm_l4_b2_w1_k, l4_b2_w1, 6266880, 512, 512)
IG_WPERM(wperm_l4_b2_w2_k, l4_b2_w2, 8626176, 512, 512)

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


__global__ void ig_transpose_k() {
  ig_chw2hwc_dev(&pool_out[0][0][0][0], ig_pool_hwc, 64, 3136);
}

IG_CONV(l1_b1_c1_k, ig_pool_hwc, ig_wp + 0, 64, 64, 56, 56, 1, 9, 160)
IG_RED(l1_b1_r1_k, 64, 3136, 4, true, false, false, l1_b1_b1,
       l1_b1_bn1_mean, l1_b1_bn1_var, l1_b1_bn1_gamma, l1_b1_bn1_beta,
       (const float*)0, ig_a1)
IG_CONV(l1_b1_c2_k, ig_a1, ig_wp + 36864, 64, 64, 56, 56, 1, 9, 160)
IG_RED(l1_b1_r2_k, 64, 3136, 4, true, true, false, l1_b1_b2,
       l1_b1_bn2_mean, l1_b1_bn2_var, l1_b1_bn2_gamma, l1_b1_bn2_beta,
       ig_pool_hwc, ig_a1)
IG_CONV(l1_b2_c1_k, ig_a1, ig_wp + 73728, 64, 64, 56, 56, 1, 9, 160)
IG_RED(l1_b2_r1_k, 64, 3136, 4, true, false, false, l1_b2_b1,
       l1_b2_bn1_mean, l1_b2_bn1_var, l1_b2_bn1_gamma, l1_b2_bn1_beta,
       (const float*)0, ig_a2)
IG_CONV(l1_b2_c2_k, ig_a2, ig_wp + 110592, 64, 64, 56, 56, 1, 9, 160)
IG_RED(l1_b2_r2_k, 64, 3136, 4, true, true, false, l1_b2_b2,
       l1_b2_bn2_mean, l1_b2_bn2_var, l1_b2_bn2_gamma, l1_b2_bn2_beta,
       ig_a1, ig_a2)
IG_CONV(l2_b1_c1_k, ig_a2, ig_wp + 147456, 64, 128, 56, 28, 2, 9, 96)
IG_RED(l2_b1_r1_k, 128, 784, 8, true, false, false, l2_b1_b1,
       l2_b1_bn1_mean, l2_b1_bn1_var, l2_b1_bn1_gamma, l2_b1_bn1_beta,
       (const float*)0, ig_a3)
IG_CONV(l2_b1_sc_k, ig_a2, (const float*)&l2_b1_shortcut_w[0][0][0][0], 64, 128, 56, 28, 2, 1, 32)
IG_RED(l2_b1_rsc_k, 128, 784, 2, false, false, false, l2_b1_shortcut_b,
       l2_b1_shortcut_mean, l2_b1_shortcut_var, l2_b1_shortcut_gamma, l2_b1_shortcut_beta,
       (const float*)0, ig_s3)
IG_CONV(l2_b1_c2_k, ig_a3, ig_wp + 221184, 128, 128, 28, 28, 1, 9, 160)
IG_RED(l2_b1_r2_k, 128, 784, 8, true, true, false, l2_b1_b2,
       l2_b1_bn2_mean, l2_b1_bn2_var, l2_b1_bn2_gamma, l2_b1_bn2_beta,
       ig_s3, ig_a3)
IG_CONV(l2_b2_c1_k, ig_a3, ig_wp + 368640, 128, 128, 28, 28, 1, 9, 160)
IG_RED(l2_b2_r1_k, 128, 784, 8, true, false, false, l2_b2_b1,
       l2_b2_bn1_mean, l2_b2_bn1_var, l2_b2_bn1_gamma, l2_b2_bn1_beta,
       (const float*)0, ig_a4)
IG_CONV(l2_b2_c2_k, ig_a4, ig_wp + 516096, 128, 128, 28, 28, 1, 9, 160)
IG_RED(l2_b2_r2_k, 128, 784, 8, true, true, false, l2_b2_b2,
       l2_b2_bn2_mean, l2_b2_bn2_var, l2_b2_bn2_gamma, l2_b2_bn2_beta,
       ig_a3, ig_a4)
IG_CONV(l3_b1_c1_k, ig_a4, ig_wp + 663552, 128, 256, 28, 14, 2, 9, 160)
IG_RED(l3_b1_r1_k, 256, 196, 8, true, false, false, l3_b1_b1,
       l3_b1_bn1_mean, l3_b1_bn1_var, l3_b1_bn1_gamma, l3_b1_bn1_beta,
       (const float*)0, ig_a5)
IG_CONV(l3_b1_sc_k, ig_a4, (const float*)&l3_b1_shortcut_w[0][0][0][0], 128, 256, 28, 14, 2, 1, 32)
IG_RED(l3_b1_rsc_k, 256, 196, 4, false, false, false, l3_b1_shortcut_b,
       l3_b1_shortcut_mean, l3_b1_shortcut_var, l3_b1_shortcut_gamma, l3_b1_shortcut_beta,
       (const float*)0, ig_s5)
IG_CONV(l3_b1_c2_k, ig_a5, ig_wp + 958464, 256, 256, 14, 14, 1, 9, 160)
IG_RED(l3_b1_r2_k, 256, 196, 16, true, true, false, l3_b1_b2,
       l3_b1_bn2_mean, l3_b1_bn2_var, l3_b1_bn2_gamma, l3_b1_bn2_beta,
       ig_s5, ig_a5)
IG_CONV(l3_b2_c1_k, ig_a5, ig_wp + 1548288, 256, 256, 14, 14, 1, 9, 160)
IG_RED(l3_b2_r1_k, 256, 196, 16, true, false, false, l3_b2_b1,
       l3_b2_bn1_mean, l3_b2_bn1_var, l3_b2_bn1_gamma, l3_b2_bn1_beta,
       (const float*)0, ig_a6)
IG_CONV(l3_b2_c2_k, ig_a6, ig_wp + 2138112, 256, 256, 14, 14, 1, 9, 160)
IG_RED(l3_b2_r2_k, 256, 196, 16, true, true, false, l3_b2_b2,
       l3_b2_bn2_mean, l3_b2_bn2_var, l3_b2_bn2_gamma, l3_b2_bn2_beta,
       ig_a5, ig_a6)
IG_CONV(l4_b1_c1_k, ig_a6, ig_wp + 2727936, 256, 512, 14, 7, 2, 9, 160)
IG_RED(l4_b1_r1_k, 512, 49, 16, true, false, false, l4_b1_b1,
       l4_b1_bn1_mean, l4_b1_bn1_var, l4_b1_bn1_gamma, l4_b1_bn1_beta,
       (const float*)0, ig_a7)
IG_CONV(l4_b1_sc_k, ig_a6, (const float*)&l4_b1_shortcut_w[0][0][0][0], 256, 512, 14, 7, 2, 1, 32)
IG_RED(l4_b1_rsc_k, 512, 49, 8, false, false, false, l4_b1_shortcut_b,
       l4_b1_shortcut_mean, l4_b1_shortcut_var, l4_b1_shortcut_gamma, l4_b1_shortcut_beta,
       (const float*)0, ig_s7)
IG_CONV(l4_b1_c2_k, ig_a7, ig_wp + 3907584, 512, 512, 7, 7, 1, 9, 160)
IG_RED(l4_b1_r2_k, 512, 49, 32, true, true, false, l4_b1_b2,
       l4_b1_bn2_mean, l4_b1_bn2_var, l4_b1_bn2_gamma, l4_b1_bn2_beta,
       ig_s7, ig_a7)
IG_CONV(l4_b2_c1_k, ig_a7, ig_wp + 6266880, 512, 512, 7, 7, 1, 9, 160)
IG_RED(l4_b2_r1_k, 512, 49, 32, true, false, false, l4_b2_b1,
       l4_b2_bn1_mean, l4_b2_bn1_var, l4_b2_bn1_gamma, l4_b2_bn1_beta,
       (const float*)0, ig_a8)
IG_CONV(l4_b2_c2_k, ig_a8, ig_wp + 8626176, 512, 512, 7, 7, 1, 9, 160)
IG_RED(l4_b2_r2_k, 512, 49, 32, true, true, true, l4_b2_b2,
       l4_b2_bn2_mean, l4_b2_bn2_var, l4_b2_bn2_gamma, l4_b2_bn2_beta,
       ig_a7, &l4_b2_add_out[0][0][0][0])
__global__ void ig_gap_k() {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= 512) return;
  float s = 0.0f;
  for (int i = 0; i < 7; i++)
    for (int j = 0; j < 7; j++)
      s += l4_b2_add_out[0][c][i][j];
  ig_ws[c] = s / 49.0f;
}

__global__ void ig_fc_k() {
  int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (wid >= 1000) return;
  const float* avg = ig_ws;
  float a = 0.0f;
  for (int c = lane; c < 512; c += 32)
    a += avg[c] * fc_w[wid][c];
  for (int off = 16; off > 0; off >>= 1)
    a += __shfl_down_sync(0xffffffffu, a, off);
  if (lane == 0) output[0][wid] = a + fc_b[wid];
}

static void ig_init(cudaStream_t st) {
  static int done = 0;
  if (done) return;
  done = 1;
  wperm_l1_b1_w1_k<<<144, 256, 0, st>>>();
  wperm_l1_b1_w2_k<<<144, 256, 0, st>>>();
  wperm_l1_b2_w1_k<<<144, 256, 0, st>>>();
  wperm_l1_b2_w2_k<<<144, 256, 0, st>>>();
  wperm_l2_b1_w1_k<<<288, 256, 0, st>>>();
  wperm_l2_b1_w2_k<<<576, 256, 0, st>>>();
  wperm_l2_b2_w1_k<<<576, 256, 0, st>>>();
  wperm_l2_b2_w2_k<<<576, 256, 0, st>>>();
  wperm_l3_b1_w1_k<<<1152, 256, 0, st>>>();
  wperm_l3_b1_w2_k<<<2304, 256, 0, st>>>();
  wperm_l3_b2_w1_k<<<2304, 256, 0, st>>>();
  wperm_l3_b2_w2_k<<<2304, 256, 0, st>>>();
  wperm_l4_b1_w1_k<<<4608, 256, 0, st>>>();
  wperm_l4_b1_w2_k<<<9216, 256, 0, st>>>();
  wperm_l4_b2_w1_k<<<9216, 256, 0, st>>>();
  wperm_l4_b2_w2_k<<<9216, 256, 0, st>>>();
}

static void resnet18_igemm_chain(cudaStream_t __st) {
  { dim3 g(7, 7, 16); dim3 b(16, 16, 1); ast0_relu1_out_fused<<<g, b, 0, __st>>>(); }
  ast1_pool_out<<<784, 256, 0, __st>>>();
  ig_transpose_k<<<784, 256, 0, __st>>>();
  { dim3 g(49, 1, 4); l1_b1_c1_k<<<g, 128, 0, __st>>>(); }
  l1_b1_r1_k<<<784, 256, 0, __st>>>();
  { dim3 g(49, 1, 4); l1_b1_c2_k<<<g, 128, 0, __st>>>(); }
  l1_b1_r2_k<<<784, 256, 0, __st>>>();
  { dim3 g(49, 1, 4); l1_b2_c1_k<<<g, 128, 0, __st>>>(); }
  l1_b2_r1_k<<<784, 256, 0, __st>>>();
  { dim3 g(49, 1, 4); l1_b2_c2_k<<<g, 128, 0, __st>>>(); }
  l1_b2_r2_k<<<784, 256, 0, __st>>>();
  { dim3 g(13, 2, 8); l2_b1_c1_k<<<g, 128, 0, __st>>>(); }
  l2_b1_r1_k<<<392, 256, 0, __st>>>();
  { dim3 g(13, 2, 2); l2_b1_sc_k<<<g, 128, 0, __st>>>(); }
  l2_b1_rsc_k<<<392, 256, 0, __st>>>();
  { dim3 g(13, 2, 8); l2_b1_c2_k<<<g, 128, 0, __st>>>(); }
  l2_b1_r2_k<<<392, 256, 0, __st>>>();
  { dim3 g(13, 2, 8); l2_b2_c1_k<<<g, 128, 0, __st>>>(); }
  l2_b2_r1_k<<<392, 256, 0, __st>>>();
  { dim3 g(13, 2, 8); l2_b2_c2_k<<<g, 128, 0, __st>>>(); }
  l2_b2_r2_k<<<392, 256, 0, __st>>>();
  { dim3 g(4, 4, 8); l3_b1_c1_k<<<g, 128, 0, __st>>>(); }
  l3_b1_r1_k<<<196, 256, 0, __st>>>();
  { dim3 g(4, 4, 4); l3_b1_sc_k<<<g, 128, 0, __st>>>(); }
  l3_b1_rsc_k<<<196, 256, 0, __st>>>();
  { dim3 g(4, 4, 16); l3_b1_c2_k<<<g, 128, 0, __st>>>(); }
  l3_b1_r2_k<<<196, 256, 0, __st>>>();
  { dim3 g(4, 4, 16); l3_b2_c1_k<<<g, 128, 0, __st>>>(); }
  l3_b2_r1_k<<<196, 256, 0, __st>>>();
  { dim3 g(4, 4, 16); l3_b2_c2_k<<<g, 128, 0, __st>>>(); }
  l3_b2_r2_k<<<196, 256, 0, __st>>>();
  { dim3 g(1, 8, 16); l4_b1_c1_k<<<g, 128, 0, __st>>>(); }
  l4_b1_r1_k<<<98, 256, 0, __st>>>();
  { dim3 g(1, 8, 8); l4_b1_sc_k<<<g, 128, 0, __st>>>(); }
  l4_b1_rsc_k<<<98, 256, 0, __st>>>();
  { dim3 g(1, 8, 32); l4_b1_c2_k<<<g, 128, 0, __st>>>(); }
  l4_b1_r2_k<<<98, 256, 0, __st>>>();
  { dim3 g(1, 8, 32); l4_b2_c1_k<<<g, 128, 0, __st>>>(); }
  l4_b2_r1_k<<<98, 256, 0, __st>>>();
  { dim3 g(1, 8, 32); l4_b2_c2_k<<<g, 128, 0, __st>>>(); }
  l4_b2_r2_k<<<98, 256, 0, __st>>>();
  ig_gap_k<<<2, 256, 0, __st>>>();
  ig_fc_k<<<125, 256, 0, __st>>>();
}

void resnet18_igemm_s(cudaStream_t __st) {
  ig_init(__st);
  resnet18_igemm_chain(__st);
}

void resnet18_igemm_seq_s(cudaStream_t __st) {
  ig_init(__st);
  resnet18_igemm_chain(__st);
}

void resnet18_igemm() {
  resnet18_igemm_s(0);
}

void resnet18_igemm_kt() {
  cudaEvent_t __kta, __ktb;
  cudaEventCreate(&__kta);
  cudaEventCreate(&__ktb);
  resnet18_igemm();
  KT_ONE(0, "ast0_relu1_out_fused", { dim3 g(7, 7, 16); dim3 b(16, 16, 1); ast0_relu1_out_fused<<<g, b>>>(); });
  KT_ONE(1, "ast1_pool_out", ast1_pool_out<<<784, 256>>>(););
  KT_ONE(2, "ig_transpose", ig_transpose_k<<<784, 256>>>(););
  KT_ONE(3, "l1_b1_c1", { dim3 g(49, 1, 4); l1_b1_c1_k<<<g, 128>>>(); });
  KT_ONE(4, "l1_b1_r1", l1_b1_r1_k<<<784, 256>>>(););
  KT_ONE(5, "l1_b1_c2", { dim3 g(49, 1, 4); l1_b1_c2_k<<<g, 128>>>(); });
  KT_ONE(6, "l1_b1_r2", l1_b1_r2_k<<<784, 256>>>(););
  KT_ONE(7, "l1_b2_c1", { dim3 g(49, 1, 4); l1_b2_c1_k<<<g, 128>>>(); });
  KT_ONE(8, "l1_b2_r1", l1_b2_r1_k<<<784, 256>>>(););
  KT_ONE(9, "l1_b2_c2", { dim3 g(49, 1, 4); l1_b2_c2_k<<<g, 128>>>(); });
  KT_ONE(10, "l1_b2_r2", l1_b2_r2_k<<<784, 256>>>(););
  KT_ONE(11, "l2_b1_c1", { dim3 g(13, 2, 8); l2_b1_c1_k<<<g, 128>>>(); });
  KT_ONE(12, "l2_b1_r1", l2_b1_r1_k<<<392, 256>>>(););
  KT_ONE(13, "l2_b1_sc", { dim3 g(13, 2, 2); l2_b1_sc_k<<<g, 128>>>(); });
  KT_ONE(14, "l2_b1_rsc", l2_b1_rsc_k<<<392, 256>>>(););
  KT_ONE(15, "l2_b1_c2", { dim3 g(13, 2, 8); l2_b1_c2_k<<<g, 128>>>(); });
  KT_ONE(16, "l2_b1_r2", l2_b1_r2_k<<<392, 256>>>(););
  KT_ONE(17, "l2_b2_c1", { dim3 g(13, 2, 8); l2_b2_c1_k<<<g, 128>>>(); });
  KT_ONE(18, "l2_b2_r1", l2_b2_r1_k<<<392, 256>>>(););
  KT_ONE(19, "l2_b2_c2", { dim3 g(13, 2, 8); l2_b2_c2_k<<<g, 128>>>(); });
  KT_ONE(20, "l2_b2_r2", l2_b2_r2_k<<<392, 256>>>(););
  KT_ONE(21, "l3_b1_c1", { dim3 g(4, 4, 8); l3_b1_c1_k<<<g, 128>>>(); });
  KT_ONE(22, "l3_b1_r1", l3_b1_r1_k<<<196, 256>>>(););
  KT_ONE(23, "l3_b1_sc", { dim3 g(4, 4, 4); l3_b1_sc_k<<<g, 128>>>(); });
  KT_ONE(24, "l3_b1_rsc", l3_b1_rsc_k<<<196, 256>>>(););
  KT_ONE(25, "l3_b1_c2", { dim3 g(4, 4, 16); l3_b1_c2_k<<<g, 128>>>(); });
  KT_ONE(26, "l3_b1_r2", l3_b1_r2_k<<<196, 256>>>(););
  KT_ONE(27, "l3_b2_c1", { dim3 g(4, 4, 16); l3_b2_c1_k<<<g, 128>>>(); });
  KT_ONE(28, "l3_b2_r1", l3_b2_r1_k<<<196, 256>>>(););
  KT_ONE(29, "l3_b2_c2", { dim3 g(4, 4, 16); l3_b2_c2_k<<<g, 128>>>(); });
  KT_ONE(30, "l3_b2_r2", l3_b2_r2_k<<<196, 256>>>(););
  KT_ONE(31, "l4_b1_c1", { dim3 g(1, 8, 16); l4_b1_c1_k<<<g, 128>>>(); });
  KT_ONE(32, "l4_b1_r1", l4_b1_r1_k<<<98, 256>>>(););
  KT_ONE(33, "l4_b1_sc", { dim3 g(1, 8, 8); l4_b1_sc_k<<<g, 128>>>(); });
  KT_ONE(34, "l4_b1_rsc", l4_b1_rsc_k<<<98, 256>>>(););
  KT_ONE(35, "l4_b1_c2", { dim3 g(1, 8, 32); l4_b1_c2_k<<<g, 128>>>(); });
  KT_ONE(36, "l4_b1_r2", l4_b1_r2_k<<<98, 256>>>(););
  KT_ONE(37, "l4_b2_c1", { dim3 g(1, 8, 32); l4_b2_c1_k<<<g, 128>>>(); });
  KT_ONE(38, "l4_b2_r1", l4_b2_r1_k<<<98, 256>>>(););
  KT_ONE(39, "l4_b2_c2", { dim3 g(1, 8, 32); l4_b2_c2_k<<<g, 128>>>(); });
  KT_ONE(40, "l4_b2_r2", l4_b2_r2_k<<<98, 256>>>(););
  KT_ONE(41, "ig_gap", ig_gap_k<<<2, 256>>>(););
  KT_ONE(42, "ig_fc", ig_fc_k<<<125, 256>>>(););
  cudaEventDestroy(__kta);
  cudaEventDestroy(__ktb);
}
