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
__constant__ float (*l1_b1_add_out)[64][56][56];
__constant__ float (*l1_b2_add_out)[64][56][56];
__constant__ float (*l2_b1_ws)[128][28][28];
__constant__ float (*l2_b1_add_out)[128][28][28];
__constant__ float (*l2_b2_ws)[128][28][28];
__constant__ float (*l2_b2_add_out)[128][28][28];
__constant__ float (*l3_b1_ws)[256][14][14];
__constant__ float (*l3_b1_add_out)[256][14][14];
__constant__ float (*l3_b2_ws)[256][14][14];
__constant__ float (*l3_b2_add_out)[256][14][14];
__constant__ float (*l4_b1_ws)[512][7][7];
__constant__ float (*l4_b1_add_out)[512][7][7];
__constant__ float (*l4_b2_ws)[512][7][7];
__constant__ float (*l4_b2_add_out)[512][7][7];
__constant__ float bn_eps;

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math.h>
#include <stdio.h>


/* ---- fused BasicBlock kernel (E4) -----------------------------------
 * One kernel computes conv1+bn1+relu+conv2 (+ partial sums) of a whole
 * BasicBlock.  Grid: (ceil(M/TILE), ceil(M/TILE), SPLIT).  1024 threads.
 *
 * Phase 1: T1 = relu(bn1(conv1(in))) for the (TILE+2)^2 halo tile and the
 *          block's CG = Cout/SPLIT channel slice, resident in shared
 *          memory.  T1 positions outside the M x M map are conv2's zero
 *          padding.  The input halo tile and the w1 slices are staged in
 *          chunks of CI channels (one __syncthreads pair per chunk).
 *          Thread mapping: sp = tid & 127 (spatial slot), cg = tid >> 7
 *          (channel group); no integer division inside the ic loop.
 * Phase 2: conv2 over the resident CG channels, ic-outer accumulation
 *          into per-thread output registers (sp2 = tid & (TILE*TILE-1)
 *          pixel, cg2 = tid / (TILE*TILE) channel group).  Per ic step the
 *          w2[:,ic,:,:] slice (COUT*9 floats) is staged into shared.
 * Epilogue: SPLIT==1 (identity blocks): bn2 + identity add + relu, single
 *          global write.  SPLIT>1: raw partial sums to ws; the reduce
 *          kernel below applies bn2 + shortcut + relu.
 */
#define FB_NTHR 1024

template<int CIN, int COUT, int M, int STRIDE, int SPLIT, int TILE, int CI, int CW>
__device__ void fused_block_dev(
    const float* __restrict__ in,
    const float* __restrict__ w1, const float* __restrict__ b1,
    const float* __restrict__ m1, const float* __restrict__ v1,
    const float* __restrict__ g1, const float* __restrict__ be1,
    const float* __restrict__ w2, const float* __restrict__ b2,
    const float* __restrict__ m2, const float* __restrict__ v2,
    const float* __restrict__ g2, const float* __restrict__ be2,
    float* __restrict__ out, float* __restrict__ ws,
    float* __restrict__ smem)
{
  constexpr int CG  = COUT / SPLIT;              /* T1 channels per block */
  constexpr int MIN = M * STRIDE;                /* input spatial size */
  constexpr int TH  = TILE + 2;                  /* T1 halo tile */
  constexpr int IH  = (TH - 1) * STRIDE + 3;     /* input halo tile */
  constexpr int KC1 = CG / (FB_NTHR / 128);      /* phase-1 ch per thread */
  constexpr int SP2 = TILE * TILE;
  constexpr int SG2 = FB_NTHR / SP2;             /* phase-2 oc groups */
  constexpr int KC2 = COUT / SG2;                /* phase-2 oc per thread */
  static_assert(SPLIT > 1 || CIN == COUT,
                "SPLIT==1 epilogue assumes an identity block");
  static_assert(CIN % CI == 0, "chunked staging needs CIN % CI == 0");
  static_assert(CG % CW == 0, "w2 staging needs CG % CW == 0");
  float* shT1 = smem;                        /* CG*TH*TH */
  float* shIn = shT1 + CG * TH * TH;         /* CI*IH*IH */
  float* shW  = shIn + CI * IH * IH;         /* max(CI*CG*9, CW*COUT*9) */
  const int tid = threadIdx.x;
  const int tx0 = blockIdx.x * TILE;
  const int ty0 = blockIdx.y * TILE;
  const int split = blockIdx.z;

  /* Phase 1: T1 = relu(bn1(conv1)) into shared */
  const int sp = tid & 127;
  const int cg = tid >> 7;
  const int py = sp / TH, px = sp % TH;
  const bool act1 = (sp < TH * TH);
  float acc[KC1];
  #pragma unroll
  for (int k = 0; k < KC1; k++) acc[k] = 0.0f;
  for (int ic0 = 0; ic0 < CIN; ic0 += CI) {
    for (int s = tid; s < CI * IH * IH; s += FB_NTHR) {
      int ci = s / (IH * IH);
      int r = s - ci * (IH * IH);
      int sy = r / IH, sx = r % IH;
      int iy = (ty0 - 1) * STRIDE - 1 + sy;
      int ix = (tx0 - 1) * STRIDE - 1 + sx;
      int ic = ic0 + ci;
      shIn[s] = (iy < 0 || iy >= MIN || ix < 0 || ix >= MIN)
                ? 0.0f : in[(ic * MIN + iy) * MIN + ix];
    }
    for (int s = tid; s < CI * CG * 9; s += FB_NTHR) {
      int ci = s / (CG * 9);
      int r = s - ci * (CG * 9);
      int c = r / 9, t = r - c * 9;
      shW[s] = w1[((split * CG + c) * CIN + ic0 + ci) * 9 + t];
    }
    __syncthreads();
    if (act1) {
      #pragma unroll
      for (int ci = 0; ci < CI; ci++) {
        #pragma unroll
        for (int k = 0; k < KC1; k++) {
          const float* wp = shW + ci * CG * 9 + (cg * KC1 + k) * 9;
          const float* in0 = shIn + ci * IH * IH
                             + py * STRIDE * IH + px * STRIDE;
          float a = acc[k];
          #pragma unroll
          for (int ky = 0; ky < 3; ky++)
            #pragma unroll
            for (int kx = 0; kx < 3; kx++)
              a += in0[ky * IH + kx] * wp[ky * 3 + kx];
          acc[k] = a;
        }
      }
    }
    __syncthreads();
  }
  if (act1) {
    int gy = ty0 - 1 + py, gx = tx0 - 1 + px;
    bool inmap = (gy >= 0 && gy < M && gx >= 0 && gx < M);
    #pragma unroll
    for (int k = 0; k < KC1; k++) {
      int c = split * CG + cg * KC1 + k;
      float v = acc[k] + b1[c];
      v = (v - m1[c]) / sqrtf(v1[c] + bn_eps) * g1[c] + be1[c];
      v = (v <= 0.0f) ? 0.0f : v;
      /* outside the M x M map T1 is conv2's zero padding */
      shT1[(cg * KC1 + k) * TH * TH + py * TH + px] = inmap ? v : 0.0f;
    }
  }
  __syncthreads();

  /* Phase 2: conv2 partial over the resident CG channels; the w2 slices
   * are staged CW input channels at a time (one __syncthreads pair per CW
   * ic steps). */
  const int sp2 = tid & (SP2 - 1);
  const int cg2 = tid / SP2;
  const int y = sp2 / TILE, x = sp2 % TILE;
  float acc2[KC2];
  #pragma unroll
  for (int k = 0; k < KC2; k++) acc2[k] = 0.0f;
  for (int ic0 = 0; ic0 < CG; ic0 += CW) {
    for (int s = tid; s < CW * COUT * 9; s += FB_NTHR) {
      int ci = s / (COUT * 9);
      int r = s - ci * (COUT * 9);
      int c = r / 9, t = r - c * 9;
      shW[s] = w2[(c * COUT + split * CG + ic0 + ci) * 9 + t];
    }
    __syncthreads();
    #pragma unroll
    for (int ci = 0; ci < CW; ci++) {
      /* preload the 3x3 T1 neighbourhood once per ic (invariant across the
       * KC2 output channels), then only the weights come from shared */
      float tv[3][3];
      {
        const float* t1 = shT1 + (ic0 + ci) * TH * TH + y * TH + x;
        #pragma unroll
        for (int ky = 0; ky < 3; ky++)
          #pragma unroll
          for (int kx = 0; kx < 3; kx++)
            tv[ky][kx] = t1[ky * TH + kx];
      }
      #pragma unroll
      for (int k = 0; k < KC2; k++) {
        const float* wp = shW + ci * (COUT * 9) + (cg2 * KC2 + k) * 9;
        float a = acc2[k];
        #pragma unroll
        for (int ky = 0; ky < 3; ky++)
          #pragma unroll
          for (int kx = 0; kx < 3; kx++)
            a += tv[ky][kx] * wp[ky * 3 + kx];
        acc2[k] = a;
      }
    }
    __syncthreads();
  }
  const int gy = ty0 + y, gx = tx0 + x;
  const bool outok = (gy < M && gx < M);   /* no early return: callers may
                                              grid.sync() after us */
  if (SPLIT == 1) {
    /* identity block: full epilogue, single global write */
    if (outok) {
      #pragma unroll
      for (int k = 0; k < KC2; k++) {
        int oc = cg2 * KC2 + k;
        float v = acc2[k] + b2[oc];
        v = (v - m2[oc]) / sqrtf(v2[oc] + bn_eps) * g2[oc] + be2[oc];
        v += in[(oc * M + gy) * M + gx];
        out[(oc * M + gy) * M + gx] = (v <= 0.0f) ? 0.0f : v;
      }
    }
  } else {
    if (outok) {
      #pragma unroll
      for (int k = 0; k < KC2; k++) {
        int oc = cg2 * KC2 + k;
        ws[((split * COUT + oc) * M + gy) * M + gx] = acc2[k];
      }
    }
  }
}

/* Reduce + epilogue for SPLIT>1 blocks: sum the partials, bn2, shortcut
 * (identity, or fused 1x1 stride-2 Conv + BN for projection blocks read
 * straight from the block input), relu. */
template<int CIN, int COUT, int M, int SPLIT, int PROJ>
__device__ void fused_block_reduce_dev(
    const float* __restrict__ ws, const float* __restrict__ in,
    const float* __restrict__ b2,
    const float* __restrict__ m2, const float* __restrict__ v2,
    const float* __restrict__ g2, const float* __restrict__ be2,
    const float* __restrict__ sw, const float* __restrict__ sb,
    const float* __restrict__ sm, const float* __restrict__ sv,
    const float* __restrict__ sg, const float* __restrict__ sbe,
    float* __restrict__ out)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= COUT * M * M) return;
  int x = idx % M;
  int r = idx / M;
  int y = r % M;
  int oc = r / M;
  float v = b2[oc];
  #pragma unroll
  for (int g = 0; g < SPLIT; g++)
    v += ws[((g * COUT + oc) * M + y) * M + x];
  v = (v - m2[oc]) / sqrtf(v2[oc] + bn_eps) * g2[oc] + be2[oc];
  float sc;
  if (PROJ) {
    constexpr int MIN = 2 * M;
    float a = sb[oc];
    for (int ic = 0; ic < CIN; ic++)
      a += in[(ic * MIN + 2 * y) * MIN + 2 * x] * sw[oc * CIN + ic];
    sc = (a - sm[oc]) / sqrtf(sv[oc] + bn_eps) * sg[oc] + sbe[oc];
  } else {
    sc = in[idx];
  }
  v += sc;
  out[idx] = (v <= 0.0f) ? 0.0f : v;
}

/* Grid-stride variant of the reduce+epilogue for use inside a cooperative
 * kernel after this_grid().sync(): identical per-element math, but the
 * element space is walked by the whole grid instead of a dedicated launch. */
template<int CIN, int COUT, int M, int SPLIT, int PROJ>
__device__ void fused_block_reduce_grid(
    const float* __restrict__ ws, const float* __restrict__ in,
    const float* __restrict__ b2,
    const float* __restrict__ m2, const float* __restrict__ v2,
    const float* __restrict__ g2, const float* __restrict__ be2,
    const float* __restrict__ sw, const float* __restrict__ sb,
    const float* __restrict__ sm, const float* __restrict__ sv,
    const float* __restrict__ sg, const float* __restrict__ sbe,
    float* __restrict__ out)
{
  const int gth = gridDim.x * gridDim.y * gridDim.z * blockDim.x;
  int gtid = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x
              + blockIdx.x) * blockDim.x + threadIdx.x;
  for (int idx = gtid; idx < COUT * M * M; idx += gth) {
    int x = idx % M;
    int r = idx / M;
    int y = r % M;
    int oc = r / M;
    float v = b2[oc];
    #pragma unroll
    for (int g = 0; g < SPLIT; g++)
      v += ws[((g * COUT + oc) * M + y) * M + x];
    v = (v - m2[oc]) / sqrtf(v2[oc] + bn_eps) * g2[oc] + be2[oc];
    float sc;
    if (PROJ) {
      constexpr int MIN = 2 * M;
      float a = sb[oc];
      for (int ic = 0; ic < CIN; ic++)
        a += in[(ic * MIN + 2 * y) * MIN + 2 * x] * sw[oc * CIN + ic];
      sc = (a - sm[oc]) / sqrtf(sv[oc] + bn_eps) * sg[oc] + sbe[oc];
    } else {
      sc = in[idx];
    }
    v += sc;
    out[idx] = (v <= 0.0f) ? 0.0f : v;
  }
}

/* Instrumentation helper for resnet18_fused_kt(): per-kernel min-of-30
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

/* Same maxpool as a grid-stride phase inside the cooperative stage-1
 * kernel (round-2 launch merging): identical per-element math. */
__device__ void pool_phase_grid()
{
  const int N = 200704;
  const int gth = gridDim.x * gridDim.y * gridDim.z * blockDim.x;
  const int gtid = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x
                    + blockIdx.x) * blockDim.x + threadIdx.x;
  for (int idx = gtid; idx < N; idx += gth) {
    int t = idx;
    int i3 = t % 56; t /= 56;
    int i2 = t % 56; t /= 56;
    int i1 = t % 64; t /= 64;
    int i0 = t;
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
}

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
__global__ void __launch_bounds__(1024) l1_b1_fused_k()
{
  extern __shared__ float smem[];
  fused_block_dev<64,64,56,1,1,8,16,4>(
      &pool_out[0][0][0][0],
      &l1_b1_w1[0][0][0][0], l1_b1_b1,
      l1_b1_bn1_mean, l1_b1_bn1_var, l1_b1_bn1_gamma, l1_b1_bn1_beta,
      &l1_b1_w2[0][0][0][0], l1_b1_b2,
      l1_b1_bn2_mean, l1_b1_bn2_var, l1_b1_bn2_gamma, l1_b1_bn2_beta,
      &l1_b1_add_out[0][0][0][0], (float*)0, smem);
}

__global__ void __launch_bounds__(1024) l1_b2_fused_k()
{
  extern __shared__ float smem[];
  fused_block_dev<64,64,56,1,1,8,16,4>(
      &l1_b1_add_out[0][0][0][0],
      &l1_b2_w1[0][0][0][0], l1_b2_b1,
      l1_b2_bn1_mean, l1_b2_bn1_var, l1_b2_bn1_gamma, l1_b2_bn1_beta,
      &l1_b2_w2[0][0][0][0], l1_b2_b2,
      l1_b2_bn2_mean, l1_b2_bn2_var, l1_b2_bn2_gamma, l1_b2_bn2_beta,
      &l1_b2_add_out[0][0][0][0], (float*)0, smem);
}

__global__ void __launch_bounds__(1024) l2_b1_fused_k()
{
  extern __shared__ float smem[];
  fused_block_dev<64,128,28,2,2,8,16,4>(
      &l1_b2_add_out[0][0][0][0],
      &l2_b1_w1[0][0][0][0], l2_b1_b1,
      l2_b1_bn1_mean, l2_b1_bn1_var, l2_b1_bn1_gamma, l2_b1_bn1_beta,
      &l2_b1_w2[0][0][0][0], l2_b1_b2,
      l2_b1_bn2_mean, l2_b1_bn2_var, l2_b1_bn2_gamma, l2_b1_bn2_beta,
      &l2_b1_add_out[0][0][0][0], &l2_b1_ws[0][0][0][0], smem);
}

__global__ void l2_b1_reduce_k()
{
  fused_block_reduce_dev<64,128,28,2,1>(
      &l2_b1_ws[0][0][0][0], &l1_b2_add_out[0][0][0][0],
      l2_b1_b2,
      l2_b1_bn2_mean, l2_b1_bn2_var, l2_b1_bn2_gamma, l2_b1_bn2_beta,
      &l2_b1_shortcut_w[0][0][0][0], l2_b1_shortcut_b,
      l2_b1_shortcut_mean, l2_b1_shortcut_var, l2_b1_shortcut_gamma, l2_b1_shortcut_beta,
      &l2_b1_add_out[0][0][0][0]);
}

__global__ void __launch_bounds__(1024) l2_b2_fused_k()
{
  extern __shared__ float smem[];
  fused_block_dev<128,128,28,1,2,8,16,4>(
      &l2_b1_add_out[0][0][0][0],
      &l2_b2_w1[0][0][0][0], l2_b2_b1,
      l2_b2_bn1_mean, l2_b2_bn1_var, l2_b2_bn1_gamma, l2_b2_bn1_beta,
      &l2_b2_w2[0][0][0][0], l2_b2_b2,
      l2_b2_bn2_mean, l2_b2_bn2_var, l2_b2_bn2_gamma, l2_b2_bn2_beta,
      &l2_b2_add_out[0][0][0][0], &l2_b2_ws[0][0][0][0], smem);
}

__global__ void l2_b2_reduce_k()
{
  fused_block_reduce_dev<128,128,28,2,0>(
      &l2_b2_ws[0][0][0][0], &l2_b1_add_out[0][0][0][0],
      l2_b2_b2,
      l2_b2_bn2_mean, l2_b2_bn2_var, l2_b2_bn2_gamma, l2_b2_bn2_beta,
      l2_b2_b2, l2_b2_b2, l2_b2_b2, l2_b2_b2, l2_b2_b2, l2_b2_b2,
      &l2_b2_add_out[0][0][0][0]);
}

__global__ void __launch_bounds__(1024) l3_b1_fused_k()
{
  extern __shared__ float smem[];
  fused_block_dev<128,256,14,2,8,8,16,4>(
      &l2_b2_add_out[0][0][0][0],
      &l3_b1_w1[0][0][0][0], l3_b1_b1,
      l3_b1_bn1_mean, l3_b1_bn1_var, l3_b1_bn1_gamma, l3_b1_bn1_beta,
      &l3_b1_w2[0][0][0][0], l3_b1_b2,
      l3_b1_bn2_mean, l3_b1_bn2_var, l3_b1_bn2_gamma, l3_b1_bn2_beta,
      &l3_b1_add_out[0][0][0][0], &l3_b1_ws[0][0][0][0], smem);
}

__global__ void l3_b1_reduce_k()
{
  fused_block_reduce_dev<128,256,14,8,1>(
      &l3_b1_ws[0][0][0][0], &l2_b2_add_out[0][0][0][0],
      l3_b1_b2,
      l3_b1_bn2_mean, l3_b1_bn2_var, l3_b1_bn2_gamma, l3_b1_bn2_beta,
      &l3_b1_shortcut_w[0][0][0][0], l3_b1_shortcut_b,
      l3_b1_shortcut_mean, l3_b1_shortcut_var, l3_b1_shortcut_gamma, l3_b1_shortcut_beta,
      &l3_b1_add_out[0][0][0][0]);
}

__global__ void __launch_bounds__(1024) l3_b2_fused_k()
{
  extern __shared__ float smem[];
  fused_block_dev<256,256,14,1,8,8,32,4>(
      &l3_b1_add_out[0][0][0][0],
      &l3_b2_w1[0][0][0][0], l3_b2_b1,
      l3_b2_bn1_mean, l3_b2_bn1_var, l3_b2_bn1_gamma, l3_b2_bn1_beta,
      &l3_b2_w2[0][0][0][0], l3_b2_b2,
      l3_b2_bn2_mean, l3_b2_bn2_var, l3_b2_bn2_gamma, l3_b2_bn2_beta,
      &l3_b2_add_out[0][0][0][0], &l3_b2_ws[0][0][0][0], smem);
}

__global__ void l3_b2_reduce_k()
{
  fused_block_reduce_dev<256,256,14,8,0>(
      &l3_b2_ws[0][0][0][0], &l3_b1_add_out[0][0][0][0],
      l3_b2_b2,
      l3_b2_bn2_mean, l3_b2_bn2_var, l3_b2_bn2_gamma, l3_b2_bn2_beta,
      l3_b2_b2, l3_b2_b2, l3_b2_b2, l3_b2_b2, l3_b2_b2, l3_b2_b2,
      &l3_b2_add_out[0][0][0][0]);
}

__global__ void __launch_bounds__(1024) l4_b1_fused_k()
{
  extern __shared__ float smem[];
  fused_block_dev<256,512,7,2,32,8,16,2>(
      &l3_b2_add_out[0][0][0][0],
      &l4_b1_w1[0][0][0][0], l4_b1_b1,
      l4_b1_bn1_mean, l4_b1_bn1_var, l4_b1_bn1_gamma, l4_b1_bn1_beta,
      &l4_b1_w2[0][0][0][0], l4_b1_b2,
      l4_b1_bn2_mean, l4_b1_bn2_var, l4_b1_bn2_gamma, l4_b1_bn2_beta,
      &l4_b1_add_out[0][0][0][0], &l4_b1_ws[0][0][0][0], smem);
}

__global__ void l4_b1_reduce_k()
{
  fused_block_reduce_dev<256,512,7,32,1>(
      &l4_b1_ws[0][0][0][0], &l3_b2_add_out[0][0][0][0],
      l4_b1_b2,
      l4_b1_bn2_mean, l4_b1_bn2_var, l4_b1_bn2_gamma, l4_b1_bn2_beta,
      &l4_b1_shortcut_w[0][0][0][0], l4_b1_shortcut_b,
      l4_b1_shortcut_mean, l4_b1_shortcut_var, l4_b1_shortcut_gamma, l4_b1_shortcut_beta,
      &l4_b1_add_out[0][0][0][0]);
}

__global__ void __launch_bounds__(1024) l4_b2_fused_k()
{
  extern __shared__ float smem[];
  fused_block_dev<512,512,7,1,32,8,32,4>(
      &l4_b1_add_out[0][0][0][0],
      &l4_b2_w1[0][0][0][0], l4_b2_b1,
      l4_b2_bn1_mean, l4_b2_bn1_var, l4_b2_bn1_gamma, l4_b2_bn1_beta,
      &l4_b2_w2[0][0][0][0], l4_b2_b2,
      l4_b2_bn2_mean, l4_b2_bn2_var, l4_b2_bn2_gamma, l4_b2_bn2_beta,
      &l4_b2_add_out[0][0][0][0], &l4_b2_ws[0][0][0][0], smem);
}

__global__ void l4_b2_reduce_k()
{
  fused_block_reduce_dev<512,512,7,32,0>(
      &l4_b2_ws[0][0][0][0], &l4_b1_add_out[0][0][0][0],
      l4_b2_b2,
      l4_b2_bn2_mean, l4_b2_bn2_var, l4_b2_bn2_gamma, l4_b2_bn2_beta,
      l4_b2_b2, l4_b2_b2, l4_b2_b2, l4_b2_b2, l4_b2_b2, l4_b2_b2,
      &l4_b2_add_out[0][0][0][0]);
}

__global__ void __launch_bounds__(1024) l1_stage_coop_k()
{
  extern __shared__ float smem[];
  pool_phase_grid();
  cooperative_groups::this_grid().sync();
  fused_block_dev<64,64,56,1,1,8,16,4>(
      &pool_out[0][0][0][0],
      &l1_b1_w1[0][0][0][0], l1_b1_b1,
      l1_b1_bn1_mean, l1_b1_bn1_var, l1_b1_bn1_gamma, l1_b1_bn1_beta,
      &l1_b1_w2[0][0][0][0], l1_b1_b2,
      l1_b1_bn2_mean, l1_b1_bn2_var, l1_b1_bn2_gamma, l1_b1_bn2_beta,
      &l1_b1_add_out[0][0][0][0], (float*)0, smem);
  cooperative_groups::this_grid().sync();
  fused_block_dev<64,64,56,1,1,8,16,4>(
      &l1_b1_add_out[0][0][0][0],
      &l1_b2_w1[0][0][0][0], l1_b2_b1,
      l1_b2_bn1_mean, l1_b2_bn1_var, l1_b2_bn1_gamma, l1_b2_bn1_beta,
      &l1_b2_w2[0][0][0][0], l1_b2_b2,
      l1_b2_bn2_mean, l1_b2_bn2_var, l1_b2_bn2_gamma, l1_b2_bn2_beta,
      &l1_b2_add_out[0][0][0][0], (float*)0, smem);
}

__global__ void __launch_bounds__(1024) l2_stage_coop_k()
{
  extern __shared__ float smem[];
  fused_block_dev<64,128,28,2,2,8,16,4>(
      &l1_b2_add_out[0][0][0][0],
      &l2_b1_w1[0][0][0][0], l2_b1_b1,
      l2_b1_bn1_mean, l2_b1_bn1_var, l2_b1_bn1_gamma, l2_b1_bn1_beta,
      &l2_b1_w2[0][0][0][0], l2_b1_b2,
      l2_b1_bn2_mean, l2_b1_bn2_var, l2_b1_bn2_gamma, l2_b1_bn2_beta,
      &l2_b1_add_out[0][0][0][0], &l2_b1_ws[0][0][0][0], smem);
  cooperative_groups::this_grid().sync();
  fused_block_reduce_grid<64,128,28,2,1>(
      &l2_b1_ws[0][0][0][0], &l1_b2_add_out[0][0][0][0],
      l2_b1_b2,
      l2_b1_bn2_mean, l2_b1_bn2_var, l2_b1_bn2_gamma, l2_b1_bn2_beta,
      &l2_b1_shortcut_w[0][0][0][0], l2_b1_shortcut_b,
      l2_b1_shortcut_mean, l2_b1_shortcut_var, l2_b1_shortcut_gamma, l2_b1_shortcut_beta,
      &l2_b1_add_out[0][0][0][0]);
  cooperative_groups::this_grid().sync();
  fused_block_dev<128,128,28,1,2,8,16,4>(
      &l2_b1_add_out[0][0][0][0],
      &l2_b2_w1[0][0][0][0], l2_b2_b1,
      l2_b2_bn1_mean, l2_b2_bn1_var, l2_b2_bn1_gamma, l2_b2_bn1_beta,
      &l2_b2_w2[0][0][0][0], l2_b2_b2,
      l2_b2_bn2_mean, l2_b2_bn2_var, l2_b2_bn2_gamma, l2_b2_bn2_beta,
      &l2_b2_add_out[0][0][0][0], &l2_b2_ws[0][0][0][0], smem);
  cooperative_groups::this_grid().sync();
  fused_block_reduce_grid<128,128,28,2,0>(
      &l2_b2_ws[0][0][0][0], &l2_b1_add_out[0][0][0][0],
      l2_b2_b2,
      l2_b2_bn2_mean, l2_b2_bn2_var, l2_b2_bn2_gamma, l2_b2_bn2_beta,
      l2_b2_b2, l2_b2_b2, l2_b2_b2, l2_b2_b2, l2_b2_b2, l2_b2_b2,
      &l2_b2_add_out[0][0][0][0]);
}

__global__ void __launch_bounds__(1024) l3_stage_coop_k()
{
  extern __shared__ float smem[];
  fused_block_dev<128,256,14,2,8,8,16,4>(
      &l2_b2_add_out[0][0][0][0],
      &l3_b1_w1[0][0][0][0], l3_b1_b1,
      l3_b1_bn1_mean, l3_b1_bn1_var, l3_b1_bn1_gamma, l3_b1_bn1_beta,
      &l3_b1_w2[0][0][0][0], l3_b1_b2,
      l3_b1_bn2_mean, l3_b1_bn2_var, l3_b1_bn2_gamma, l3_b1_bn2_beta,
      &l3_b1_add_out[0][0][0][0], &l3_b1_ws[0][0][0][0], smem);
  cooperative_groups::this_grid().sync();
  fused_block_reduce_grid<128,256,14,8,1>(
      &l3_b1_ws[0][0][0][0], &l2_b2_add_out[0][0][0][0],
      l3_b1_b2,
      l3_b1_bn2_mean, l3_b1_bn2_var, l3_b1_bn2_gamma, l3_b1_bn2_beta,
      &l3_b1_shortcut_w[0][0][0][0], l3_b1_shortcut_b,
      l3_b1_shortcut_mean, l3_b1_shortcut_var, l3_b1_shortcut_gamma, l3_b1_shortcut_beta,
      &l3_b1_add_out[0][0][0][0]);
  cooperative_groups::this_grid().sync();
  fused_block_dev<256,256,14,1,8,8,32,4>(
      &l3_b1_add_out[0][0][0][0],
      &l3_b2_w1[0][0][0][0], l3_b2_b1,
      l3_b2_bn1_mean, l3_b2_bn1_var, l3_b2_bn1_gamma, l3_b2_bn1_beta,
      &l3_b2_w2[0][0][0][0], l3_b2_b2,
      l3_b2_bn2_mean, l3_b2_bn2_var, l3_b2_bn2_gamma, l3_b2_bn2_beta,
      &l3_b2_add_out[0][0][0][0], &l3_b2_ws[0][0][0][0], smem);
  cooperative_groups::this_grid().sync();
  fused_block_reduce_grid<256,256,14,8,0>(
      &l3_b2_ws[0][0][0][0], &l3_b1_add_out[0][0][0][0],
      l3_b2_b2,
      l3_b2_bn2_mean, l3_b2_bn2_var, l3_b2_bn2_gamma, l3_b2_bn2_beta,
      l3_b2_b2, l3_b2_b2, l3_b2_b2, l3_b2_b2, l3_b2_b2, l3_b2_b2,
      &l3_b2_add_out[0][0][0][0]);
}

__global__ void __launch_bounds__(1024) l4_stage_coop_k()
{
  extern __shared__ float smem[];
  fused_block_dev<256,512,7,2,32,8,16,2>(
      &l3_b2_add_out[0][0][0][0],
      &l4_b1_w1[0][0][0][0], l4_b1_b1,
      l4_b1_bn1_mean, l4_b1_bn1_var, l4_b1_bn1_gamma, l4_b1_bn1_beta,
      &l4_b1_w2[0][0][0][0], l4_b1_b2,
      l4_b1_bn2_mean, l4_b1_bn2_var, l4_b1_bn2_gamma, l4_b1_bn2_beta,
      &l4_b1_add_out[0][0][0][0], &l4_b1_ws[0][0][0][0], smem);
  cooperative_groups::this_grid().sync();
  fused_block_reduce_grid<256,512,7,32,1>(
      &l4_b1_ws[0][0][0][0], &l3_b2_add_out[0][0][0][0],
      l4_b1_b2,
      l4_b1_bn2_mean, l4_b1_bn2_var, l4_b1_bn2_gamma, l4_b1_bn2_beta,
      &l4_b1_shortcut_w[0][0][0][0], l4_b1_shortcut_b,
      l4_b1_shortcut_mean, l4_b1_shortcut_var, l4_b1_shortcut_gamma, l4_b1_shortcut_beta,
      &l4_b1_add_out[0][0][0][0]);
  cooperative_groups::this_grid().sync();
  fused_block_dev<512,512,7,1,32,8,32,4>(
      &l4_b1_add_out[0][0][0][0],
      &l4_b2_w1[0][0][0][0], l4_b2_b1,
      l4_b2_bn1_mean, l4_b2_bn1_var, l4_b2_bn1_gamma, l4_b2_bn1_beta,
      &l4_b2_w2[0][0][0][0], l4_b2_b2,
      l4_b2_bn2_mean, l4_b2_bn2_var, l4_b2_bn2_gamma, l4_b2_bn2_beta,
      &l4_b2_add_out[0][0][0][0], &l4_b2_ws[0][0][0][0], smem);
  cooperative_groups::this_grid().sync();
  fused_block_reduce_grid<512,512,7,32,0>(
      &l4_b2_ws[0][0][0][0], &l4_b1_add_out[0][0][0][0],
      l4_b2_b2,
      l4_b2_bn2_mean, l4_b2_bn2_var, l4_b2_bn2_gamma, l4_b2_bn2_beta,
      l4_b2_b2, l4_b2_b2, l4_b2_b2, l4_b2_b2, l4_b2_b2, l4_b2_b2,
      &l4_b2_add_out[0][0][0][0]);
  cooperative_groups::this_grid().sync();
  { /* GAP phase: 512 channel means over 7x7 */
    const int gtid = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    float* avg = &l4_b2_ws[0][0][0][0];
    if (gtid < 512) {
      float s = 0.0f;
      for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++)
          s += l4_b2_add_out[0][gtid][i][j];
      avg[gtid] = s / 49.0f;
    }
  }
  cooperative_groups::this_grid().sync();
  { /* FC phase: one warp per output, shuffle reduction */
    const int gtid2 = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    const int gth2 = gridDim.x * gridDim.y * gridDim.z * blockDim.x;
    const int wid = gtid2 >> 5, lane = gtid2 & 31;
    const float* avg = &l4_b2_ws[0][0][0][0];
    for (int o = wid; o < 1000; o += gth2 >> 5) {
      float a = 0.0f;
      for (int c = lane; c < 512; c += 32)
        a += avg[c] * fc_w[o][c];
      for (int off = 16; off > 0; off >>= 1)
        a += __shfl_down_sync(0xffffffffu, a, off);
      if (lane == 0) output[0][o] = a + fc_b[o];
    }
  }
}


static void resnet18_fused_seq_s(cudaStream_t __st)
{
  { dim3 g(7,7,16); dim3 b(16,16,1); ast0_relu1_out_fused<<<g, b, 0, __st>>>(); }
  ast1_pool_out<<<784, 256, 0, __st>>>();
  { dim3 g(7,7,1); l1_b1_fused_k<<<g, 1024, 71680, __st>>>(); }
  { dim3 g(7,7,1); l1_b2_fused_k<<<g, 1024, 71680, __st>>>(); }
  { dim3 g(4,4,2); l2_b1_fused_k<<<g, 1024, 90688, __st>>>(); }
  l2_b1_reduce_k<<<392, 256, 0, __st>>>();
  { dim3 g(4,4,2); l2_b2_fused_k<<<g, 1024, 71680, __st>>>(); }
  l2_b2_reduce_k<<<392, 256, 0, __st>>>();
  { dim3 g(2,2,8); l3_b1_fused_k<<<g, 1024, 77888, __st>>>(); }
  l3_b1_reduce_k<<<196, 256, 0, __st>>>();
  { dim3 g(2,2,8); l3_b2_fused_k<<<g, 1024, 68096, __st>>>(); }
  l3_b2_reduce_k<<<196, 256, 0, __st>>>();
  { dim3 g(1,1,32); l4_b1_fused_k<<<g, 1024, 71488, __st>>>(); }
  l4_b1_reduce_k<<<98, 256, 0, __st>>>();
  { dim3 g(1,1,32); l4_b2_fused_k<<<g, 1024, 98560, __st>>>(); }
  l4_b2_reduce_k<<<98, 256, 0, __st>>>();
  { dim3 g(1000,1,1); dim3 b(256,1,1); ast32_output_fused<<<g, b, 0, __st>>>(); }
}

void resnet18_fused_s(cudaStream_t __st)
{
  static int __init = 0;
  static int __coop = 0;
  if (!__init) {
    __init = 1;
    cudaFuncSetAttribute(l1_b1_fused_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 71680);
    cudaFuncSetAttribute(l1_b2_fused_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 71680);
    cudaFuncSetAttribute(l2_b1_fused_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 90688);
    cudaFuncSetAttribute(l2_b2_fused_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 71680);
    cudaFuncSetAttribute(l3_b1_fused_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 77888);
    cudaFuncSetAttribute(l3_b2_fused_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 68096);
    cudaFuncSetAttribute(l4_b1_fused_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 71488);
    cudaFuncSetAttribute(l4_b2_fused_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 98560);
    cudaFuncSetAttribute(l1_stage_coop_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 71680);
    cudaFuncSetAttribute(l2_stage_coop_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 90688);
    cudaFuncSetAttribute(l3_stage_coop_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 77888);
    cudaFuncSetAttribute(l4_stage_coop_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 98560);
    {
      int dev = 0, nsm = 0, c = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&c, cudaDevAttrCooperativeLaunch, dev);
      cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
      if (c) {
        int per;
        __coop = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, l1_stage_coop_k, 1024, 71680);
        __coop = __coop && (per * nsm >= 49);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, l2_stage_coop_k, 1024, 90688);
        __coop = __coop && (per * nsm >= 32);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, l3_stage_coop_k, 1024, 77888);
        __coop = __coop && (per * nsm >= 32);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, l4_stage_coop_k, 1024, 98560);
        __coop = __coop && (per * nsm >= 32);
      }
    }
  }
  if (__coop) {
    { dim3 g(7,7,16); dim3 b(16,16,1); ast0_relu1_out_fused<<<g, b, 0, __st>>>(); }
    { void* __a[1] = {0}; cudaLaunchCooperativeKernel((const void*)l1_stage_coop_k, dim3(7,7,1), dim3(1024), __a, 71680, __st); }
    { void* __a[1] = {0}; cudaLaunchCooperativeKernel((const void*)l2_stage_coop_k, dim3(4,4,2), dim3(1024), __a, 90688, __st); }
    { void* __a[1] = {0}; cudaLaunchCooperativeKernel((const void*)l3_stage_coop_k, dim3(2,2,8), dim3(1024), __a, 77888, __st); }
    { void* __a[1] = {0}; cudaLaunchCooperativeKernel((const void*)l4_stage_coop_k, dim3(1,1,32), dim3(1024), __a, 98560, __st); }
  } else {
    resnet18_fused_seq_s(__st);
  }
}

void resnet18_fused()
{
  resnet18_fused_s(0);
}

void resnet18_fused_kt()
{
  cudaEvent_t __kta, __ktb;
  cudaEventCreate(&__kta); cudaEventCreate(&__ktb);
  cudaStream_t __st = 0;
  resnet18_fused();  /* warmup + produce valid intermediates */
  KT_ONE(0, "stem_conv_bn_relu", { dim3 g(7,7,16); dim3 b(16,16,1); ast0_relu1_out_fused<<<g, b>>>(); });
  KT_ONE(1, "stem_pool", ast1_pool_out<<<784, 256>>>());
  KT_ONE(2, "l1_b1_fused", { dim3 g(7,7,1); l1_b1_fused_k<<<g, 1024, 71680, __st>>>(); };);
  KT_ONE(3, "l1_b2_fused", { dim3 g(7,7,1); l1_b2_fused_k<<<g, 1024, 71680, __st>>>(); };);
  KT_ONE(4, "l2_b1_fused", { dim3 g(4,4,2); l2_b1_fused_k<<<g, 1024, 90688, __st>>>(); };);
  KT_ONE(5, "l2_b1_reduce",  l2_b1_reduce_k<<<392, 256, 0, __st>>>();;);
  KT_ONE(6, "l2_b2_fused", { dim3 g(4,4,2); l2_b2_fused_k<<<g, 1024, 71680, __st>>>(); };);
  KT_ONE(7, "l2_b2_reduce",  l2_b2_reduce_k<<<392, 256, 0, __st>>>();;);
  KT_ONE(8, "l3_b1_fused", { dim3 g(2,2,8); l3_b1_fused_k<<<g, 1024, 77888, __st>>>(); };);
  KT_ONE(9, "l3_b1_reduce",  l3_b1_reduce_k<<<196, 256, 0, __st>>>();;);
  KT_ONE(10, "l3_b2_fused", { dim3 g(2,2,8); l3_b2_fused_k<<<g, 1024, 68096, __st>>>(); };);
  KT_ONE(11, "l3_b2_reduce",  l3_b2_reduce_k<<<196, 256, 0, __st>>>();;);
  KT_ONE(12, "l4_b1_fused", { dim3 g(1,1,32); l4_b1_fused_k<<<g, 1024, 71488, __st>>>(); };);
  KT_ONE(13, "l4_b1_reduce",  l4_b1_reduce_k<<<98, 256, 0, __st>>>();;);
  KT_ONE(14, "l4_b2_fused", { dim3 g(1,1,32); l4_b2_fused_k<<<g, 1024, 98560, __st>>>(); };);
  KT_ONE(15, "l4_b2_reduce",  l4_b2_reduce_k<<<98, 256, 0, __st>>>();;);
  KT_ONE(16, "gap_fc", { dim3 g(1000,1,1); dim3 b(256,1,1); ast32_output_fused<<<g, b>>>(); });
  KT_ONE(17, "l1_stage_coop", { void* __a[1] = {0}; cudaLaunchCooperativeKernel((const void*)l1_stage_coop_k, dim3(7,7,1), dim3(1024), __a, 71680, 0); });
  KT_ONE(18, "l2_stage_coop", { void* __a[1] = {0}; cudaLaunchCooperativeKernel((const void*)l2_stage_coop_k, dim3(4,4,2), dim3(1024), __a, 90688, 0); });
  KT_ONE(19, "l3_stage_coop", { void* __a[1] = {0}; cudaLaunchCooperativeKernel((const void*)l3_stage_coop_k, dim3(2,2,8), dim3(1024), __a, 77888, 0); });
  KT_ONE(20, "l4_stage_coop", { void* __a[1] = {0}; cudaLaunchCooperativeKernel((const void*)l4_stage_coop_k, dim3(1,1,32), dim3(1024), __a, 98560, 0); });
  cudaEventDestroy(__kta); cudaEventDestroy(__ktb);
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
  size_t l1_b1_add_out_bytes = 802816;
  float* d_l1_b1_add_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b1_add_out, l1_b1_add_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b1_add_out, &d_l1_b1_add_out, sizeof(d_l1_b1_add_out)));
  size_t l1_b2_add_out_bytes = 802816;
  float* d_l1_b2_add_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l1_b2_add_out, l1_b2_add_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l1_b2_add_out, &d_l1_b2_add_out, sizeof(d_l1_b2_add_out)));
  size_t l2_b1_ws_bytes = 802816;
  float* d_l2_b1_ws = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_ws, l2_b1_ws_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_ws, &d_l2_b1_ws, sizeof(d_l2_b1_ws)));
  size_t l2_b1_add_out_bytes = 401408;
  float* d_l2_b1_add_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b1_add_out, l2_b1_add_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b1_add_out, &d_l2_b1_add_out, sizeof(d_l2_b1_add_out)));
  size_t l2_b2_ws_bytes = 802816;
  float* d_l2_b2_ws = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b2_ws, l2_b2_ws_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b2_ws, &d_l2_b2_ws, sizeof(d_l2_b2_ws)));
  size_t l2_b2_add_out_bytes = 401408;
  float* d_l2_b2_add_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l2_b2_add_out, l2_b2_add_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l2_b2_add_out, &d_l2_b2_add_out, sizeof(d_l2_b2_add_out)));
  size_t l3_b1_ws_bytes = 1605632;
  float* d_l3_b1_ws = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_ws, l3_b1_ws_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_ws, &d_l3_b1_ws, sizeof(d_l3_b1_ws)));
  size_t l3_b1_add_out_bytes = 200704;
  float* d_l3_b1_add_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b1_add_out, l3_b1_add_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b1_add_out, &d_l3_b1_add_out, sizeof(d_l3_b1_add_out)));
  size_t l3_b2_ws_bytes = 1605632;
  float* d_l3_b2_ws = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b2_ws, l3_b2_ws_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b2_ws, &d_l3_b2_ws, sizeof(d_l3_b2_ws)));
  size_t l3_b2_add_out_bytes = 200704;
  float* d_l3_b2_add_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l3_b2_add_out, l3_b2_add_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l3_b2_add_out, &d_l3_b2_add_out, sizeof(d_l3_b2_add_out)));
  size_t l4_b1_ws_bytes = 3211264;
  float* d_l4_b1_ws = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_ws, l4_b1_ws_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_ws, &d_l4_b1_ws, sizeof(d_l4_b1_ws)));
  size_t l4_b1_add_out_bytes = 100352;
  float* d_l4_b1_add_out = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b1_add_out, l4_b1_add_out_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b1_add_out, &d_l4_b1_add_out, sizeof(d_l4_b1_add_out)));
  size_t l4_b2_ws_bytes = 3211264;
  float* d_l4_b2_ws = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_l4_b2_ws, l4_b2_ws_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(l4_b2_ws, &d_l4_b2_ws, sizeof(d_l4_b2_ws)));
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
  { FILE* f = fopen("resnet18_fused_input.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_input.bin\n"); return 1; }
    size_t nread = fread(h_input, sizeof(float), 150528, f);
    fclose(f);
    if (nread != 150528) { fprintf(stderr, "Failed to read resnet18_fused_input.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_input, h_input, 602112, cudaMemcpyHostToDevice));
  float* h_conv1_w = (float*)malloc(37632);
  if (!h_conv1_w) { fprintf(stderr, "malloc failed for conv1_w\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_conv1_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_conv1_w.bin\n"); return 1; }
    size_t nread = fread(h_conv1_w, sizeof(float), 9408, f);
    fclose(f);
    if (nread != 9408) { fprintf(stderr, "Failed to read resnet18_fused_conv1_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv1_w, h_conv1_w, 37632, cudaMemcpyHostToDevice));
  float* h_conv1_b = (float*)malloc(256);
  if (!h_conv1_b) { fprintf(stderr, "malloc failed for conv1_b\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_conv1_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_conv1_b.bin\n"); return 1; }
    size_t nread = fread(h_conv1_b, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_conv1_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv1_b, h_conv1_b, 256, cudaMemcpyHostToDevice));
  float* h_bn1_mean = (float*)malloc(256);
  if (!h_bn1_mean) { fprintf(stderr, "malloc failed for bn1_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_bn1_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_bn1_mean.bin\n"); return 1; }
    size_t nread = fread(h_bn1_mean, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_bn1_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_bn1_mean, h_bn1_mean, 256, cudaMemcpyHostToDevice));
  float* h_bn1_var = (float*)malloc(256);
  if (!h_bn1_var) { fprintf(stderr, "malloc failed for bn1_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_bn1_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_bn1_var.bin\n"); return 1; }
    size_t nread = fread(h_bn1_var, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_bn1_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_bn1_var, h_bn1_var, 256, cudaMemcpyHostToDevice));
  float* h_bn1_gamma = (float*)malloc(256);
  if (!h_bn1_gamma) { fprintf(stderr, "malloc failed for bn1_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_bn1_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_bn1_gamma.bin\n"); return 1; }
    size_t nread = fread(h_bn1_gamma, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_bn1_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_bn1_gamma, h_bn1_gamma, 256, cudaMemcpyHostToDevice));
  float* h_bn1_beta = (float*)malloc(256);
  if (!h_bn1_beta) { fprintf(stderr, "malloc failed for bn1_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_bn1_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_bn1_beta.bin\n"); return 1; }
    size_t nread = fread(h_bn1_beta, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_bn1_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_bn1_beta, h_bn1_beta, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_w1 = (float*)malloc(147456);
  if (!h_l1_b1_w1) { fprintf(stderr, "malloc failed for l1_b1_w1\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b1_w1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b1_w1.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_w1, sizeof(float), 36864, f);
    fclose(f);
    if (nread != 36864) { fprintf(stderr, "Failed to read resnet18_fused_l1_b1_w1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_w1, h_l1_b1_w1, 147456, cudaMemcpyHostToDevice));
  float* h_l1_b1_b1 = (float*)malloc(256);
  if (!h_l1_b1_b1) { fprintf(stderr, "malloc failed for l1_b1_b1\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b1_b1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b1_b1.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_b1, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b1_b1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_b1, h_l1_b1_b1, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_w2 = (float*)malloc(147456);
  if (!h_l1_b1_w2) { fprintf(stderr, "malloc failed for l1_b1_w2\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b1_w2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b1_w2.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_w2, sizeof(float), 36864, f);
    fclose(f);
    if (nread != 36864) { fprintf(stderr, "Failed to read resnet18_fused_l1_b1_w2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_w2, h_l1_b1_w2, 147456, cudaMemcpyHostToDevice));
  float* h_l1_b1_b2 = (float*)malloc(256);
  if (!h_l1_b1_b2) { fprintf(stderr, "malloc failed for l1_b1_b2\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b1_b2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b1_b2.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_b2, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b1_b2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_b2, h_l1_b1_b2, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_bn1_mean = (float*)malloc(256);
  if (!h_l1_b1_bn1_mean) { fprintf(stderr, "malloc failed for l1_b1_bn1_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b1_bn1_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b1_bn1_mean.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_bn1_mean, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b1_bn1_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_bn1_mean, h_l1_b1_bn1_mean, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_bn1_var = (float*)malloc(256);
  if (!h_l1_b1_bn1_var) { fprintf(stderr, "malloc failed for l1_b1_bn1_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b1_bn1_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b1_bn1_var.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_bn1_var, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b1_bn1_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_bn1_var, h_l1_b1_bn1_var, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_bn1_gamma = (float*)malloc(256);
  if (!h_l1_b1_bn1_gamma) { fprintf(stderr, "malloc failed for l1_b1_bn1_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b1_bn1_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b1_bn1_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_bn1_gamma, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b1_bn1_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_bn1_gamma, h_l1_b1_bn1_gamma, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_bn1_beta = (float*)malloc(256);
  if (!h_l1_b1_bn1_beta) { fprintf(stderr, "malloc failed for l1_b1_bn1_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b1_bn1_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b1_bn1_beta.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_bn1_beta, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b1_bn1_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_bn1_beta, h_l1_b1_bn1_beta, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_bn2_mean = (float*)malloc(256);
  if (!h_l1_b1_bn2_mean) { fprintf(stderr, "malloc failed for l1_b1_bn2_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b1_bn2_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b1_bn2_mean.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_bn2_mean, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b1_bn2_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_bn2_mean, h_l1_b1_bn2_mean, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_bn2_var = (float*)malloc(256);
  if (!h_l1_b1_bn2_var) { fprintf(stderr, "malloc failed for l1_b1_bn2_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b1_bn2_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b1_bn2_var.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_bn2_var, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b1_bn2_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_bn2_var, h_l1_b1_bn2_var, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_bn2_gamma = (float*)malloc(256);
  if (!h_l1_b1_bn2_gamma) { fprintf(stderr, "malloc failed for l1_b1_bn2_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b1_bn2_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b1_bn2_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_bn2_gamma, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b1_bn2_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_bn2_gamma, h_l1_b1_bn2_gamma, 256, cudaMemcpyHostToDevice));
  float* h_l1_b1_bn2_beta = (float*)malloc(256);
  if (!h_l1_b1_bn2_beta) { fprintf(stderr, "malloc failed for l1_b1_bn2_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b1_bn2_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b1_bn2_beta.bin\n"); return 1; }
    size_t nread = fread(h_l1_b1_bn2_beta, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b1_bn2_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b1_bn2_beta, h_l1_b1_bn2_beta, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_w1 = (float*)malloc(147456);
  if (!h_l1_b2_w1) { fprintf(stderr, "malloc failed for l1_b2_w1\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b2_w1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b2_w1.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_w1, sizeof(float), 36864, f);
    fclose(f);
    if (nread != 36864) { fprintf(stderr, "Failed to read resnet18_fused_l1_b2_w1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_w1, h_l1_b2_w1, 147456, cudaMemcpyHostToDevice));
  float* h_l1_b2_b1 = (float*)malloc(256);
  if (!h_l1_b2_b1) { fprintf(stderr, "malloc failed for l1_b2_b1\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b2_b1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b2_b1.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_b1, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b2_b1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_b1, h_l1_b2_b1, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_w2 = (float*)malloc(147456);
  if (!h_l1_b2_w2) { fprintf(stderr, "malloc failed for l1_b2_w2\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b2_w2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b2_w2.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_w2, sizeof(float), 36864, f);
    fclose(f);
    if (nread != 36864) { fprintf(stderr, "Failed to read resnet18_fused_l1_b2_w2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_w2, h_l1_b2_w2, 147456, cudaMemcpyHostToDevice));
  float* h_l1_b2_b2 = (float*)malloc(256);
  if (!h_l1_b2_b2) { fprintf(stderr, "malloc failed for l1_b2_b2\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b2_b2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b2_b2.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_b2, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b2_b2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_b2, h_l1_b2_b2, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_bn1_mean = (float*)malloc(256);
  if (!h_l1_b2_bn1_mean) { fprintf(stderr, "malloc failed for l1_b2_bn1_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b2_bn1_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b2_bn1_mean.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_bn1_mean, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b2_bn1_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_bn1_mean, h_l1_b2_bn1_mean, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_bn1_var = (float*)malloc(256);
  if (!h_l1_b2_bn1_var) { fprintf(stderr, "malloc failed for l1_b2_bn1_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b2_bn1_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b2_bn1_var.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_bn1_var, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b2_bn1_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_bn1_var, h_l1_b2_bn1_var, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_bn1_gamma = (float*)malloc(256);
  if (!h_l1_b2_bn1_gamma) { fprintf(stderr, "malloc failed for l1_b2_bn1_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b2_bn1_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b2_bn1_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_bn1_gamma, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b2_bn1_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_bn1_gamma, h_l1_b2_bn1_gamma, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_bn1_beta = (float*)malloc(256);
  if (!h_l1_b2_bn1_beta) { fprintf(stderr, "malloc failed for l1_b2_bn1_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b2_bn1_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b2_bn1_beta.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_bn1_beta, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b2_bn1_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_bn1_beta, h_l1_b2_bn1_beta, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_bn2_mean = (float*)malloc(256);
  if (!h_l1_b2_bn2_mean) { fprintf(stderr, "malloc failed for l1_b2_bn2_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b2_bn2_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b2_bn2_mean.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_bn2_mean, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b2_bn2_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_bn2_mean, h_l1_b2_bn2_mean, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_bn2_var = (float*)malloc(256);
  if (!h_l1_b2_bn2_var) { fprintf(stderr, "malloc failed for l1_b2_bn2_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b2_bn2_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b2_bn2_var.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_bn2_var, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b2_bn2_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_bn2_var, h_l1_b2_bn2_var, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_bn2_gamma = (float*)malloc(256);
  if (!h_l1_b2_bn2_gamma) { fprintf(stderr, "malloc failed for l1_b2_bn2_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b2_bn2_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b2_bn2_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_bn2_gamma, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b2_bn2_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_bn2_gamma, h_l1_b2_bn2_gamma, 256, cudaMemcpyHostToDevice));
  float* h_l1_b2_bn2_beta = (float*)malloc(256);
  if (!h_l1_b2_bn2_beta) { fprintf(stderr, "malloc failed for l1_b2_bn2_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l1_b2_bn2_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l1_b2_bn2_beta.bin\n"); return 1; }
    size_t nread = fread(h_l1_b2_bn2_beta, sizeof(float), 64, f);
    fclose(f);
    if (nread != 64) { fprintf(stderr, "Failed to read resnet18_fused_l1_b2_bn2_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l1_b2_bn2_beta, h_l1_b2_bn2_beta, 256, cudaMemcpyHostToDevice));
  float* h_l2_b1_w1 = (float*)malloc(294912);
  if (!h_l2_b1_w1) { fprintf(stderr, "malloc failed for l2_b1_w1\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b1_w1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b1_w1.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_w1, sizeof(float), 73728, f);
    fclose(f);
    if (nread != 73728) { fprintf(stderr, "Failed to read resnet18_fused_l2_b1_w1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_w1, h_l2_b1_w1, 294912, cudaMemcpyHostToDevice));
  float* h_l2_b1_b1 = (float*)malloc(512);
  if (!h_l2_b1_b1) { fprintf(stderr, "malloc failed for l2_b1_b1\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b1_b1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b1_b1.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_b1, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b1_b1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_b1, h_l2_b1_b1, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_w2 = (float*)malloc(589824);
  if (!h_l2_b1_w2) { fprintf(stderr, "malloc failed for l2_b1_w2\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b1_w2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b1_w2.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_w2, sizeof(float), 147456, f);
    fclose(f);
    if (nread != 147456) { fprintf(stderr, "Failed to read resnet18_fused_l2_b1_w2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_w2, h_l2_b1_w2, 589824, cudaMemcpyHostToDevice));
  float* h_l2_b1_b2 = (float*)malloc(512);
  if (!h_l2_b1_b2) { fprintf(stderr, "malloc failed for l2_b1_b2\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b1_b2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b1_b2.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_b2, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b1_b2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_b2, h_l2_b1_b2, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_bn1_mean = (float*)malloc(512);
  if (!h_l2_b1_bn1_mean) { fprintf(stderr, "malloc failed for l2_b1_bn1_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b1_bn1_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b1_bn1_mean.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_bn1_mean, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b1_bn1_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_bn1_mean, h_l2_b1_bn1_mean, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_bn1_var = (float*)malloc(512);
  if (!h_l2_b1_bn1_var) { fprintf(stderr, "malloc failed for l2_b1_bn1_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b1_bn1_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b1_bn1_var.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_bn1_var, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b1_bn1_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_bn1_var, h_l2_b1_bn1_var, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_bn1_gamma = (float*)malloc(512);
  if (!h_l2_b1_bn1_gamma) { fprintf(stderr, "malloc failed for l2_b1_bn1_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b1_bn1_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b1_bn1_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_bn1_gamma, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b1_bn1_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_bn1_gamma, h_l2_b1_bn1_gamma, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_bn1_beta = (float*)malloc(512);
  if (!h_l2_b1_bn1_beta) { fprintf(stderr, "malloc failed for l2_b1_bn1_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b1_bn1_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b1_bn1_beta.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_bn1_beta, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b1_bn1_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_bn1_beta, h_l2_b1_bn1_beta, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_bn2_mean = (float*)malloc(512);
  if (!h_l2_b1_bn2_mean) { fprintf(stderr, "malloc failed for l2_b1_bn2_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b1_bn2_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b1_bn2_mean.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_bn2_mean, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b1_bn2_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_bn2_mean, h_l2_b1_bn2_mean, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_bn2_var = (float*)malloc(512);
  if (!h_l2_b1_bn2_var) { fprintf(stderr, "malloc failed for l2_b1_bn2_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b1_bn2_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b1_bn2_var.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_bn2_var, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b1_bn2_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_bn2_var, h_l2_b1_bn2_var, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_bn2_gamma = (float*)malloc(512);
  if (!h_l2_b1_bn2_gamma) { fprintf(stderr, "malloc failed for l2_b1_bn2_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b1_bn2_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b1_bn2_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_bn2_gamma, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b1_bn2_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_bn2_gamma, h_l2_b1_bn2_gamma, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_bn2_beta = (float*)malloc(512);
  if (!h_l2_b1_bn2_beta) { fprintf(stderr, "malloc failed for l2_b1_bn2_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b1_bn2_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b1_bn2_beta.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_bn2_beta, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b1_bn2_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_bn2_beta, h_l2_b1_bn2_beta, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_shortcut_w = (float*)malloc(32768);
  if (!h_l2_b1_shortcut_w) { fprintf(stderr, "malloc failed for l2_b1_shortcut_w\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b1_shortcut_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b1_shortcut_w.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_shortcut_w, sizeof(float), 8192, f);
    fclose(f);
    if (nread != 8192) { fprintf(stderr, "Failed to read resnet18_fused_l2_b1_shortcut_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_shortcut_w, h_l2_b1_shortcut_w, 32768, cudaMemcpyHostToDevice));
  float* h_l2_b1_shortcut_b = (float*)malloc(512);
  if (!h_l2_b1_shortcut_b) { fprintf(stderr, "malloc failed for l2_b1_shortcut_b\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b1_shortcut_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b1_shortcut_b.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_shortcut_b, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b1_shortcut_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_shortcut_b, h_l2_b1_shortcut_b, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_shortcut_mean = (float*)malloc(512);
  if (!h_l2_b1_shortcut_mean) { fprintf(stderr, "malloc failed for l2_b1_shortcut_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b1_shortcut_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b1_shortcut_mean.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_shortcut_mean, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b1_shortcut_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_shortcut_mean, h_l2_b1_shortcut_mean, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_shortcut_var = (float*)malloc(512);
  if (!h_l2_b1_shortcut_var) { fprintf(stderr, "malloc failed for l2_b1_shortcut_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b1_shortcut_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b1_shortcut_var.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_shortcut_var, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b1_shortcut_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_shortcut_var, h_l2_b1_shortcut_var, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_shortcut_gamma = (float*)malloc(512);
  if (!h_l2_b1_shortcut_gamma) { fprintf(stderr, "malloc failed for l2_b1_shortcut_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b1_shortcut_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b1_shortcut_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_shortcut_gamma, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b1_shortcut_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_shortcut_gamma, h_l2_b1_shortcut_gamma, 512, cudaMemcpyHostToDevice));
  float* h_l2_b1_shortcut_beta = (float*)malloc(512);
  if (!h_l2_b1_shortcut_beta) { fprintf(stderr, "malloc failed for l2_b1_shortcut_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b1_shortcut_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b1_shortcut_beta.bin\n"); return 1; }
    size_t nread = fread(h_l2_b1_shortcut_beta, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b1_shortcut_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b1_shortcut_beta, h_l2_b1_shortcut_beta, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_w1 = (float*)malloc(589824);
  if (!h_l2_b2_w1) { fprintf(stderr, "malloc failed for l2_b2_w1\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b2_w1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b2_w1.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_w1, sizeof(float), 147456, f);
    fclose(f);
    if (nread != 147456) { fprintf(stderr, "Failed to read resnet18_fused_l2_b2_w1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_w1, h_l2_b2_w1, 589824, cudaMemcpyHostToDevice));
  float* h_l2_b2_b1 = (float*)malloc(512);
  if (!h_l2_b2_b1) { fprintf(stderr, "malloc failed for l2_b2_b1\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b2_b1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b2_b1.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_b1, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b2_b1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_b1, h_l2_b2_b1, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_w2 = (float*)malloc(589824);
  if (!h_l2_b2_w2) { fprintf(stderr, "malloc failed for l2_b2_w2\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b2_w2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b2_w2.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_w2, sizeof(float), 147456, f);
    fclose(f);
    if (nread != 147456) { fprintf(stderr, "Failed to read resnet18_fused_l2_b2_w2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_w2, h_l2_b2_w2, 589824, cudaMemcpyHostToDevice));
  float* h_l2_b2_b2 = (float*)malloc(512);
  if (!h_l2_b2_b2) { fprintf(stderr, "malloc failed for l2_b2_b2\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b2_b2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b2_b2.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_b2, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b2_b2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_b2, h_l2_b2_b2, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_bn1_mean = (float*)malloc(512);
  if (!h_l2_b2_bn1_mean) { fprintf(stderr, "malloc failed for l2_b2_bn1_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b2_bn1_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b2_bn1_mean.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_bn1_mean, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b2_bn1_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_bn1_mean, h_l2_b2_bn1_mean, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_bn1_var = (float*)malloc(512);
  if (!h_l2_b2_bn1_var) { fprintf(stderr, "malloc failed for l2_b2_bn1_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b2_bn1_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b2_bn1_var.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_bn1_var, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b2_bn1_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_bn1_var, h_l2_b2_bn1_var, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_bn1_gamma = (float*)malloc(512);
  if (!h_l2_b2_bn1_gamma) { fprintf(stderr, "malloc failed for l2_b2_bn1_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b2_bn1_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b2_bn1_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_bn1_gamma, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b2_bn1_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_bn1_gamma, h_l2_b2_bn1_gamma, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_bn1_beta = (float*)malloc(512);
  if (!h_l2_b2_bn1_beta) { fprintf(stderr, "malloc failed for l2_b2_bn1_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b2_bn1_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b2_bn1_beta.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_bn1_beta, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b2_bn1_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_bn1_beta, h_l2_b2_bn1_beta, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_bn2_mean = (float*)malloc(512);
  if (!h_l2_b2_bn2_mean) { fprintf(stderr, "malloc failed for l2_b2_bn2_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b2_bn2_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b2_bn2_mean.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_bn2_mean, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b2_bn2_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_bn2_mean, h_l2_b2_bn2_mean, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_bn2_var = (float*)malloc(512);
  if (!h_l2_b2_bn2_var) { fprintf(stderr, "malloc failed for l2_b2_bn2_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b2_bn2_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b2_bn2_var.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_bn2_var, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b2_bn2_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_bn2_var, h_l2_b2_bn2_var, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_bn2_gamma = (float*)malloc(512);
  if (!h_l2_b2_bn2_gamma) { fprintf(stderr, "malloc failed for l2_b2_bn2_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b2_bn2_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b2_bn2_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_bn2_gamma, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b2_bn2_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_bn2_gamma, h_l2_b2_bn2_gamma, 512, cudaMemcpyHostToDevice));
  float* h_l2_b2_bn2_beta = (float*)malloc(512);
  if (!h_l2_b2_bn2_beta) { fprintf(stderr, "malloc failed for l2_b2_bn2_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l2_b2_bn2_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l2_b2_bn2_beta.bin\n"); return 1; }
    size_t nread = fread(h_l2_b2_bn2_beta, sizeof(float), 128, f);
    fclose(f);
    if (nread != 128) { fprintf(stderr, "Failed to read resnet18_fused_l2_b2_bn2_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l2_b2_bn2_beta, h_l2_b2_bn2_beta, 512, cudaMemcpyHostToDevice));
  float* h_l3_b1_w1 = (float*)malloc(1179648);
  if (!h_l3_b1_w1) { fprintf(stderr, "malloc failed for l3_b1_w1\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b1_w1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b1_w1.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_w1, sizeof(float), 294912, f);
    fclose(f);
    if (nread != 294912) { fprintf(stderr, "Failed to read resnet18_fused_l3_b1_w1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_w1, h_l3_b1_w1, 1179648, cudaMemcpyHostToDevice));
  float* h_l3_b1_b1 = (float*)malloc(1024);
  if (!h_l3_b1_b1) { fprintf(stderr, "malloc failed for l3_b1_b1\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b1_b1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b1_b1.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_b1, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b1_b1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_b1, h_l3_b1_b1, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_w2 = (float*)malloc(2359296);
  if (!h_l3_b1_w2) { fprintf(stderr, "malloc failed for l3_b1_w2\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b1_w2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b1_w2.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_w2, sizeof(float), 589824, f);
    fclose(f);
    if (nread != 589824) { fprintf(stderr, "Failed to read resnet18_fused_l3_b1_w2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_w2, h_l3_b1_w2, 2359296, cudaMemcpyHostToDevice));
  float* h_l3_b1_b2 = (float*)malloc(1024);
  if (!h_l3_b1_b2) { fprintf(stderr, "malloc failed for l3_b1_b2\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b1_b2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b1_b2.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_b2, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b1_b2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_b2, h_l3_b1_b2, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_bn1_mean = (float*)malloc(1024);
  if (!h_l3_b1_bn1_mean) { fprintf(stderr, "malloc failed for l3_b1_bn1_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b1_bn1_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b1_bn1_mean.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_bn1_mean, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b1_bn1_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_bn1_mean, h_l3_b1_bn1_mean, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_bn1_var = (float*)malloc(1024);
  if (!h_l3_b1_bn1_var) { fprintf(stderr, "malloc failed for l3_b1_bn1_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b1_bn1_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b1_bn1_var.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_bn1_var, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b1_bn1_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_bn1_var, h_l3_b1_bn1_var, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_bn1_gamma = (float*)malloc(1024);
  if (!h_l3_b1_bn1_gamma) { fprintf(stderr, "malloc failed for l3_b1_bn1_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b1_bn1_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b1_bn1_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_bn1_gamma, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b1_bn1_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_bn1_gamma, h_l3_b1_bn1_gamma, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_bn1_beta = (float*)malloc(1024);
  if (!h_l3_b1_bn1_beta) { fprintf(stderr, "malloc failed for l3_b1_bn1_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b1_bn1_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b1_bn1_beta.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_bn1_beta, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b1_bn1_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_bn1_beta, h_l3_b1_bn1_beta, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_bn2_mean = (float*)malloc(1024);
  if (!h_l3_b1_bn2_mean) { fprintf(stderr, "malloc failed for l3_b1_bn2_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b1_bn2_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b1_bn2_mean.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_bn2_mean, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b1_bn2_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_bn2_mean, h_l3_b1_bn2_mean, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_bn2_var = (float*)malloc(1024);
  if (!h_l3_b1_bn2_var) { fprintf(stderr, "malloc failed for l3_b1_bn2_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b1_bn2_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b1_bn2_var.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_bn2_var, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b1_bn2_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_bn2_var, h_l3_b1_bn2_var, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_bn2_gamma = (float*)malloc(1024);
  if (!h_l3_b1_bn2_gamma) { fprintf(stderr, "malloc failed for l3_b1_bn2_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b1_bn2_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b1_bn2_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_bn2_gamma, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b1_bn2_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_bn2_gamma, h_l3_b1_bn2_gamma, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_bn2_beta = (float*)malloc(1024);
  if (!h_l3_b1_bn2_beta) { fprintf(stderr, "malloc failed for l3_b1_bn2_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b1_bn2_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b1_bn2_beta.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_bn2_beta, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b1_bn2_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_bn2_beta, h_l3_b1_bn2_beta, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_shortcut_w = (float*)malloc(131072);
  if (!h_l3_b1_shortcut_w) { fprintf(stderr, "malloc failed for l3_b1_shortcut_w\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b1_shortcut_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b1_shortcut_w.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_shortcut_w, sizeof(float), 32768, f);
    fclose(f);
    if (nread != 32768) { fprintf(stderr, "Failed to read resnet18_fused_l3_b1_shortcut_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_shortcut_w, h_l3_b1_shortcut_w, 131072, cudaMemcpyHostToDevice));
  float* h_l3_b1_shortcut_b = (float*)malloc(1024);
  if (!h_l3_b1_shortcut_b) { fprintf(stderr, "malloc failed for l3_b1_shortcut_b\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b1_shortcut_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b1_shortcut_b.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_shortcut_b, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b1_shortcut_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_shortcut_b, h_l3_b1_shortcut_b, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_shortcut_mean = (float*)malloc(1024);
  if (!h_l3_b1_shortcut_mean) { fprintf(stderr, "malloc failed for l3_b1_shortcut_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b1_shortcut_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b1_shortcut_mean.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_shortcut_mean, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b1_shortcut_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_shortcut_mean, h_l3_b1_shortcut_mean, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_shortcut_var = (float*)malloc(1024);
  if (!h_l3_b1_shortcut_var) { fprintf(stderr, "malloc failed for l3_b1_shortcut_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b1_shortcut_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b1_shortcut_var.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_shortcut_var, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b1_shortcut_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_shortcut_var, h_l3_b1_shortcut_var, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_shortcut_gamma = (float*)malloc(1024);
  if (!h_l3_b1_shortcut_gamma) { fprintf(stderr, "malloc failed for l3_b1_shortcut_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b1_shortcut_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b1_shortcut_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_shortcut_gamma, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b1_shortcut_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_shortcut_gamma, h_l3_b1_shortcut_gamma, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b1_shortcut_beta = (float*)malloc(1024);
  if (!h_l3_b1_shortcut_beta) { fprintf(stderr, "malloc failed for l3_b1_shortcut_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b1_shortcut_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b1_shortcut_beta.bin\n"); return 1; }
    size_t nread = fread(h_l3_b1_shortcut_beta, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b1_shortcut_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b1_shortcut_beta, h_l3_b1_shortcut_beta, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_w1 = (float*)malloc(2359296);
  if (!h_l3_b2_w1) { fprintf(stderr, "malloc failed for l3_b2_w1\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b2_w1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b2_w1.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_w1, sizeof(float), 589824, f);
    fclose(f);
    if (nread != 589824) { fprintf(stderr, "Failed to read resnet18_fused_l3_b2_w1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_w1, h_l3_b2_w1, 2359296, cudaMemcpyHostToDevice));
  float* h_l3_b2_b1 = (float*)malloc(1024);
  if (!h_l3_b2_b1) { fprintf(stderr, "malloc failed for l3_b2_b1\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b2_b1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b2_b1.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_b1, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b2_b1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_b1, h_l3_b2_b1, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_w2 = (float*)malloc(2359296);
  if (!h_l3_b2_w2) { fprintf(stderr, "malloc failed for l3_b2_w2\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b2_w2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b2_w2.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_w2, sizeof(float), 589824, f);
    fclose(f);
    if (nread != 589824) { fprintf(stderr, "Failed to read resnet18_fused_l3_b2_w2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_w2, h_l3_b2_w2, 2359296, cudaMemcpyHostToDevice));
  float* h_l3_b2_b2 = (float*)malloc(1024);
  if (!h_l3_b2_b2) { fprintf(stderr, "malloc failed for l3_b2_b2\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b2_b2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b2_b2.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_b2, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b2_b2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_b2, h_l3_b2_b2, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_bn1_mean = (float*)malloc(1024);
  if (!h_l3_b2_bn1_mean) { fprintf(stderr, "malloc failed for l3_b2_bn1_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b2_bn1_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b2_bn1_mean.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_bn1_mean, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b2_bn1_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_bn1_mean, h_l3_b2_bn1_mean, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_bn1_var = (float*)malloc(1024);
  if (!h_l3_b2_bn1_var) { fprintf(stderr, "malloc failed for l3_b2_bn1_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b2_bn1_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b2_bn1_var.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_bn1_var, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b2_bn1_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_bn1_var, h_l3_b2_bn1_var, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_bn1_gamma = (float*)malloc(1024);
  if (!h_l3_b2_bn1_gamma) { fprintf(stderr, "malloc failed for l3_b2_bn1_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b2_bn1_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b2_bn1_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_bn1_gamma, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b2_bn1_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_bn1_gamma, h_l3_b2_bn1_gamma, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_bn1_beta = (float*)malloc(1024);
  if (!h_l3_b2_bn1_beta) { fprintf(stderr, "malloc failed for l3_b2_bn1_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b2_bn1_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b2_bn1_beta.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_bn1_beta, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b2_bn1_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_bn1_beta, h_l3_b2_bn1_beta, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_bn2_mean = (float*)malloc(1024);
  if (!h_l3_b2_bn2_mean) { fprintf(stderr, "malloc failed for l3_b2_bn2_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b2_bn2_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b2_bn2_mean.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_bn2_mean, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b2_bn2_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_bn2_mean, h_l3_b2_bn2_mean, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_bn2_var = (float*)malloc(1024);
  if (!h_l3_b2_bn2_var) { fprintf(stderr, "malloc failed for l3_b2_bn2_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b2_bn2_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b2_bn2_var.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_bn2_var, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b2_bn2_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_bn2_var, h_l3_b2_bn2_var, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_bn2_gamma = (float*)malloc(1024);
  if (!h_l3_b2_bn2_gamma) { fprintf(stderr, "malloc failed for l3_b2_bn2_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b2_bn2_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b2_bn2_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_bn2_gamma, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b2_bn2_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_bn2_gamma, h_l3_b2_bn2_gamma, 1024, cudaMemcpyHostToDevice));
  float* h_l3_b2_bn2_beta = (float*)malloc(1024);
  if (!h_l3_b2_bn2_beta) { fprintf(stderr, "malloc failed for l3_b2_bn2_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l3_b2_bn2_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l3_b2_bn2_beta.bin\n"); return 1; }
    size_t nread = fread(h_l3_b2_bn2_beta, sizeof(float), 256, f);
    fclose(f);
    if (nread != 256) { fprintf(stderr, "Failed to read resnet18_fused_l3_b2_bn2_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l3_b2_bn2_beta, h_l3_b2_bn2_beta, 1024, cudaMemcpyHostToDevice));
  float* h_l4_b1_w1 = (float*)malloc(4718592);
  if (!h_l4_b1_w1) { fprintf(stderr, "malloc failed for l4_b1_w1\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b1_w1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b1_w1.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_w1, sizeof(float), 1179648, f);
    fclose(f);
    if (nread != 1179648) { fprintf(stderr, "Failed to read resnet18_fused_l4_b1_w1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_w1, h_l4_b1_w1, 4718592, cudaMemcpyHostToDevice));
  float* h_l4_b1_b1 = (float*)malloc(2048);
  if (!h_l4_b1_b1) { fprintf(stderr, "malloc failed for l4_b1_b1\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b1_b1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b1_b1.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_b1, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b1_b1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_b1, h_l4_b1_b1, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_w2 = (float*)malloc(9437184);
  if (!h_l4_b1_w2) { fprintf(stderr, "malloc failed for l4_b1_w2\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b1_w2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b1_w2.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_w2, sizeof(float), 2359296, f);
    fclose(f);
    if (nread != 2359296) { fprintf(stderr, "Failed to read resnet18_fused_l4_b1_w2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_w2, h_l4_b1_w2, 9437184, cudaMemcpyHostToDevice));
  float* h_l4_b1_b2 = (float*)malloc(2048);
  if (!h_l4_b1_b2) { fprintf(stderr, "malloc failed for l4_b1_b2\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b1_b2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b1_b2.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_b2, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b1_b2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_b2, h_l4_b1_b2, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_bn1_mean = (float*)malloc(2048);
  if (!h_l4_b1_bn1_mean) { fprintf(stderr, "malloc failed for l4_b1_bn1_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b1_bn1_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b1_bn1_mean.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_bn1_mean, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b1_bn1_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_bn1_mean, h_l4_b1_bn1_mean, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_bn1_var = (float*)malloc(2048);
  if (!h_l4_b1_bn1_var) { fprintf(stderr, "malloc failed for l4_b1_bn1_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b1_bn1_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b1_bn1_var.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_bn1_var, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b1_bn1_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_bn1_var, h_l4_b1_bn1_var, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_bn1_gamma = (float*)malloc(2048);
  if (!h_l4_b1_bn1_gamma) { fprintf(stderr, "malloc failed for l4_b1_bn1_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b1_bn1_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b1_bn1_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_bn1_gamma, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b1_bn1_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_bn1_gamma, h_l4_b1_bn1_gamma, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_bn1_beta = (float*)malloc(2048);
  if (!h_l4_b1_bn1_beta) { fprintf(stderr, "malloc failed for l4_b1_bn1_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b1_bn1_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b1_bn1_beta.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_bn1_beta, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b1_bn1_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_bn1_beta, h_l4_b1_bn1_beta, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_bn2_mean = (float*)malloc(2048);
  if (!h_l4_b1_bn2_mean) { fprintf(stderr, "malloc failed for l4_b1_bn2_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b1_bn2_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b1_bn2_mean.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_bn2_mean, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b1_bn2_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_bn2_mean, h_l4_b1_bn2_mean, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_bn2_var = (float*)malloc(2048);
  if (!h_l4_b1_bn2_var) { fprintf(stderr, "malloc failed for l4_b1_bn2_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b1_bn2_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b1_bn2_var.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_bn2_var, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b1_bn2_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_bn2_var, h_l4_b1_bn2_var, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_bn2_gamma = (float*)malloc(2048);
  if (!h_l4_b1_bn2_gamma) { fprintf(stderr, "malloc failed for l4_b1_bn2_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b1_bn2_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b1_bn2_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_bn2_gamma, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b1_bn2_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_bn2_gamma, h_l4_b1_bn2_gamma, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_bn2_beta = (float*)malloc(2048);
  if (!h_l4_b1_bn2_beta) { fprintf(stderr, "malloc failed for l4_b1_bn2_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b1_bn2_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b1_bn2_beta.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_bn2_beta, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b1_bn2_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_bn2_beta, h_l4_b1_bn2_beta, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_shortcut_w = (float*)malloc(524288);
  if (!h_l4_b1_shortcut_w) { fprintf(stderr, "malloc failed for l4_b1_shortcut_w\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b1_shortcut_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b1_shortcut_w.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_shortcut_w, sizeof(float), 131072, f);
    fclose(f);
    if (nread != 131072) { fprintf(stderr, "Failed to read resnet18_fused_l4_b1_shortcut_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_shortcut_w, h_l4_b1_shortcut_w, 524288, cudaMemcpyHostToDevice));
  float* h_l4_b1_shortcut_b = (float*)malloc(2048);
  if (!h_l4_b1_shortcut_b) { fprintf(stderr, "malloc failed for l4_b1_shortcut_b\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b1_shortcut_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b1_shortcut_b.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_shortcut_b, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b1_shortcut_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_shortcut_b, h_l4_b1_shortcut_b, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_shortcut_mean = (float*)malloc(2048);
  if (!h_l4_b1_shortcut_mean) { fprintf(stderr, "malloc failed for l4_b1_shortcut_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b1_shortcut_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b1_shortcut_mean.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_shortcut_mean, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b1_shortcut_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_shortcut_mean, h_l4_b1_shortcut_mean, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_shortcut_var = (float*)malloc(2048);
  if (!h_l4_b1_shortcut_var) { fprintf(stderr, "malloc failed for l4_b1_shortcut_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b1_shortcut_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b1_shortcut_var.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_shortcut_var, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b1_shortcut_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_shortcut_var, h_l4_b1_shortcut_var, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_shortcut_gamma = (float*)malloc(2048);
  if (!h_l4_b1_shortcut_gamma) { fprintf(stderr, "malloc failed for l4_b1_shortcut_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b1_shortcut_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b1_shortcut_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_shortcut_gamma, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b1_shortcut_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_shortcut_gamma, h_l4_b1_shortcut_gamma, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b1_shortcut_beta = (float*)malloc(2048);
  if (!h_l4_b1_shortcut_beta) { fprintf(stderr, "malloc failed for l4_b1_shortcut_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b1_shortcut_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b1_shortcut_beta.bin\n"); return 1; }
    size_t nread = fread(h_l4_b1_shortcut_beta, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b1_shortcut_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b1_shortcut_beta, h_l4_b1_shortcut_beta, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_w1 = (float*)malloc(9437184);
  if (!h_l4_b2_w1) { fprintf(stderr, "malloc failed for l4_b2_w1\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b2_w1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b2_w1.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_w1, sizeof(float), 2359296, f);
    fclose(f);
    if (nread != 2359296) { fprintf(stderr, "Failed to read resnet18_fused_l4_b2_w1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_w1, h_l4_b2_w1, 9437184, cudaMemcpyHostToDevice));
  float* h_l4_b2_b1 = (float*)malloc(2048);
  if (!h_l4_b2_b1) { fprintf(stderr, "malloc failed for l4_b2_b1\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b2_b1.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b2_b1.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_b1, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b2_b1.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_b1, h_l4_b2_b1, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_w2 = (float*)malloc(9437184);
  if (!h_l4_b2_w2) { fprintf(stderr, "malloc failed for l4_b2_w2\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b2_w2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b2_w2.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_w2, sizeof(float), 2359296, f);
    fclose(f);
    if (nread != 2359296) { fprintf(stderr, "Failed to read resnet18_fused_l4_b2_w2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_w2, h_l4_b2_w2, 9437184, cudaMemcpyHostToDevice));
  float* h_l4_b2_b2 = (float*)malloc(2048);
  if (!h_l4_b2_b2) { fprintf(stderr, "malloc failed for l4_b2_b2\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b2_b2.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b2_b2.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_b2, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b2_b2.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_b2, h_l4_b2_b2, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_bn1_mean = (float*)malloc(2048);
  if (!h_l4_b2_bn1_mean) { fprintf(stderr, "malloc failed for l4_b2_bn1_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b2_bn1_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b2_bn1_mean.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_bn1_mean, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b2_bn1_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_bn1_mean, h_l4_b2_bn1_mean, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_bn1_var = (float*)malloc(2048);
  if (!h_l4_b2_bn1_var) { fprintf(stderr, "malloc failed for l4_b2_bn1_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b2_bn1_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b2_bn1_var.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_bn1_var, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b2_bn1_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_bn1_var, h_l4_b2_bn1_var, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_bn1_gamma = (float*)malloc(2048);
  if (!h_l4_b2_bn1_gamma) { fprintf(stderr, "malloc failed for l4_b2_bn1_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b2_bn1_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b2_bn1_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_bn1_gamma, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b2_bn1_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_bn1_gamma, h_l4_b2_bn1_gamma, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_bn1_beta = (float*)malloc(2048);
  if (!h_l4_b2_bn1_beta) { fprintf(stderr, "malloc failed for l4_b2_bn1_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b2_bn1_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b2_bn1_beta.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_bn1_beta, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b2_bn1_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_bn1_beta, h_l4_b2_bn1_beta, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_bn2_mean = (float*)malloc(2048);
  if (!h_l4_b2_bn2_mean) { fprintf(stderr, "malloc failed for l4_b2_bn2_mean\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b2_bn2_mean.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b2_bn2_mean.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_bn2_mean, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b2_bn2_mean.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_bn2_mean, h_l4_b2_bn2_mean, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_bn2_var = (float*)malloc(2048);
  if (!h_l4_b2_bn2_var) { fprintf(stderr, "malloc failed for l4_b2_bn2_var\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b2_bn2_var.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b2_bn2_var.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_bn2_var, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b2_bn2_var.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_bn2_var, h_l4_b2_bn2_var, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_bn2_gamma = (float*)malloc(2048);
  if (!h_l4_b2_bn2_gamma) { fprintf(stderr, "malloc failed for l4_b2_bn2_gamma\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b2_bn2_gamma.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b2_bn2_gamma.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_bn2_gamma, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b2_bn2_gamma.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_bn2_gamma, h_l4_b2_bn2_gamma, 2048, cudaMemcpyHostToDevice));
  float* h_l4_b2_bn2_beta = (float*)malloc(2048);
  if (!h_l4_b2_bn2_beta) { fprintf(stderr, "malloc failed for l4_b2_bn2_beta\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_l4_b2_bn2_beta.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_l4_b2_bn2_beta.bin\n"); return 1; }
    size_t nread = fread(h_l4_b2_bn2_beta, sizeof(float), 512, f);
    fclose(f);
    if (nread != 512) { fprintf(stderr, "Failed to read resnet18_fused_l4_b2_bn2_beta.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_l4_b2_bn2_beta, h_l4_b2_bn2_beta, 2048, cudaMemcpyHostToDevice));
  float* h_fc_w = (float*)malloc(2048000);
  if (!h_fc_w) { fprintf(stderr, "malloc failed for fc_w\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_fc_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_fc_w.bin\n"); return 1; }
    size_t nread = fread(h_fc_w, sizeof(float), 512000, f);
    fclose(f);
    if (nread != 512000) { fprintf(stderr, "Failed to read resnet18_fused_fc_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc_w, h_fc_w, 2048000, cudaMemcpyHostToDevice));
  float* h_fc_b = (float*)malloc(4000);
  if (!h_fc_b) { fprintf(stderr, "malloc failed for fc_b\n"); return 1; }
  { FILE* f = fopen("resnet18_fused_fc_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_fc_b.bin\n"); return 1; }
    size_t nread = fread(h_fc_b, sizeof(float), 1000, f);
    fclose(f);
    if (nread != 1000) { fprintf(stderr, "Failed to read resnet18_fused_fc_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc_b, h_fc_b, 4000, cudaMemcpyHostToDevice));
  float h_bn_eps;
  { FILE* f = fopen("resnet18_fused_bn_eps.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open resnet18_fused_bn_eps.bin\n"); return 1; }
    fread(&h_bn_eps, sizeof(float), 1, f);
    fclose(f);
  }
  CUDA_CHECK(cudaMemcpyToSymbol(bn_eps, &h_bn_eps, sizeof(h_bn_eps)));
  CUDA_CHECK(cudaEventRecord(__io_stop));
  CUDA_CHECK(cudaEventSynchronize(__io_stop));
  {
    float __io_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&__io_ms, __io_start, __io_stop));
    printf("MEMTIME_H2D resnet18_fused %.6f\n", __io_ms);
  }

  float* h_output = (float*)malloc(4000);
  if (!h_output) { fprintf(stderr, "malloc failed for output\n"); return 1; }

  cudaEvent_t __timer_start, __timer_stop;
  CUDA_CHECK(cudaEventCreate(&__timer_start));
  CUDA_CHECK(cudaEventCreate(&__timer_stop));
  for (int __w = 0; __w < 5; __w++) resnet18_fused();
  CUDA_CHECK(cudaDeviceSynchronize());
  float __best_ms = 1e30f;
  float __all_ms[20];
  for (int __it = 0; __it < 20; __it++) {
    CUDA_CHECK(cudaEventRecord(__timer_start));
    resnet18_fused();
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
    printf("TIME resnet18_fused %.6f\n", __best_ms);
    printf("TIMEMED resnet18_fused %.6f\n", __all_ms[10]);
  CUDA_CHECK(cudaEventDestroy(__timer_start));
  CUDA_CHECK(cudaEventDestroy(__timer_stop));

  {
    cudaStream_t __gs;
    cudaGraph_t __graph = NULL;
    cudaGraphExec_t __gexec = NULL;
    CUDA_CHECK(cudaStreamCreate(&__gs));
    cudaError_t __ce = cudaStreamBeginCapture(__gs, cudaStreamCaptureModeThreadLocal);
    if (__ce == cudaSuccess) { resnet18_fused_s(__gs); __ce = cudaStreamEndCapture(__gs, &__graph); }
    if (__ce != cudaSuccess) {
      cudaGetLastError();  /* clear poisoned capture state */
      __graph = NULL;
      __ce = cudaStreamBeginCapture(__gs, cudaStreamCaptureModeThreadLocal);
      if (__ce == cudaSuccess) { resnet18_fused_seq_s(__gs); __ce = cudaStreamEndCapture(__gs, &__graph); }
    }
    if (__ce == cudaSuccess && __graph)
      __ce = cudaGraphInstantiate(&__gexec, __graph, NULL, NULL, 0);
    if (__ce == cudaSuccess && __gexec) {
      for (int __w = 0; __w < 3; __w++) CUDA_CHECK(cudaGraphLaunch(__gexec, __gs));
      CUDA_CHECK(cudaStreamSynchronize(__gs));
      cudaEvent_t __ga, __gb;
      CUDA_CHECK(cudaEventCreate(&__ga));
      CUDA_CHECK(cudaEventCreate(&__gb));
      float __gbest = 1e30f;
      float __gall[20];
      for (int __it = 0; __it < 20; __it++) {
        CUDA_CHECK(cudaEventRecord(__ga, __gs));
        CUDA_CHECK(cudaGraphLaunch(__gexec, __gs));
        CUDA_CHECK(cudaEventRecord(__gb, __gs));
        CUDA_CHECK(cudaEventSynchronize(__gb));
        float __gms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&__gms, __ga, __gb));
        __gall[__it] = __gms;
        if (__gms < __gbest) __gbest = __gms;
      }
      for (int __i = 0; __i < 20; __i++)
        for (int __j = __i + 1; __j < 20; __j++)
          if (__gall[__j] < __gall[__i]) { float __t = __gall[__i]; __gall[__i] = __gall[__j]; __gall[__j] = __t; }
      printf("TIME_GRAPH resnet18_fused %.6f\n", __gbest);
      printf("TIMEGRAPH_MED resnet18_fused %.6f\n", __gall[10]);
      CUDA_CHECK(cudaEventDestroy(__ga));
      CUDA_CHECK(cudaEventDestroy(__gb));
    } else {
      printf("TIME_GRAPH resnet18_fused unsupported\n");
    }
  }

  resnet18_fused_kt();

  CUDA_CHECK(cudaEventRecord(__io_start));
  CUDA_CHECK(cudaMemcpy(h_output, d_output, 4000, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(__io_stop));
  CUDA_CHECK(cudaEventSynchronize(__io_stop));
  {
    float __d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&__d2h_ms, __io_start, __io_stop));
    printf("MEMTIME_D2H resnet18_fused %.6f\n", __d2h_ms);
  }
  CUDA_CHECK(cudaEventDestroy(__io_start));
  CUDA_CHECK(cudaEventDestroy(__io_stop));

  FILE* fout = fopen("resnet18_fused_output.bin", "wb");
  if (!fout) { fprintf(stderr, "Cannot open %s\n", "resnet18_fused_output.bin"); return 1; }
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
  CUDA_CHECK(cudaFree(d_l1_b1_add_out));
  CUDA_CHECK(cudaFree(d_l1_b2_add_out));
  CUDA_CHECK(cudaFree(d_l2_b1_ws));
  CUDA_CHECK(cudaFree(d_l2_b1_add_out));
  CUDA_CHECK(cudaFree(d_l2_b2_ws));
  CUDA_CHECK(cudaFree(d_l2_b2_add_out));
  CUDA_CHECK(cudaFree(d_l3_b1_ws));
  CUDA_CHECK(cudaFree(d_l3_b1_add_out));
  CUDA_CHECK(cudaFree(d_l3_b2_ws));
  CUDA_CHECK(cudaFree(d_l3_b2_add_out));
  CUDA_CHECK(cudaFree(d_l4_b1_ws));
  CUDA_CHECK(cudaFree(d_l4_b1_add_out));
  CUDA_CHECK(cudaFree(d_l4_b2_ws));
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
