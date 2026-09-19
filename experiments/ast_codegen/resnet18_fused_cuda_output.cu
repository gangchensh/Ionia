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
