/* E2 pilot: fused ResNet18 BasicBlock (l1_b2, 64ch, 56x56, identity shortcut)
 *
 * Three variants, same input, same weights:
 *   A: unfused per-op sequence (7 naive kernels, all intermediates in DRAM)
 *      conv -> bn -> relu -> conv -> bn -> add -> relu
 *   B: G1-style tiled per-op kernels (3 kernels)
 *      conv+bn+relu (tiled) -> conv+bn+add (tiled) -> relu (in-place)
 *   C: single fused kernel conv-bn-relu-conv-bn-add-relu
 *      8x8 output tile per block; T1 = relu(bn1(conv1)) tile (64ch x 10x10)
 *      resident in shared memory; shortcut read from global input.
 *
 * Validation: CPU double-precision reference, max abs/rel error reported.
 * Timing: CUDA events around the whole kernel chain, warmup + min-of-N.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#define C 64
#define H 56
#define W 56
#ifndef TILE
#define TILE 8
#endif
#define TH (TILE + 2)   /* T1 halo tile: 10 */
#define IH (TILE + 4)   /* input halo tile: 12 */

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

/* ----------------------------- naive kernels (variant A) ------------- */

__global__ void conv3x3_naive(const float* in, const float* w, const float* b, float* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= C * H * W) return;
  int oc = idx / (H * W);
  int r = idx % (H * W);
  int oy = r / W, ox = r % W;
  float acc = 0.0f;
  for (int ic = 0; ic < C; ic++)
    for (int ky = 0; ky < 3; ky++)
      for (int kx = 0; kx < 3; kx++) {
        int iy = oy + ky - 1, ix = ox + kx - 1;
        if (iy >= 0 && iy < H && ix >= 0 && ix < W)
          acc += in[(ic * H + iy) * W + ix] * w[((oc * C + ic) * 3 + ky) * 3 + kx];
      }
  out[idx] = acc + b[oc];
}

__global__ void bn_naive(float* x, const float* mean, const float* var,
                         const float* gamma, const float* beta, float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= C * H * W) return;
  int c = idx / (H * W);
  x[idx] = (x[idx] - mean[c]) / sqrtf(var[c] + eps) * gamma[c] + beta[c];
}

__global__ void relu_naive(float* x) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= C * H * W) return;
  x[idx] = (x[idx] <= 0.0f) ? 0.0f : x[idx];
}

__global__ void add_naive(float* x, const float* y) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= C * H * W) return;
  x[idx] += y[idx];
}

/* ----------------------- G1-style tiled kernels (variant B) ---------- */
/* Port of the G1 conv2d_tiled_patch template: 16x16 block, 2x2 outputs per
 * thread, 32x32 output tile, 34x35 double-buffered shared input tile,
 * CB=2 output channels per block (blockIdx.z = 32 channel groups). */

#define EPI_BN_RELU(cb) \
  do { \
    int oc = oc0 + cb; \
    float v = acc[cb][dy][dx] + b[oc]; \
    v = (v - mean[oc]) / sqrtf(var[oc] + eps) * gamma[oc] + beta[oc]; \
    v = (v <= 0.0f) ? 0.0f : v; \
    out[(oc * H + oy) * W + ox] = v; \
  } while (0)

#define EPI_BN_ADD(cb) \
  do { \
    int oc = oc0 + cb; \
    float v = acc[cb][dy][dx] + b[oc]; \
    v = (v - mean[oc]) / sqrtf(var[oc] + eps) * gamma[oc] + beta[oc]; \
    v += res[(oc * H + oy) * W + ox]; \
    out[(oc * H + oy) * W + ox] = v; \
  } while (0)

template <int EPI>  /* 0 = bn+relu, 1 = bn+add */
__global__ void conv3x3_tiled(const float* in, const float* w, const float* b,
                              const float* mean, const float* var,
                              const float* gamma, const float* beta, float eps,
                              const float* res, float* out) {
  __shared__ float sh2[2][34][35];
  __shared__ float shw2[2][2][9];
  int tx = threadIdx.x, ty = threadIdx.y;
  int oy0 = blockIdx.y * 32 + ty * 2;
  int ox0 = blockIdx.x * 32 + tx * 2;
  int ih0 = blockIdx.y * 32 - 1;
  int iw0 = blockIdx.x * 32 - 1;
  int oc0 = blockIdx.z * 2;
  float acc[2][2][2];
  #pragma unroll
  for (int cb = 0; cb < 2; cb++)
    for (int dy = 0; dy < 2; dy++)
      for (int dx = 0; dx < 2; dx++) acc[cb][dy][dx] = 0.0f;
  for (int sy = ty; sy < 34; sy += 16)
    for (int sx = tx; sx < 34; sx += 16) {
      int iy = ih0 + sy, ix = iw0 + sx;
      sh2[0][sy][sx] = (iy < 0 || iy >= H || ix < 0 || ix >= W) ? 0.0f
                       : in[(0 * H + iy) * W + ix];
    }
  {
    int tid = ty * 16 + tx;
    if (tid < 18) {
      int cb = tid / 9, t = tid - cb * 9;
      shw2[0][cb][t] = w[((oc0 + cb) * C + 0) * 9 + t];
    }
  }
  __syncthreads();
  for (int c = 0; c < C; c++) {
    int cur = c & 1;
    if (c + 1 < C) {
      for (int sy = ty; sy < 34; sy += 16)
        for (int sx = tx; sx < 34; sx += 16) {
          int iy = ih0 + sy, ix = iw0 + sx;
          sh2[cur ^ 1][sy][sx] = (iy < 0 || iy >= H || ix < 0 || ix >= W) ? 0.0f
                                 : in[((c + 1) * H + iy) * W + ix];
        }
      int tid = ty * 16 + tx;
      if (tid < 18) {
        int cb = tid / 9, t = tid - cb * 9;
        shw2[cur ^ 1][cb][t] = w[((oc0 + cb) * C + (c + 1)) * 9 + t];
      }
    }
    float win[4][4];
    #pragma unroll
    for (int wy = 0; wy < 4; wy++)
      #pragma unroll
      for (int wx = 0; wx < 4; wx++)
        win[wy][wx] = sh2[cur][ty * 2 + wy][tx * 2 + wx];
    #pragma unroll
    for (int cb = 0; cb < 2; cb++)
      #pragma unroll
      for (int k1 = 0; k1 < 3; k1++)
        #pragma unroll
        for (int k2 = 0; k2 < 3; k2++) {
          float wv = shw2[cur][cb][k1 * 3 + k2];
          acc[cb][0][0] += win[0 + k1][0 + k2] * wv;
          acc[cb][0][1] += win[0 + k1][1 + k2] * wv;
          acc[cb][1][0] += win[1 + k1][0 + k2] * wv;
          acc[cb][1][1] += win[1 + k1][1 + k2] * wv;
        }
    __syncthreads();
  }
  #pragma unroll
  for (int cb = 0; cb < 2; cb++) {
    #pragma unroll
    for (int dy = 0; dy < 2; dy++)
      #pragma unroll
      for (int dx = 0; dx < 2; dx++) {
        int oy = oy0 + dy, ox = ox0 + dx;
        if (oy < H && ox < W) {
          if (EPI == 0) { EPI_BN_RELU(cb); } else { EPI_BN_ADD(cb); }
        }
      }
  }
}

__global__ void relu_inplace(float* x) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= C * H * W) return;
  x[idx] = (x[idx] <= 0.0f) ? 0.0f : x[idx];
}

/* ------------------------- fused kernel (variant C) ------------------ */
/* Block computes one TILE x TILE output tile for all 64 channels.
 * Phase 1: T1[c][py][px] = relu(bn1(conv1(in))) for the (TILE+2)^2 halo tile,
 *          kept in shared memory (64ch x 10x10 x 4B = 25.6 KB for TILE=8).
 *          T1 positions outside the 56x56 map are conv2's zero padding.
 *          Per input-channel step: stage the (TILE+4)^2 input tile AND the
 *          w1[:,ic,:,:] slice (64*9 floats) into shared, one sync per step.
 * Phase 2: ic-outer accumulation into per-thread output registers; per step
 *          stage the w2[:,ic,:,:] slice into shared. Epilogue: bn2, identity
 *          shortcut from global input (L2), relu, single global write.
 */
/* Thread mapping: one-time geometry, no per-step integer division.
 * Phase 1: sp = tid & 127 indexes the TH*TH spatial slots (sp >= TH*TH idle
 *          in compute but helps staging); cg = tid >> 7 selects one of
 *          SG1 = NTHR/128 channel groups; thread accumulates KC1 = C/SG1
 *          channels at its fixed spatial slot (acc registers).
 * Phase 2: sp2 = tid & (TILE*TILE-1) is the output pixel; cg2 = tid / (TILE*TILE)
 *          selects one of SG2 = NTHR/(TILE*TILE) channel groups; KC2 = C/SG2
 *          output channels per thread. */
#ifndef NTHR
#define NTHR 512
#endif
#define SG1 (NTHR / 128)
#define KC1 (C / SG1)
#define SP2 (TILE * TILE)
#define SG2 (NTHR / SP2)
#define KC2 (C / SG2)

__global__ void basicblock_fused(const float* in,
                                 const float* w1, const float* b1,
                                 const float* m1, const float* v1,
                                 const float* g1, const float* be1,
                                 const float* w2, const float* b2,
                                 const float* m2, const float* v2,
                                 const float* g2, const float* be2,
                                 float eps, float* out) {
  __shared__ float shT1[C][TH][TH];
  __shared__ float shIn[IH][IH];
  __shared__ float shW[C][9];
  const int tid = threadIdx.x;   /* NTHR threads */
  const int tx0 = blockIdx.x * TILE;
  const int ty0 = blockIdx.y * TILE;

  /* Phase 1: T1 = relu(bn1(conv1)) into shared */
  const int sp = tid & 127;
  const int cg = tid >> 7;
  const int py = sp / TH, px = sp % TH;
  const bool act1 = (sp < TH * TH);
  float acc[KC1];
  #pragma unroll
  for (int k = 0; k < KC1; k++) acc[k] = 0.0f;
  for (int ic = 0; ic < C; ic++) {
    for (int s = tid; s < IH * IH; s += NTHR) {
      int sy = s / IH, sx = s % IH;
      int iy = ty0 - 2 + sy, ix = tx0 - 2 + sx;
      shIn[sy][sx] = (iy < 0 || iy >= H || ix < 0 || ix >= W) ? 0.0f
                     : in[(ic * H + iy) * W + ix];
    }
    for (int s = tid; s < C * 9; s += NTHR)
      shW[s / 9][s % 9] = w1[((s / 9) * C + ic) * 9 + s % 9];
    __syncthreads();
    if (act1) {
      #pragma unroll
      for (int k = 0; k < KC1; k++) {
        int c = cg * KC1 + k;
        float a = acc[k];
        #pragma unroll
        for (int ky = 0; ky < 3; ky++)
          #pragma unroll
          for (int kx = 0; kx < 3; kx++)
            a += shIn[py + ky][px + kx] * shW[c][ky * 3 + kx];
        acc[k] = a;
      }
    }
    __syncthreads();
  }
  if (act1) {
    int gy = ty0 - 1 + py, gx = tx0 - 1 + px;
    bool inmap = (gy >= 0 && gy < H && gx >= 0 && gx < W);
    #pragma unroll
    for (int k = 0; k < KC1; k++) {
      int c = cg * KC1 + k;
      float v = acc[k] + b1[c];
      v = (v - m1[c]) / sqrtf(v1[c] + eps) * g1[c] + be1[c];
      v = (v <= 0.0f) ? 0.0f : v;
      /* outside the 56x56 map T1 is conv2's zero padding */
      shT1[c][py][px] = inmap ? v : 0.0f;
    }
  }
  __syncthreads();

  /* Phase 2: conv2 + bn2 + shortcut-add + relu, ic-outer accumulation */
  const int sp2 = tid & (SP2 - 1);
  const int cg2 = tid / SP2;
  const int y = sp2 / TILE, x = sp2 % TILE;
  float acc2[KC2];
  #pragma unroll
  for (int k = 0; k < KC2; k++) acc2[k] = 0.0f;
  for (int ic = 0; ic < C; ic++) {
    for (int s = tid; s < C * 9; s += NTHR)
      shW[s / 9][s % 9] = w2[((s / 9) * C + ic) * 9 + s % 9];
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < KC2; k++) {
      int oc = cg2 * KC2 + k;
      float a = acc2[k];
      #pragma unroll
      for (int ky = 0; ky < 3; ky++)
        #pragma unroll
        for (int kx = 0; kx < 3; kx++)
          a += shT1[ic][y + ky][x + kx] * shW[oc][ky * 3 + kx];
      acc2[k] = a;
    }
    __syncthreads();
  }
  {
    int gy = ty0 + y, gx = tx0 + x;
    #pragma unroll
    for (int k = 0; k < KC2; k++) {
      int oc = cg2 * KC2 + k;
      float a = acc2[k] + b2[oc];
      a = (a - m2[oc]) / sqrtf(v2[oc] + eps) * g2[oc] + be2[oc];
      float v = a + in[(oc * H + gy) * W + gx];   /* identity shortcut */
      out[(oc * H + gy) * W + gx] = (v <= 0.0f) ? 0.0f : v;
    }
  }
}

/* ------------------------------- host -------------------------------- */

static float* load_bin(const char* path, size_t n) {
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  float* p = (float*)malloc(n * sizeof(float));
  if (fread(p, sizeof(float), n, f) != n) { fprintf(stderr, "short read %s\n", path); exit(1); }
  fclose(f);
  return p;
}

int main(int argc, char** argv) {
  int reps = (argc > 1) ? atoi(argv[1]) : 100;
  const int N = C * H * W;

  float* h_w1  = load_bin("resnet18_test_l1_b2_w1.bin", C * C * 9);
  float* h_b1  = load_bin("resnet18_test_l1_b2_b1.bin", C);
  float* h_m1  = load_bin("resnet18_test_l1_b2_bn1_mean.bin", C);
  float* h_v1  = load_bin("resnet18_test_l1_b2_bn1_var.bin", C);
  float* h_g1  = load_bin("resnet18_test_l1_b2_bn1_gamma.bin", C);
  float* h_be1 = load_bin("resnet18_test_l1_b2_bn1_beta.bin", C);
  float* h_w2  = load_bin("resnet18_test_l1_b2_w2.bin", C * C * 9);
  float* h_b2  = load_bin("resnet18_test_l1_b2_b2.bin", C);
  float* h_m2  = load_bin("resnet18_test_l1_b2_bn2_mean.bin", C);
  float* h_v2  = load_bin("resnet18_test_l1_b2_bn2_var.bin", C);
  float* h_g2  = load_bin("resnet18_test_l1_b2_bn2_gamma.bin", C);
  float* h_be2 = load_bin("resnet18_test_l1_b2_bn2_beta.bin", C);
  float  eps   = load_bin("resnet18_test_bn_eps.bin", 1)[0];

  /* deterministic pseudo-random input in [-1,1] (LCG) */
  float* h_in = (float*)malloc(N * sizeof(float));
  unsigned long long st = 0x9e3779b97f4a7c15ULL;
  for (int i = 0; i < N; i++) {
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    h_in[i] = ((double)(st >> 33) / (double)(1u << 31)) * 2.0 - 1.0;
  }

  /* CPU double-precision reference */
  double* t1 = (double*)malloc(N * sizeof(double));
  double* ref = (double*)malloc(N * sizeof(double));
  for (int oc = 0; oc < C; oc++)
    for (int oy = 0; oy < H; oy++)
      for (int ox = 0; ox < W; ox++) {
        double a = 0.0;
        for (int ic = 0; ic < C; ic++)
          for (int ky = 0; ky < 3; ky++)
            for (int kx = 0; kx < 3; kx++) {
              int iy = oy + ky - 1, ix = ox + kx - 1;
              if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                a += (double)h_in[(ic * H + iy) * W + ix] *
                     (double)h_w1[((oc * C + ic) * 3 + ky) * 3 + kx];
            }
        a += h_b1[oc];
        a = (a - h_m1[oc]) / sqrt(h_v1[oc] + (double)eps) * h_g1[oc] + h_be1[oc];
        t1[(oc * H + oy) * W + ox] = (a <= 0.0) ? 0.0 : a;
      }
  for (int oc = 0; oc < C; oc++)
    for (int oy = 0; oy < H; oy++)
      for (int ox = 0; ox < W; ox++) {
        double a = 0.0;
        for (int ic = 0; ic < C; ic++)
          for (int ky = 0; ky < 3; ky++)
            for (int kx = 0; kx < 3; kx++) {
              int iy = oy + ky - 1, ix = ox + kx - 1;
              if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                a += t1[(ic * H + iy) * W + ix] *
                     (double)h_w2[((oc * C + ic) * 3 + ky) * 3 + kx];
            }
        a += h_b2[oc];
        a = (a - h_m2[oc]) / sqrt(h_v2[oc] + (double)eps) * h_g2[oc] + h_be2[oc];
        a += (double)h_in[(oc * H + oy) * W + ox];
        ref[(oc * H + oy) * W + ox] = (a <= 0.0) ? 0.0 : a;
      }
  fprintf(stderr, "cpu reference done\n");

  /* device buffers */
  float *d_in, *d_out, *d_t1, *d_t2;
  float *d_w1, *d_b1, *d_m1, *d_v1, *d_g1, *d_be1;
  float *d_w2, *d_b2, *d_m2, *d_v2, *d_g2, *d_be2;
  CUDA_CHECK(cudaMalloc(&d_in, N * 4));
  CUDA_CHECK(cudaMalloc(&d_out, N * 4));
  CUDA_CHECK(cudaMalloc(&d_t1, N * 4));
  CUDA_CHECK(cudaMalloc(&d_t2, N * 4));
  CUDA_CHECK(cudaMalloc(&d_w1, C * C * 9 * 4)); CUDA_CHECK(cudaMalloc(&d_b1, C * 4));
  CUDA_CHECK(cudaMalloc(&d_m1, C * 4)); CUDA_CHECK(cudaMalloc(&d_v1, C * 4));
  CUDA_CHECK(cudaMalloc(&d_g1, C * 4)); CUDA_CHECK(cudaMalloc(&d_be1, C * 4));
  CUDA_CHECK(cudaMalloc(&d_w2, C * C * 9 * 4)); CUDA_CHECK(cudaMalloc(&d_b2, C * 4));
  CUDA_CHECK(cudaMalloc(&d_m2, C * 4)); CUDA_CHECK(cudaMalloc(&d_v2, C * 4));
  CUDA_CHECK(cudaMalloc(&d_g2, C * 4)); CUDA_CHECK(cudaMalloc(&d_be2, C * 4));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, N * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w1, h_w1, C * C * 9 * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b1, h_b1, C * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_m1, h_m1, C * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v1, h_v1, C * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_g1, h_g1, C * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_be1, h_be1, C * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w2, h_w2, C * C * 9 * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b2, h_b2, C * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_m2, h_m2, C * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v2, h_v2, C * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_g2, h_g2, C * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_be2, h_be2, C * 4, cudaMemcpyHostToDevice));

  float* h_out = (float*)malloc(N * 4);
  const int thr = 256;
  const int grd = (N + thr - 1) / thr;
  dim3 tgrid(2, 2, 32), tblock(16, 16);
  dim3 fgrid(W / TILE, H / TILE);
  cudaEvent_t eva, evb;
  CUDA_CHECK(cudaEventCreate(&eva));
  CUDA_CHECK(cudaEventCreate(&evb));

  auto check = [&](const char* name) {
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * 4, cudaMemcpyDeviceToHost));
    double maxabs = 0.0, maxrel = 0.0;
    for (int i = 0; i < N; i++) {
      double d = fabs((double)h_out[i] - ref[i]);
      if (d > maxabs) maxabs = d;
      double r = d / (fabs(ref[i]) + 1e-6);
      if (r > maxrel) maxrel = r;
    }
    printf("CHECK %s max_abs=%.3e max_rel=%.3e %s\n", name, maxabs, maxrel,
           (maxabs <= 1e-5) ? "PASS(1e-5)" : "FAIL(1e-5)");
  };

  /* variant A: unfused sequence */
  conv3x3_naive<<<grd, thr>>>(d_in, d_w1, d_b1, d_t1);
  bn_naive<<<grd, thr>>>(d_t1, d_m1, d_v1, d_g1, d_be1, eps);
  relu_naive<<<grd, thr>>>(d_t1);
  conv3x3_naive<<<grd, thr>>>(d_t1, d_w2, d_b2, d_t2);
  bn_naive<<<grd, thr>>>(d_t2, d_m2, d_v2, d_g2, d_be2, eps);
  add_naive<<<grd, thr>>>(d_t2, d_in);
  relu_naive<<<grd, thr>>>(d_t2);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(d_out, d_t2, N * 4, cudaMemcpyDeviceToDevice));
  check("A_unfused");
  float bestA = 1e30f;
  for (int i = 0; i < reps; i++) {
    CUDA_CHECK(cudaEventRecord(eva));
    conv3x3_naive<<<grd, thr>>>(d_in, d_w1, d_b1, d_t1);
    bn_naive<<<grd, thr>>>(d_t1, d_m1, d_v1, d_g1, d_be1, eps);
    relu_naive<<<grd, thr>>>(d_t1);
    conv3x3_naive<<<grd, thr>>>(d_t1, d_w2, d_b2, d_t2);
    bn_naive<<<grd, thr>>>(d_t2, d_m2, d_v2, d_g2, d_be2, eps);
    add_naive<<<grd, thr>>>(d_t2, d_in);
    relu_naive<<<grd, thr>>>(d_t2);
    CUDA_CHECK(cudaEventRecord(evb));
    CUDA_CHECK(cudaEventSynchronize(evb));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, eva, evb));
    if (ms < bestA) bestA = ms;
  }
  CUDA_CHECK(cudaGetLastError());

  /* variant B: G1-style tiled */
  conv3x3_tiled<0><<<tgrid, tblock>>>(d_in, d_w1, d_b1, d_m1, d_v1, d_g1, d_be1, eps, NULL, d_t1);
  conv3x3_tiled<1><<<tgrid, tblock>>>(d_t1, d_w2, d_b2, d_m2, d_v2, d_g2, d_be2, eps, d_in, d_out);
  relu_inplace<<<grd, thr>>>(d_out);
  CUDA_CHECK(cudaGetLastError());
  check("B_g1tiled");
  float bestB = 1e30f;
  for (int i = 0; i < reps; i++) {
    CUDA_CHECK(cudaEventRecord(eva));
    conv3x3_tiled<0><<<tgrid, tblock>>>(d_in, d_w1, d_b1, d_m1, d_v1, d_g1, d_be1, eps, NULL, d_t1);
    conv3x3_tiled<1><<<tgrid, tblock>>>(d_t1, d_w2, d_b2, d_m2, d_v2, d_g2, d_be2, eps, d_in, d_out);
    relu_inplace<<<grd, thr>>>(d_out);
    CUDA_CHECK(cudaEventRecord(evb));
    CUDA_CHECK(cudaEventSynchronize(evb));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, eva, evb));
    if (ms < bestB) bestB = ms;
  }
  CUDA_CHECK(cudaGetLastError());

  /* variant C: fully fused single kernel */
  basicblock_fused<<<fgrid, NTHR>>>(d_in, d_w1, d_b1, d_m1, d_v1, d_g1, d_be1,
                                   d_w2, d_b2, d_m2, d_v2, d_g2, d_be2, eps, d_out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  check("C_fused");
  float bestC = 1e30f;
  for (int i = 0; i < reps; i++) {
    CUDA_CHECK(cudaEventRecord(eva));
    basicblock_fused<<<fgrid, NTHR>>>(d_in, d_w1, d_b1, d_m1, d_v1, d_g1, d_be1,
                                     d_w2, d_b2, d_m2, d_v2, d_g2, d_be2, eps, d_out);
    CUDA_CHECK(cudaEventRecord(evb));
    CUDA_CHECK(cudaEventSynchronize(evb));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, eva, evb));
    if (ms < bestC) bestC = ms;
  }
  CUDA_CHECK(cudaGetLastError());

  double flops = 2.0 * 2.0 * (double)C * C * 9 * H * W;  /* 2 convs, 1 MAC = 2 FLOP */
  printf("TIME A_unfused %.4f ms  (%.2f GFLOP/s)\n", bestA, flops / bestA / 1e6);
  printf("TIME B_g1tiled %.4f ms  (%.2f GFLOP/s)\n", bestB, flops / bestB / 1e6);
  printf("TIME C_fused   %.4f ms  (%.2f GFLOP/s)\n", bestC, flops / bestC / 1e6);
  printf("SPEEDUP C vs A: %.2fx   C vs B: %.2fx\n", bestA / bestC, bestB / bestC);
  return 0;
}
