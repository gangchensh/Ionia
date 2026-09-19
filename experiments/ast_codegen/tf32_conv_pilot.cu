// TF32 tensor-core convolution pilot (2026-08-28).
// Standalone pilot: does NOT touch the existing generators (fused_block etc.).
//
// 3x3 stride-1 pad-1 conv as implicit GEMM: C[CO, N=H*W] = W[CO, K=CI*9] * Xcol[K, N].
// K is ordered r-major (k' = r*CI + ci, ci fastest) and the input is
// pre-transposed to HWC on the host, so B-tile staging is float4 vectorized
// along channels (the cuDNN NHWC trick); each k-tile (BK=32, CI%32==0) lies
// within a single (kh,kw) tap.  Weights are pre-permuted to [CO][9][CI].
//
// Two kernels, identical tiling (BM=64, BN=64, BK=32, 128 threads):
//   - conv3x3_fp32 : CUDA-core FMA, 8x4 register tile per thread, float4 smem reads
//   - conv3x3_tf32 : nvcuda::wmma m16n16k8 precision::tf32 fragments, 4 warps x 2x2 tiles
// so the measured delta isolates the tensor-core (TF32) vs CUDA-core (FP32) difference.
// Optional split-K (gridDim.z) with a deterministic ordered reduction kernel.
// Requires CO%64==0 and CI%32==0 (true for all ResNet18 3x3 convs).
//
// Host: loads input/weight bins (validate_tf32_pilot.py), does the HWC/weight
// permutation on host, runs both kernels, dumps outputs (in original C[CO][N]
// layout), benchmarks warmup 10 + 20 timed iters (CUDA events, min-of-20).
//
// Build: /tmp/cuda-12.9/bin/nvcc -O3 -arch=sm_120 tf32_conv_pilot.cu -o tf32_pilot/tf32_conv_pilot
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
              cudaGetErrorString(err));                                     \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

#define BM 64
#define BN 64
#define BK 32
#define LDW 40 /* smem leading dim for wmma tiles (multiple of 8, padded) */

/* cp.async 16B with zero-fill on predicate-off (sm_80+, works on sm_120).
   Set PILOT_SYNC_STAGING=1 to fall back to synchronous copies (debug). */
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem,
                                           bool pred) {
#if PILOT_SYNC_STAGING
  *(float4*)smem = pred ? *(const float4*)gmem : make_float4(0, 0, 0, 0);
#else
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  int sz = pred ? 16 : 0;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(s),
               "l"(gmem), "r"(sz));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if !PILOT_SYNC_STAGING
  asm volatile("cp.async.commit_group;\n");
#endif
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if !PILOT_SYNC_STAGING
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// Per-block smem LUTs for the spatial decode of n (computed once per block).
#define MAKE_LUTS                                                          \
  __shared__ int oh_lut[BN], ow_lut[BN];                                   \
  __shared__ bool n_valid[BN];                                             \
  if (tid < BN) {                                                          \
    int gn = n0 + tid;                                                     \
    n_valid[tid] = gn < N;                                                 \
    oh_lut[tid] = n_valid[tid] ? gn / W : 0;                               \
    ow_lut[tid] = n_valid[tid] ? gn % W : 0;                               \
  }                                                                        \
  __syncthreads();

// Stage the B (im2col) tile: 512 float4 tasks (64 n x 8 quads), each loads
// Xhwc[(ih*W+iw)*CI + ci0 + 4q .. +3].  kh/kw/ci0 are uniform per tile.
#define STAGE_B(WRITE4)                                                    \
  for (int idx = tid; idx < BN * (BK / 4); idx += 128) {                   \
    int n = idx >> 3, q = idx & 7;                                         \
    int ih = oh_lut[n] + kh - 1, iw = ow_lut[n] + kw - 1;                  \
    float4 g = make_float4(0.f, 0.f, 0.f, 0.f);                            \
    if (n_valid[n] && ih >= 0 && ih < H && iw >= 0 && iw < W)              \
      g = *(const float4*)&X[(long)(ih * W + iw) * CI + ci0 + q * 4];      \
    WRITE4;                                                                \
  }

// ---------------------------------------------------------------------------
// TF32 tensor-core kernel: implicit GEMM via wmma m16n16k8 tf32.
// A tile As[m][k] (row-major), B tile Bs[n][k] (col-major view of im2col).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(128, 2)
conv3x3_tf32(const float* __restrict__ X, const float* __restrict__ Wt,
             float* __restrict__ Y, int CI, int CO, int H, int W, int kchunk,
             long out_stride) {
  using namespace nvcuda;
  /* Double-buffered staging via cp.async.  Cs (BM x (BN+4) = 17.4KB) aliases
     the staging area (2 x 20.5KB) after the k-loop; the trailing
     __syncthreads() of the last k-tile makes the overwrite safe. */
  __shared__ float smem_raw[2 * (BM + BN) * LDW];
  float(*As)[LDW] = (float(*)[LDW])smem_raw;
  float(*Bs)[LDW] = (float(*)[LDW])(smem_raw + BM * LDW);
  float(*Cs)[BN + 4] = (float(*)[BN + 4])smem_raw;
  const int N = H * W;
  const int K = CI * 9;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int wm = (warp >> 1) * 32; /* 2x2 warp grid, each warp a 32x32 tile */
  const int wn = (warp & 1) * 32;
  const int kbeg = blockIdx.z * kchunk;
  const int kend = min(K, kbeg + kchunk);
  float* Out = Y + (long)blockIdx.z * out_stride;

  MAKE_LUTS

  wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[2][2];
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.0f);

  /* async staging of one k-tile into buffer `buf` (16B granularity) */
  auto stage = [&](int buf, int k0) {
    /* CI is a power of two and %32==0: the tile sits inside one tap r */
    int r = k0 / CI, ci0 = k0 - r * CI;
    int kh = (r * 11) >> 5, kw = r - 3 * ((r * 11) >> 5);
    float(*a)[LDW] = As + buf * (BM + BN);
    float(*b)[LDW] = Bs + buf * (BM + BN);
    for (int idx = tid; idx < BM * (BK / 4); idx += 128) {
      int m = idx >> 3, q = idx & 7;
      cp_async16(&a[m][q * 4], &Wt[(long)(m0 + m) * K + k0 + q * 4], true);
    }
    for (int idx = tid; idx < BN * (BK / 4); idx += 128) {
      int n = idx >> 3, q = idx & 7;
      int ih = oh_lut[n] + kh - 1, iw = ow_lut[n] + kw - 1;
      bool ok = n_valid[n] && ih >= 0 && ih < H && iw >= 0 && iw < W;
      cp_async16(&b[n][q * 4],
                 ok ? &X[(long)(ih * W + iw) * CI + ci0 + q * 4] : X, ok);
    }
    cp_async_commit();
  };

  stage(0, kbeg);
  int buf = 0;
  for (int k0 = kbeg; k0 < kend; k0 += BK, buf ^= 1) {
    if (k0 + BK < kend) {
      stage(buf ^ 1, k0 + BK);
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    const float(*a)[LDW] = As + buf * (BM + BN);
    const float(*b)[LDW] = Bs + buf * (BM + BN);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 8) {
#pragma unroll
      for (int i = 0; i < 2; i++) {
        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                       wmma::row_major>
            af;
        wmma::load_matrix_sync(af, &a[wm + i * 16][kk], LDW);
#pragma unroll
        for (int j = 0; j < 2; j++) {
          wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32,
                         wmma::col_major>
              bf;
          wmma::load_matrix_sync(bf, &b[wn + j * 16][kk], LDW);
          wmma::mma_sync(acc[i][j], af, bf, acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      wmma::store_matrix_sync(&Cs[wm + i * 16][wn + j * 16], acc[i][j], BN + 4,
                              wmma::mem_row_major);
  __syncthreads();
  for (int idx = tid; idx < BM * BN; idx += 128) {
    int m = idx / BN, n = idx % BN;
    if (n_valid[n]) Out[(long)(m0 + m) * N + n0 + n] = Cs[m][n];
  }
}

// ---------------------------------------------------------------------------
// FP32 CUDA-core kernel: same tiling, 8x4 register block per thread
// (16x8 thread grid), k-major smem, float4 vector reads.
// ---------------------------------------------------------------------------
__global__ void conv3x3_fp32(const float* __restrict__ X,
                             const float* __restrict__ Wt,
                             float* __restrict__ Y, int CI, int CO, int H,
                             int W, int kchunk, long out_stride) {
  __shared__ float As[BK][BM + 4];
  __shared__ float Bs[BK][BN + 4];
  const int N = H * W;
  const int K = CI * 9;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int tid = threadIdx.x;
  const int tx = tid & 15; /* n dim: 16 threads x 4 cols = 64 */
  const int ty = tid >> 4; /* m dim: 8 threads x 8 rows = 64 */
  const int kbeg = blockIdx.z * kchunk;
  const int kend = min(K, kbeg + kchunk);
  float* Out = Y + (long)blockIdx.z * out_stride;

  MAKE_LUTS

  float c[8][4];
#pragma unroll
  for (int i = 0; i < 8; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) c[i][j] = 0.f;

  for (int k0 = kbeg; k0 < kend; k0 += BK) {
    int r = k0 / CI, ci0 = k0 - r * CI;
    int kh = (r * 11) >> 5, kw = r - 3 * ((r * 11) >> 5);
    for (int idx = tid; idx < BM * (BK / 4); idx += 128) {
      int m = idx >> 3, q = idx & 7;
      float4 g = *(const float4*)&Wt[(long)(m0 + m) * K + k0 + q * 4];
      As[q * 4 + 0][m] = g.x;
      As[q * 4 + 1][m] = g.y;
      As[q * 4 + 2][m] = g.z;
      As[q * 4 + 3][m] = g.w;
    }
    STAGE_B({
      Bs[q * 4 + 0][n] = g.x;
      Bs[q * 4 + 1][n] = g.y;
      Bs[q * 4 + 2][n] = g.z;
      Bs[q * 4 + 3][n] = g.w;
    })
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; k++) {
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

// Deterministic ordered reduction over split-K partials.
__global__ void reduce_z(const float* __restrict__ P, float* __restrict__ Y,
                         long total, int Z) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    float s = 0.f;
    for (int z = 0; z < Z; z++) s += P[(long)z * total + i];
    Y[i] = s;
  }
}

// ---------------------------------------------------------------------------
// Host: shapes, IO, benchmark (min-of-20, CUDA events).
// ---------------------------------------------------------------------------
struct Shape {
  const char* name;
  int ci, co, hw, zsplit;
};

static float* read_bin(const char* path, long n) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path);
    exit(1);
  }
  float* p = (float*)malloc(n * sizeof(float));
  if (fread(p, sizeof(float), n, f) != (size_t)n) {
    fprintf(stderr, "short read %s\n", path);
    exit(1);
  }
  fclose(f);
  return p;
}

static void write_bin(const char* path, const float* p, long n) {
  FILE* f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "cannot write %s\n", path);
    exit(1);
  }
  fwrite(p, sizeof(float), n, f);
  fclose(f);
}

int main(int argc, char** argv) {
  const char* dir = (argc > 1) ? argv[1] : "tf32_pilot";
  Shape shapes[] = {
      {"l1_64ch_56x56", 64, 64, 56, 2},
      {"l2_128ch_28x28", 128, 128, 28, 4},
      {"l4_512ch_7x7", 512, 512, 7, 8},
  };
  for (int s = 0; s < 3 && 2 + s < argc; s++)
    shapes[s].zsplit = atoi(argv[2 + s]);
  char path[512];
  printf("%-16s %8s %3s %10s %10s %8s %8s %8s\n", "shape", "MFLOP", "Z",
         "fp32_us", "tf32_us", "fp32_TF", "tf32_TF", "speedup");
  for (int s = 0; s < 3; s++) {
    int CI = shapes[s].ci, CO = shapes[s].co, H = shapes[s].hw, W = H;
    int Z = shapes[s].zsplit;
    long N = (long)H * W, K = (long)CI * 9;
    if (CO % BM || CI % BK || (CI & (CI - 1))) {
      fprintf(stderr, "shape %s violates CO%%64/CI%%32/pow2\n", shapes[s].name);
      return 1;
    }
    int kchunk = (int)((K + (long)BK * Z - 1) / ((long)BK * Z)) * BK;
    double mflop = 2.0 * CO * K * N / 1e6;

    snprintf(path, sizeof(path), "%s/%s_input.bin", dir, shapes[s].name);
    float* hx = read_bin(path, (long)CI * N);
    snprintf(path, sizeof(path), "%s/%s_weight.bin", dir, shapes[s].name);
    float* hw = read_bin(path, (long)CO * K);

    /* host-side layout transforms: input CHW -> HWC, weight k = ci*9+r -> r*CI+ci */
    float* hxh = (float*)malloc((long)CI * N * 4);
    for (long p = 0; p < N; p++)
      for (int c = 0; c < CI; c++) hxh[p * CI + c] = hx[c * N + p];
    float* hwp = (float*)malloc((long)CO * K * 4);
    for (long co = 0; co < CO; co++)
      for (int r = 0; r < 9; r++)
        for (int c = 0; c < CI; c++)
          hwp[co * K + r * CI + c] = hw[co * K + c * 9 + r];

    /* PILOT_PREROUND=1: round inputs to tf32 (RNE, 10 explicit mantissa bits)
       on host — isolates the tensor core's internal conversion mode. */
    if (getenv("PILOT_PREROUND")) {
      long nx = (long)CI * N, nw = (long)CO * K;
      for (long i = 0; i < nx; i++) {
        unsigned b;
        memcpy(&b, &hxh[i], 4);
        b = (b + 0x1000u + ((b >> 13) & 1u)) & ~0x1FFFu;
        memcpy(&hxh[i], &b, 4);
      }
      for (long i = 0; i < nw; i++) {
        unsigned b;
        memcpy(&b, &hwp[i], 4);
        b = (b + 0x1000u + ((b >> 13) & 1u)) & ~0x1FFFu;
        memcpy(&hwp[i], &b, 4);
      }
    }

    float *dx, *dw, *dy, *dp = nullptr;
    CUDA_CHECK(cudaMalloc(&dx, (long)CI * N * 4));
    CUDA_CHECK(cudaMalloc(&dw, (long)CO * K * 4));
    CUDA_CHECK(cudaMalloc(&dy, (long)CO * N * 4));
    if (Z > 1) CUDA_CHECK(cudaMalloc(&dp, (long)CO * N * Z * 4));
    CUDA_CHECK(cudaMemcpy(dx, hxh, (long)CI * N * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw, hwp, (long)CO * K * 4, cudaMemcpyHostToDevice));

    dim3 grid((N + BN - 1) / BN, CO / BM, Z);
    long total = (long)CO * N;
    float* hy = (float*)malloc(total * 4);
    float* kout = (Z > 1) ? dp : dy;
    long rthreads = 256, rblocks = (total + rthreads - 1) / rthreads;

    float best[2] = {1e30f, 1e30f};
    for (int impl = 0; impl < 2; impl++) { /* 0 = fp32, 1 = tf32 */
      for (int it = 0; it < 30; it++) {   /* warmup 10 + timed 20 */
        cudaEvent_t ea, eb;
        CUDA_CHECK(cudaEventCreate(&ea));
        CUDA_CHECK(cudaEventCreate(&eb));
        CUDA_CHECK(cudaEventRecord(ea));
        if (impl == 0)
          conv3x3_fp32<<<grid, 128>>>(dx, dw, kout, CI, CO, H, W, kchunk, total);
        else
          conv3x3_tf32<<<grid, 128>>>(dx, dw, kout, CI, CO, H, W, kchunk, total);
        if (Z > 1) reduce_z<<<rblocks, rthreads>>>(dp, dy, total, Z);
        CUDA_CHECK(cudaEventRecord(eb));
        CUDA_CHECK(cudaEventSynchronize(eb));
        if (it >= 10) {
          float ms;
          CUDA_CHECK(cudaEventElapsedTime(&ms, ea, eb));
          if (ms < best[impl]) best[impl] = ms;
        }
        CUDA_CHECK(cudaEventDestroy(ea));
        CUDA_CHECK(cudaEventDestroy(eb));
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(hy, dy, total * 4, cudaMemcpyDeviceToHost));
      snprintf(path, sizeof(path), "%s/%s_out_%s.bin", dir, shapes[s].name,
               impl == 0 ? "fp32" : "tf32");
      write_bin(path, hy, total);
    }

    double f_tf = mflop / (best[0] * 1e3); /* MFLOP / ms / 1e3 = TFLOP/s */
    double t_tf = mflop / (best[1] * 1e3);
    printf("%-16s %8.1f %3d %10.1f %10.1f %8.2f %8.2f %7.2fx\n", shapes[s].name,
           mflop, Z, best[0] * 1e3, best[1] * 1e3, f_tf, t_tf,
           best[0] / best[1]);
    fflush(stdout);

    free(hx);
    free(hw);
    free(hxh);
    free(hwp);
    free(hy);
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dy);
    if (dp) cudaFree(dp);
  }
  return 0;
}
