(* CUDA backend generator for ResNet-18 (224x224), E4 fully-fused variant
   (2026-08-27).

   Derived from the E2 fused-block pilot (fused_block_pilot.cu, 2.10x on the
   l1_b2 block) and the G1 tiled generator (comm_to_cuda_resnet18_tiled.ml).
   Unlike the AST-driven generators, this emitter is structure-specialised:
   the ResNet-18 topology is fixed, so the block table below drives the
   launch sequence directly.  Every one of the 8 BasicBlocks is emitted as a
   single fused kernel (conv-bn-relu-conv-bn[-add]-relu), plus a split-c
   reduction/epilogue kernel for the deep blocks:

     - id blocks, SPLIT=1 (stage 1, 56x56x64): pilot template, T1 =
       relu(bn1(conv1)) tile resident in shared memory, full epilogue
       (bn2 + identity add + relu) in-kernel, single global write;
     - id/proj blocks, SPLIT>1 (stages 2-4): channel-grouped variant, grid
       (tiles, tiles, SPLIT); each block produces CG = Cout/SPLIT channels of
       the T1 halo tile and accumulates conv2 partial sums over them into a
       workspace; a reduce kernel applies bn2 + shortcut (identity or fused
       1x1-Conv+BN projection read from the block input) + relu.

   Deep-block adaptations vs the pilot (measured): phase 1 stages the input halo tile AND the
   w1 slices in chunks of CI channels (one __syncthreads pair per CI input
   channels instead of per channel — 512/256 channel-deep stages are
   otherwise sync-bound); weights must go through shared memory, reading
   them per-thread from global is L1-instruction bound (measured regression).
   TILE is a per-block parameter (8 for stages 1-3, 4 for the 7x7 stage 4)
   to keep per-thread register accumulators at KC2 <= 16 and to recover grid
   parallelism at 7x7 (64 blocks instead of 16).

   Stem (conv7x7+BN+ReLU tiled kernel, ast0) / pool (ast1) / GAP+FC (ast32)
   are reused verbatim from the G1 tiled generated body.

   Correctness basis: route-A theorems A_BasicBlock_fused /
   A_BasicBlock_id_fused (src/Ionia/Opt_BlockFusion.v, proved);
   engineering-side correctness by numerical validation against the M7 NumPy
   reference (validate_cuda_resnet18_fused.py, atol/rtol 1e-5). *)

let blocks = [
  (* prefix   cin  cout   m  stride split proj   tile ci *)
  ("l1_b1",   64,   64,  56, 1,     1,    false, 8,  16);
  ("l1_b2",   64,   64,  56, 1,     1,    false, 8,  16);
  ("l2_b1",   64,  128,  28, 2,     2,    true,  8,  16);
  ("l2_b2",  128,  128,  28, 1,     2,    false, 8,  16);
  ("l3_b1",  128,  256,  14, 2,     8,    true,  8,  16);
  ("l3_b2",  256,  256,  14, 1,     8,    false, 8,  32);
  ("l4_b1",  256,  512,   7, 2,    32,    true,  8,  16);
  ("l4_b2",  512,  512,   7, 1,    32,    false, 8,  32);
]

(* input buffer of each block: stem pool output for l1_b1, previous block
   output otherwise *)
let block_inputs =
  let rec go prev = function
    | [] -> []
    | (p, _, _, _, _, _, _, _, _) :: tl -> prev :: go (p ^ "_add_out") tl
  in
  go "pool_out" blocks

let nth = 1024

(* phase-2 w2 staging width (ic slices per __syncthreads pair).  CW=4
   quarters the phase-2 sync count; stage 4 is shared-memory bound
   (99KB opt-in cap), so l4_b1 keeps CW=1 and l4_b2 takes CW=2. *)
let cw_of = function "l4_b1" -> 2 | "l4_b2" -> 4 | _ -> 4

let shmem_bytes ~cout ~stride ~split ~tile ~ci ~cw =
  let cg = cout / split in
  let th = tile + 2 in
  let ih = (th - 1) * stride + 3 in
  (cg * th * th + ci * ih * ih + max (ci * cg * 9) (cw * cout * 9)) * 4

let cuda_templates = {cuda|
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
|cuda}

(* stem conv7x7+BN+ReLU tiled kernel (ast0), pool (ast1) and GAP+FC (ast32),
   reused verbatim from the G1 tiled generated body *)
let cuda_reused_kernels = {cuda|
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
|cuda}

let buf_add = Buffer.add_string

(* Per-block __global__ wrappers: the __constant__ tensor symbols may only be
   dereferenced in device code, so the templated device functions above are
   called from one small wrapper kernel per block (no host-side symbol
   reads). *)
let emit_wrappers buf (prefix, cin, cout, m, stride, split, proj, tile, ci) inbuf =
  let ws_arg =
    if split = 1 then "(float*)0"
    else Printf.sprintf "&%s_ws[0][0][0][0]" prefix
  in
  buf_add buf (Printf.sprintf
    "__global__ void __launch_bounds__(1024) %s_fused_k()\n{\n\
    \  extern __shared__ float smem[];\n\
    \  fused_block_dev<%d,%d,%d,%d,%d,%d,%d,%d>(\n\
    \      &%s[0][0][0][0],\n\
    \      &%s_w1[0][0][0][0], %s_b1,\n\
    \      %s_bn1_mean, %s_bn1_var, %s_bn1_gamma, %s_bn1_beta,\n\
    \      &%s_w2[0][0][0][0], %s_b2,\n\
    \      %s_bn2_mean, %s_bn2_var, %s_bn2_gamma, %s_bn2_beta,\n\
    \      &%s_add_out[0][0][0][0], %s, smem);\n}\n\n"
    prefix cin cout m stride split tile ci (cw_of prefix)
    inbuf
    prefix prefix
    prefix prefix prefix prefix
    prefix prefix
    prefix prefix prefix prefix
    prefix ws_arg);
  if split > 1 then begin
    let sc_args =
      if proj then
        Printf.sprintf
          "&%s_shortcut_w[0][0][0][0], %s_shortcut_b,\n\
          \      %s_shortcut_mean, %s_shortcut_var, %s_shortcut_gamma, %s_shortcut_beta"
          prefix prefix prefix prefix prefix prefix
      else
        (* dummies, unused when PROJ=0 *)
        Printf.sprintf "%s_b2, %s_b2, %s_b2, %s_b2, %s_b2, %s_b2"
          prefix prefix prefix prefix prefix prefix
    in
    buf_add buf (Printf.sprintf
      "__global__ void %s_reduce_k()\n{\n\
      \  fused_block_reduce_dev<%d,%d,%d,%d,%d>(\n\
      \      &%s_ws[0][0][0][0], &%s[0][0][0][0],\n\
      \      %s_b2,\n\
      \      %s_bn2_mean, %s_bn2_var, %s_bn2_gamma, %s_bn2_beta,\n\
      \      %s,\n\
      \      &%s_add_out[0][0][0][0]);\n}\n\n"
      prefix cin cout m split (if proj then 1 else 0)
      prefix inbuf
      prefix
      prefix prefix prefix prefix
      sc_args
      prefix)
  end

(* ---- round 2 (2026-08-28): cooperative stage kernels ------------------
   One cooperative kernel per stage: pool phase (stage 1 only), then both
   blocks' fused_block_dev phases and split-c reduce phases separated by
   this_grid().sync().  The two blocks of a stage happen to share the same
   grid shape (ts,ts,split), so a single launch covers the whole stage.
   Falls back to the per-block launch sequence when cooperative launch is
   unavailable or the grid is not co-resident. *)

let stages = [ ("l1", [0; 1]); ("l2", [2; 3]); ("l3", [4; 5]); ("l4", [6; 7]) ]

let block_pairs = List.combine blocks block_inputs

let dev_call (prefix, cin, cout, m, stride, split, _proj, tile, ci) inbuf =
  let ws_arg =
    if split = 1 then "(float*)0"
    else Printf.sprintf "&%s_ws[0][0][0][0]" prefix
  in
  Printf.sprintf
    "fused_block_dev<%d,%d,%d,%d,%d,%d,%d,%d>(\n\
    \      &%s[0][0][0][0],\n\
    \      &%s_w1[0][0][0][0], %s_b1,\n\
    \      %s_bn1_mean, %s_bn1_var, %s_bn1_gamma, %s_bn1_beta,\n\
    \      &%s_w2[0][0][0][0], %s_b2,\n\
    \      %s_bn2_mean, %s_bn2_var, %s_bn2_gamma, %s_bn2_beta,\n\
    \      &%s_add_out[0][0][0][0], %s, smem)"
    cin cout m stride split tile ci (cw_of prefix)
    inbuf
    prefix prefix
    prefix prefix prefix prefix
    prefix prefix
    prefix prefix prefix prefix
    prefix ws_arg

let reduce_grid_call (prefix, cin, cout, m, _stride, split, proj, _tile, _ci) inbuf =
  let sc_args =
    if proj then
      Printf.sprintf
        "&%s_shortcut_w[0][0][0][0], %s_shortcut_b,\n\
        \      %s_shortcut_mean, %s_shortcut_var, %s_shortcut_gamma, %s_shortcut_beta"
        prefix prefix prefix prefix prefix prefix
    else
      (* dummies, unused when PROJ=0 *)
      Printf.sprintf "%s_b2, %s_b2, %s_b2, %s_b2, %s_b2, %s_b2"
        prefix prefix prefix prefix prefix prefix
  in
  Printf.sprintf
    "fused_block_reduce_grid<%d,%d,%d,%d,%d>(\n\
    \      &%s_ws[0][0][0][0], &%s[0][0][0][0],\n\
    \      %s_b2,\n\
    \      %s_bn2_mean, %s_bn2_var, %s_bn2_gamma, %s_bn2_beta,\n\
    \      %s,\n\
    \      &%s_add_out[0][0][0][0])"
    cin cout m split (if proj then 1 else 0)
    prefix inbuf
    prefix
    prefix prefix prefix prefix
    sc_args
    prefix

let stage_grid_shm idxs =
  let bl = List.map (List.nth block_pairs) idxs in
  let dims = List.map (fun ((_, _, _, m, _, split, _, tile, _), _) ->
      ((m + tile - 1) / tile, split)) bl in
  (match dims with
   | [a; b] when a = b -> ()
   | _ -> failwith "stage blocks disagree on grid");
  let ts, split = List.hd dims in
  let shm = List.fold_left (fun a ((p, _, cout, _, stride, split, _, tile, ci), _) ->
      max a (shmem_bytes ~cout ~stride ~split ~tile ~ci ~cw:(cw_of p))) 0 bl in
  (ts, split, shm)

let emit_stage_coop buf (stage, idxs) =
  buf_add buf (Printf.sprintf
    "__global__ void __launch_bounds__(1024) %s_stage_coop_k()\n{\n  extern __shared__ float smem[];\n"
    stage);
  if stage = "l1" then
    buf_add buf "  pool_phase_grid();\n  cooperative_groups::this_grid().sync();\n";
  List.iteri
    (fun i idx ->
       let (b, inbuf) = List.nth block_pairs idx in
       let (_, _, _, _, _, split, _, _, _) = b in
       buf_add buf "  ";
       buf_add buf (dev_call b inbuf);
       buf_add buf ";\n";
       if split > 1 then begin
         buf_add buf "  cooperative_groups::this_grid().sync();\n  ";
         buf_add buf (reduce_grid_call b inbuf);
         buf_add buf ";\n"
       end;
       if i = 0 then
         buf_add buf "  cooperative_groups::this_grid().sync();\n")
    idxs;
  if stage = "l4" then begin
    (* GAP + FC fused into the stage-4 cooperative kernel (round 2): the
       standalone ast32 kernel re-read l4_b2_add_out per output block
       (~100MB L2 traffic); here GAP is computed once into the dead
       l4_b2_ws scratch, then FC is a block-per-output reduction. *)
    buf_add buf
      "  cooperative_groups::this_grid().sync();\n\
       \  { /* GAP phase: 512 channel means over 7x7 */\n\
       \    const int gtid = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;\n\
       \    float* avg = &l4_b2_ws[0][0][0][0];\n\
       \    if (gtid < 512) {\n\
       \      float s = 0.0f;\n\
       \      for (int i = 0; i < 7; i++)\n\
       \        for (int j = 0; j < 7; j++)\n\
       \          s += l4_b2_add_out[0][gtid][i][j];\n\
       \      avg[gtid] = s / 49.0f;\n\
       \    }\n\
       \  }\n\
       \  cooperative_groups::this_grid().sync();\n\
       \  { /* FC phase: one warp per output, shuffle reduction */\n\
       \    const int gtid2 = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;\n\
       \    const int gth2 = gridDim.x * gridDim.y * gridDim.z * blockDim.x;\n\
       \    const int wid = gtid2 >> 5, lane = gtid2 & 31;\n\
       \    const float* avg = &l4_b2_ws[0][0][0][0];\n\
       \    for (int o = wid; o < 1000; o += gth2 >> 5) {\n\
       \      float a = 0.0f;\n\
       \      for (int c = lane; c < 512; c += 32)\n\
       \        a += avg[c] * fc_w[o][c];\n\
       \      for (int off = 16; off > 0; off >>= 1)\n\
       \        a += __shfl_down_sync(0xffffffffu, a, off);\n\
       \      if (lane == 0) output[0][o] = a + fc_b[o];\n\
       \    }\n\
       \  }\n"
  end;
  buf_add buf "}\n\n"

(* launch statement for one block (fused kernel + reduce when SPLIT>1) *)
let emit_block_launch buf (prefix, cin, cout, m, stride, split, _proj, tile, ci) =
  let ts = (m + tile - 1) / tile in
  let shm = shmem_bytes ~cout ~stride ~split ~tile ~ci ~cw:(cw_of prefix) in
  buf_add buf (Printf.sprintf
    "  { dim3 g(%d,%d,%d); %s_fused_k<<<g, %d, %d, __st>>>(); }\n"
    ts ts split prefix nth shm);
  if split > 1 then
    buf_add buf (Printf.sprintf
      "  %s_reduce_k<<<%d, 256, 0, __st>>>();\n"
      prefix ((cout * m * m + 255) / 256))

let () =
  let buf = Buffer.create 65536 in
  buf_add buf "#include <cuda_runtime.h>\n#include <cooperative_groups.h>\n#include <math.h>\n#include <stdio.h>\n\n";
  buf_add buf cuda_templates;
  buf_add buf cuda_reused_kernels;
  (* per-block wrapper kernels *)
  List.iter (fun (b, inbuf) -> emit_wrappers buf b inbuf)
    block_pairs;
  (* cooperative stage kernels (round 2) *)
  List.iter (emit_stage_coop buf) stages;
  (* sequential (per-block launch) chain on a stream; also the graph-
     capture body since cooperative launches cannot be stream-captured *)
  buf_add buf "\nstatic void resnet18_fused_seq_s(cudaStream_t __st)\n{\n";
  buf_add buf "  { dim3 g(7,7,16); dim3 b(16,16,1); ast0_relu1_out_fused<<<g, b, 0, __st>>>(); }\n";
  buf_add buf "  ast1_pool_out<<<784, 256, 0, __st>>>();\n";
  List.iter (emit_block_launch buf) blocks;
  buf_add buf "  { dim3 g(1000,1,1); dim3 b(256,1,1); ast32_output_fused<<<g, b, 0, __st>>>(); }\n";
  buf_add buf "}\n";
  (* stream-taking host function: cooperative stage path when co-resident *)
  buf_add buf "\nvoid resnet18_fused_s(cudaStream_t __st)\n{\n";
  buf_add buf "  static int __init = 0;\n  static int __coop = 0;\n";
  buf_add buf "  if (!__init) {\n    __init = 1;\n";
  List.iter
    (fun (prefix, _, cout, _, stride, split, _, tile, ci) ->
       let shm = shmem_bytes ~cout ~stride ~split ~tile ~ci ~cw:(cw_of prefix) in
       buf_add buf (Printf.sprintf
         "    cudaFuncSetAttribute(%s_fused_k, cudaFuncAttributeMaxDynamicSharedMemorySize, %d);\n"
         prefix shm))
    blocks;
  List.iter
    (fun (stage, idxs) ->
       let _, _, shm = stage_grid_shm idxs in
       buf_add buf (Printf.sprintf
         "    cudaFuncSetAttribute(%s_stage_coop_k, cudaFuncAttributeMaxDynamicSharedMemorySize, %d);\n"
         stage shm))
    stages;
  buf_add buf "    {\n      int dev = 0, nsm = 0, c = 0;\n";
  buf_add buf "      cudaGetDevice(&dev);\n";
  buf_add buf "      cudaDeviceGetAttribute(&c, cudaDevAttrCooperativeLaunch, dev);\n";
  buf_add buf "      cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);\n";
  buf_add buf "      if (c) {\n        int per;\n        __coop = 1;\n";
  List.iter
    (fun (stage, idxs) ->
       let ts, split, shm = stage_grid_shm idxs in
       buf_add buf (Printf.sprintf
         "        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, %s_stage_coop_k, %d, %d);\n        __coop = __coop && (per * nsm >= %d);\n"
         stage nth shm (ts * ts * split)))
    stages;
  buf_add buf "      }\n    }\n  }\n";
  buf_add buf "  if (__coop) {\n";
  buf_add buf "    { dim3 g(7,7,16); dim3 b(16,16,1); ast0_relu1_out_fused<<<g, b, 0, __st>>>(); }\n";
  List.iter
    (fun (stage, idxs) ->
       let ts, split, shm = stage_grid_shm idxs in
       buf_add buf (Printf.sprintf
         "    { void* __a[1] = {0}; cudaLaunchCooperativeKernel((const void*)%s_stage_coop_k, dim3(%d,%d,%d), dim3(%d), __a, %d, __st); }\n"
         stage ts ts split nth shm))
    stages;
  buf_add buf "  } else {\n";
  buf_add buf "    resnet18_fused_seq_s(__st);\n";
  buf_add buf "  }\n";
  buf_add buf "}\n";
  buf_add buf "\nvoid resnet18_fused()\n{\n  resnet18_fused_s(0);\n}\n";
  (* instrumented variant *)
  buf_add buf "\nvoid resnet18_fused_kt()\n{\n";
  buf_add buf "  cudaEvent_t __kta, __ktb;\n";
  buf_add buf "  cudaEventCreate(&__kta); cudaEventCreate(&__ktb);\n";
  buf_add buf "  cudaStream_t __st = 0;\n";
  buf_add buf "  resnet18_fused();  /* warmup + produce valid intermediates */\n";
  buf_add buf "  KT_ONE(0, \"stem_conv_bn_relu\", { dim3 g(7,7,16); dim3 b(16,16,1); ast0_relu1_out_fused<<<g, b>>>(); });\n";
  buf_add buf "  KT_ONE(1, \"stem_pool\", ast1_pool_out<<<784, 256>>>());\n";
  let idx = ref 2 in
  List.iter
    (fun ((prefix, _, _, _, _, split, _, _, _) as b) ->
       let lbuf = Buffer.create 256 in
       emit_block_launch lbuf b;
       let code = Buffer.contents lbuf in
       let code = String.sub code 2 (String.length code - 3) in
       (* code holds the fused launch, and the reduce launch when SPLIT>1;
          split them at the newline so each gets its own KT_ONE entry *)
       (match String.index_opt code '\n' with
        | None ->
            buf_add buf (Printf.sprintf "  KT_ONE(%d, \"%s_fused\", %s;);\n"
                           !idx prefix code);
            incr idx
        | Some nl ->
            let fused = String.sub code 0 nl in
            let reduce = String.sub code (nl + 2)
                (String.length code - nl - 2) in
            buf_add buf (Printf.sprintf "  KT_ONE(%d, \"%s_fused\", %s;);\n"
                           !idx prefix fused);
            incr idx;
            assert (split > 1);
            buf_add buf (Printf.sprintf "  KT_ONE(%d, \"%s_reduce\", %s;);\n"
                           !idx prefix reduce);
            incr idx))
    blocks;
  buf_add buf (Printf.sprintf
    "  KT_ONE(%d, \"gap_fc\", { dim3 g(1000,1,1); dim3 b(256,1,1); ast32_output_fused<<<g, b>>>(); });\n" !idx);
  List.iteri
    (fun i (stage, idxs) ->
       let ts, split, shm = stage_grid_shm idxs in
       buf_add buf (Printf.sprintf
         "  KT_ONE(%d, \"%s_stage_coop\", { void* __a[1] = {0}; cudaLaunchCooperativeKernel((const void*)%s_stage_coop_k, dim3(%d,%d,%d), dim3(%d), __a, %d, 0); });\n"
         (!idx + 1 + i) stage stage ts ts split nth shm))
    stages;
  buf_add buf "  cudaEventDestroy(__kta); cudaEventDestroy(__ktb);\n";
  buf_add buf "}\n";
  let oc = open_out "resnet18_fused_cuda_output.cu" in
  Buffer.output_buffer oc buf;
  close_out oc;
  print_endline "ResNet-18 fully-fused CUDA body written to resnet18_fused_cuda_output.cu"
