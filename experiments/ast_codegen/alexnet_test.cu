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

__constant__ float (*input)[3][8][8];
__constant__ float (*conv1_w)[3][3][3];
__constant__ float* conv1_b;
__constant__ float (*conv2_w)[8][3][3];
__constant__ float* conv2_b;
__constant__ float (*fc1_w)[64];
__constant__ float* fc1_b;
__constant__ float (*fc2_w)[16];
__constant__ float* fc2_b;
__constant__ float (*output)[10];
__constant__ float (*conv1_relu)[8][8][8];
__constant__ float (*conv1_pool)[8][4][4];
__constant__ float (*conv2_relu)[16][4][4];
__constant__ float (*conv2_pool)[16][2][2];
__constant__ float (*flat)[64];
__constant__ float (*fc1)[16];
__constant__ float (*fc2)[10];

#include <cuda_runtime.h>
#include <math.h>

__global__ void ast_conv1_relu_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 512;
  if (idx >= N) return;
  int i3 = idx % 8; idx /= 8;
  int i2 = idx % 8; idx /= 8;
  int i1 = idx % 8; idx /= 8;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 3; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 9 || pj < 1 || pj >= 9)
                    ? 0.0f
                    : input[i0][c][pi - 1][pj - 1];
        acc += v * conv1_w[i1][c][k1][k2];
      }
  float result = acc + conv1_b[i1];
  conv1_relu[i0][i1][i2][i3] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_conv1_pool()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 128;
  if (idx >= N) return;
  int i3 = idx % 4; idx /= 4;
  int i2 = idx % 4; idx /= 4;
  int i1 = idx % 8; idx /= 8;
  int i0 = idx;
  float best = conv1_relu[i0][i1][2 * i2][2 * i3];
  for (int k1 = 0; k1 < 2; k1 += 1)
    for (int k2 = 0; k2 < 2; k2 += 1) {
      float v = conv1_relu[i0][i1][2 * i2 + k1][2 * i3 + k2];
      if (v > best) best = v;
    }
  conv1_pool[i0][i1][i2][i3] = best;
}

__global__ void ast_conv2_relu_fused()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 256;
  if (idx >= N) return;
  int i3 = idx % 4; idx /= 4;
  int i2 = idx % 4; idx /= 4;
  int i1 = idx % 16; idx /= 16;
  int i0 = idx;
  float acc = 0.0f;
  for (int c = 0; c < 8; c += 1)
    for (int k1 = 0; k1 < 3; k1 += 1)
      for (int k2 = 0; k2 < 3; k2 += 1) {
        int pi = i2 + k1;
        int pj = i3 + k2;
        float v = (pi < 1 || pi >= 5 || pj < 1 || pj >= 5)
                    ? 0.0f
                    : conv1_pool[i0][c][pi - 1][pj - 1];
        acc += v * conv2_w[i1][c][k1][k2];
      }
  float result = acc + conv2_b[i1];
  conv2_relu[i0][i1][i2][i3] = (result <= 0.0f) ? 0.0f : result;
}

__global__ void ast_conv2_pool()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 64;
  if (idx >= N) return;
  int i3 = idx % 2; idx /= 2;
  int i2 = idx % 2; idx /= 2;
  int i1 = idx % 16; idx /= 16;
  int i0 = idx;
  float best = conv2_relu[i0][i1][2 * i2][2 * i3];
  for (int k1 = 0; k1 < 2; k1 += 1)
    for (int k2 = 0; k2 < 2; k2 += 1) {
      float v = conv2_relu[i0][i1][2 * i2 + k1][2 * i3 + k2];
      if (v > best) best = v;
    }
  conv2_pool[i0][i1][i2][i3] = best;
}

__global__ void ast_flat()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 64;
  if (idx >= N) return;
  int i1 = idx % 64; idx /= 64;
  int i0 = idx;
  int hw = 4;
  int c = i1 / hw;
  int rem = i1 % hw;
  int h0 = rem / 2;
  int w0 = rem % 2;
  flat[i0][i1] = conv2_pool[i0][c][h0][w0];
}

__global__ void ast_fc1()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 16;
  if (idx >= N) return;
  int i1 = idx % 16; idx /= 16;
  int i0 = idx;
  float acc = 0.0f;
  for (int i = 0; i < 64; i += 1)
    acc += flat[i0][i] * fc1_w[i1][i];
  fc1[i0][i1] = acc + fc1_b[i1];
}

__global__ void ast_fc2()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 10;
  if (idx >= N) return;
  int i1 = idx % 10; idx /= 10;
  int i0 = idx;
  float acc = 0.0f;
  for (int i = 0; i < 16; i += 1)
    acc += fc1[i0][i] * fc2_w[i1][i];
  fc2[i0][i1] = acc + fc2_b[i1];
}

__global__ void ast_output()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = 10;
  if (idx >= N) return;
  int i1 = idx % 10; idx /= 10;
  int i0 = idx;
  float sum = 0.0f;
  for (int j = 0; j < 10; j += 1)
    sum += expf(fc2[i0][j]);
  output[i0][i1] = expf(fc2[i0][i1]) / sum;
}

void alexnet_test()
{
  ast_conv1_relu_fused<<<2, 256>>>();
  cudaDeviceSynchronize();
  ast_conv1_pool<<<1, 256>>>();
  cudaDeviceSynchronize();
  ast_conv2_relu_fused<<<1, 256>>>();
  cudaDeviceSynchronize();
  ast_conv2_pool<<<1, 256>>>();
  cudaDeviceSynchronize();
  ast_flat<<<1, 256>>>();
  cudaDeviceSynchronize();
  ast_fc1<<<1, 256>>>();
  cudaDeviceSynchronize();
  ast_fc2<<<1, 256>>>();
  cudaDeviceSynchronize();
  ast_output<<<1, 256>>>();
  cudaDeviceSynchronize();
}


int main(int argc, char** argv)
{
  size_t input_bytes = 768;
  float* d_input = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_input, input_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(input, &d_input, sizeof(d_input)));
  size_t conv1_w_bytes = 864;
  float* d_conv1_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv1_w, conv1_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv1_w, &d_conv1_w, sizeof(d_conv1_w)));
  size_t conv1_b_bytes = 32;
  float* d_conv1_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv1_b, conv1_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv1_b, &d_conv1_b, sizeof(d_conv1_b)));
  size_t conv2_w_bytes = 4608;
  float* d_conv2_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv2_w, conv2_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv2_w, &d_conv2_w, sizeof(d_conv2_w)));
  size_t conv2_b_bytes = 64;
  float* d_conv2_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv2_b, conv2_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv2_b, &d_conv2_b, sizeof(d_conv2_b)));
  size_t fc1_w_bytes = 4096;
  float* d_fc1_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_fc1_w, fc1_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(fc1_w, &d_fc1_w, sizeof(d_fc1_w)));
  size_t fc1_b_bytes = 64;
  float* d_fc1_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_fc1_b, fc1_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(fc1_b, &d_fc1_b, sizeof(d_fc1_b)));
  size_t fc2_w_bytes = 640;
  float* d_fc2_w = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_fc2_w, fc2_w_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(fc2_w, &d_fc2_w, sizeof(d_fc2_w)));
  size_t fc2_b_bytes = 40;
  float* d_fc2_b = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_fc2_b, fc2_b_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(fc2_b, &d_fc2_b, sizeof(d_fc2_b)));
  size_t output_bytes = 40;
  float* d_output = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_output, output_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(output, &d_output, sizeof(d_output)));
  size_t conv1_relu_bytes = 2048;
  float* d_conv1_relu = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv1_relu, conv1_relu_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv1_relu, &d_conv1_relu, sizeof(d_conv1_relu)));
  size_t conv1_pool_bytes = 512;
  float* d_conv1_pool = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv1_pool, conv1_pool_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv1_pool, &d_conv1_pool, sizeof(d_conv1_pool)));
  size_t conv2_relu_bytes = 1024;
  float* d_conv2_relu = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv2_relu, conv2_relu_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv2_relu, &d_conv2_relu, sizeof(d_conv2_relu)));
  size_t conv2_pool_bytes = 256;
  float* d_conv2_pool = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_conv2_pool, conv2_pool_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(conv2_pool, &d_conv2_pool, sizeof(d_conv2_pool)));
  size_t flat_bytes = 256;
  float* d_flat = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_flat, flat_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(flat, &d_flat, sizeof(d_flat)));
  size_t fc1_bytes = 64;
  float* d_fc1 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_fc1, fc1_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(fc1, &d_fc1, sizeof(d_fc1)));
  size_t fc2_bytes = 40;
  float* d_fc2 = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_fc2, fc2_bytes));
  CUDA_CHECK(cudaMemcpyToSymbol(fc2, &d_fc2, sizeof(d_fc2)));

  cudaEvent_t __io_start, __io_stop;
  CUDA_CHECK(cudaEventCreate(&__io_start));
  CUDA_CHECK(cudaEventCreate(&__io_stop));
  CUDA_CHECK(cudaEventRecord(__io_start));
  float* h_input = (float*)malloc(768);
  if (!h_input) { fprintf(stderr, "malloc failed for input\n"); return 1; }
  { FILE* f = fopen("alexnet_test_input.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open alexnet_test_input.bin\n"); return 1; }
    size_t nread = fread(h_input, sizeof(float), 192, f);
    fclose(f);
    if (nread != 192) { fprintf(stderr, "Failed to read alexnet_test_input.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_input, h_input, 768, cudaMemcpyHostToDevice));
  float* h_conv1_w = (float*)malloc(864);
  if (!h_conv1_w) { fprintf(stderr, "malloc failed for conv1_w\n"); return 1; }
  { FILE* f = fopen("alexnet_test_conv1_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open alexnet_test_conv1_w.bin\n"); return 1; }
    size_t nread = fread(h_conv1_w, sizeof(float), 216, f);
    fclose(f);
    if (nread != 216) { fprintf(stderr, "Failed to read alexnet_test_conv1_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv1_w, h_conv1_w, 864, cudaMemcpyHostToDevice));
  float* h_conv1_b = (float*)malloc(32);
  if (!h_conv1_b) { fprintf(stderr, "malloc failed for conv1_b\n"); return 1; }
  { FILE* f = fopen("alexnet_test_conv1_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open alexnet_test_conv1_b.bin\n"); return 1; }
    size_t nread = fread(h_conv1_b, sizeof(float), 8, f);
    fclose(f);
    if (nread != 8) { fprintf(stderr, "Failed to read alexnet_test_conv1_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv1_b, h_conv1_b, 32, cudaMemcpyHostToDevice));
  float* h_conv2_w = (float*)malloc(4608);
  if (!h_conv2_w) { fprintf(stderr, "malloc failed for conv2_w\n"); return 1; }
  { FILE* f = fopen("alexnet_test_conv2_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open alexnet_test_conv2_w.bin\n"); return 1; }
    size_t nread = fread(h_conv2_w, sizeof(float), 1152, f);
    fclose(f);
    if (nread != 1152) { fprintf(stderr, "Failed to read alexnet_test_conv2_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv2_w, h_conv2_w, 4608, cudaMemcpyHostToDevice));
  float* h_conv2_b = (float*)malloc(64);
  if (!h_conv2_b) { fprintf(stderr, "malloc failed for conv2_b\n"); return 1; }
  { FILE* f = fopen("alexnet_test_conv2_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open alexnet_test_conv2_b.bin\n"); return 1; }
    size_t nread = fread(h_conv2_b, sizeof(float), 16, f);
    fclose(f);
    if (nread != 16) { fprintf(stderr, "Failed to read alexnet_test_conv2_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_conv2_b, h_conv2_b, 64, cudaMemcpyHostToDevice));
  float* h_fc1_w = (float*)malloc(4096);
  if (!h_fc1_w) { fprintf(stderr, "malloc failed for fc1_w\n"); return 1; }
  { FILE* f = fopen("alexnet_test_fc1_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open alexnet_test_fc1_w.bin\n"); return 1; }
    size_t nread = fread(h_fc1_w, sizeof(float), 1024, f);
    fclose(f);
    if (nread != 1024) { fprintf(stderr, "Failed to read alexnet_test_fc1_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc1_w, h_fc1_w, 4096, cudaMemcpyHostToDevice));
  float* h_fc1_b = (float*)malloc(64);
  if (!h_fc1_b) { fprintf(stderr, "malloc failed for fc1_b\n"); return 1; }
  { FILE* f = fopen("alexnet_test_fc1_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open alexnet_test_fc1_b.bin\n"); return 1; }
    size_t nread = fread(h_fc1_b, sizeof(float), 16, f);
    fclose(f);
    if (nread != 16) { fprintf(stderr, "Failed to read alexnet_test_fc1_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc1_b, h_fc1_b, 64, cudaMemcpyHostToDevice));
  float* h_fc2_w = (float*)malloc(640);
  if (!h_fc2_w) { fprintf(stderr, "malloc failed for fc2_w\n"); return 1; }
  { FILE* f = fopen("alexnet_test_fc2_w.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open alexnet_test_fc2_w.bin\n"); return 1; }
    size_t nread = fread(h_fc2_w, sizeof(float), 160, f);
    fclose(f);
    if (nread != 160) { fprintf(stderr, "Failed to read alexnet_test_fc2_w.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc2_w, h_fc2_w, 640, cudaMemcpyHostToDevice));
  float* h_fc2_b = (float*)malloc(40);
  if (!h_fc2_b) { fprintf(stderr, "malloc failed for fc2_b\n"); return 1; }
  { FILE* f = fopen("alexnet_test_fc2_b.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open alexnet_test_fc2_b.bin\n"); return 1; }
    size_t nread = fread(h_fc2_b, sizeof(float), 10, f);
    fclose(f);
    if (nread != 10) { fprintf(stderr, "Failed to read alexnet_test_fc2_b.bin\n"); return 1; }
  }
  CUDA_CHECK(cudaMemcpy(d_fc2_b, h_fc2_b, 40, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(__io_stop));
  CUDA_CHECK(cudaEventSynchronize(__io_stop));
  {
    float __io_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&__io_ms, __io_start, __io_stop));
    printf("MEMTIME_H2D alexnet_test %.6f\n", __io_ms);
  }

  float* h_output = (float*)malloc(40);
  if (!h_output) { fprintf(stderr, "malloc failed for output\n"); return 1; }

  cudaEvent_t __timer_start, __timer_stop;
  CUDA_CHECK(cudaEventCreate(&__timer_start));
  CUDA_CHECK(cudaEventCreate(&__timer_stop));
  CUDA_CHECK(cudaEventRecord(__timer_start));
  alexnet_test();
  CUDA_CHECK(cudaEventRecord(__timer_stop));
  CUDA_CHECK(cudaEventSynchronize(__timer_stop));
  {
    float __elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&__elapsed_ms, __timer_start, __timer_stop));
    printf("TIME alexnet_test %.6f\n", __elapsed_ms);
  }
  CUDA_CHECK(cudaEventDestroy(__timer_start));
  CUDA_CHECK(cudaEventDestroy(__timer_stop));

  CUDA_CHECK(cudaEventRecord(__io_start));
  CUDA_CHECK(cudaMemcpy(h_output, d_output, 40, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(__io_stop));
  CUDA_CHECK(cudaEventSynchronize(__io_stop));
  {
    float __d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&__d2h_ms, __io_start, __io_stop));
    printf("MEMTIME_D2H alexnet_test %.6f\n", __d2h_ms);
  }
  CUDA_CHECK(cudaEventDestroy(__io_start));
  CUDA_CHECK(cudaEventDestroy(__io_stop));

  FILE* fout = fopen("alexnet_test_output.bin", "wb");
  if (!fout) { fprintf(stderr, "Cannot open %s\n", "alexnet_test_output.bin"); return 1; }
  fwrite(h_output, sizeof(float), 10, fout);
  fclose(fout);

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_conv1_w));
  CUDA_CHECK(cudaFree(d_conv1_b));
  CUDA_CHECK(cudaFree(d_conv2_w));
  CUDA_CHECK(cudaFree(d_conv2_b));
  CUDA_CHECK(cudaFree(d_fc1_w));
  CUDA_CHECK(cudaFree(d_fc1_b));
  CUDA_CHECK(cudaFree(d_fc2_w));
  CUDA_CHECK(cudaFree(d_fc2_b));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_conv1_relu));
  CUDA_CHECK(cudaFree(d_conv1_pool));
  CUDA_CHECK(cudaFree(d_conv2_relu));
  CUDA_CHECK(cudaFree(d_conv2_pool));
  CUDA_CHECK(cudaFree(d_flat));
  CUDA_CHECK(cudaFree(d_fc1));
  CUDA_CHECK(cudaFree(d_fc2));
  free(h_input);
  free(h_conv1_w);
  free(h_conv1_b);
  free(h_conv2_w);
  free(h_conv2_b);
  free(h_fc1_w);
  free(h_fc1_b);
  free(h_fc2_w);
  free(h_fc2_b);
  free(h_output);
  return 0;
}
