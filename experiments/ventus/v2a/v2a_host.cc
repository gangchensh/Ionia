// Ionia Ventus V2a generic host: compiles a kernel from .cl source, reads
// binary float inputs per the manifest, runs it on the spike simulator,
// and writes the result binary.
// Usage: v2a_host <kernel.cl> <kname> <gws> <lws> <out.bin> <out_count>
//                <in1.bin> <in1_count> [in2.bin in2_count ...]
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define CL_CHECK(err)                                                        \
  if (err != CL_SUCCESS) {                                                   \
    printf("OpenCL Error: %d (line %d)\n", err, __LINE__);                   \
    exit(-1);                                                                \
  }

static float *read_bin(const char *path, int n) {
  FILE *fp = fopen(path, "rb");
  if (!fp) { printf("Failed to open %s\n", path); exit(-1); }
  float *buf = (float *)malloc(sizeof(float) * n);
  if (fread(buf, sizeof(float), n, fp) != (size_t)n) {
    printf("Short read on %s\n", path); exit(-1);
  }
  fclose(fp);
  return buf;
}

static char *read_file(const char *path, size_t *size) {
  FILE *fp = fopen(path, "rb");
  if (!fp) { printf("Failed to open %s\n", path); exit(-1); }
  fseek(fp, 0, SEEK_END);
  *size = ftell(fp);
  rewind(fp);
  char *buf = (char *)malloc(*size);
  fread(buf, 1, *size, fp);
  fclose(fp);
  return buf;
}

int main(int argc, char **argv) {
  if (argc < 7) {
    printf("usage: %s kernel.cl kname gws lws out.bin out_count "
           "[in.bin in_count ...]\n", argv[0]);
    return 1;
  }
  const char *cl_path = argv[1];
  const char *kname = argv[2];
  size_t gws = (size_t)atol(argv[3]);
  size_t lws = (size_t)atol(argv[4]);
  const char *out_path = argv[5];
  int out_count = atoi(argv[6]);
  int nin = (argc - 7) / 2;

  cl_platform_id platform_id;
  cl_device_id device_id;
  cl_int err;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  CL_CHECK(err);
  cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);
  CL_CHECK(err);

  size_t src_size;
  char *src = read_file(cl_path, &src_size);
  cl_program program =
      clCreateProgramWithSource(context, 1, (const char **)&src, &src_size, &err);
  CL_CHECK(err);
  free(src);
  err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
    printf("Build log:\n%s\n", log.data());
    CL_CHECK(err);
  }
  cl_kernel kernel = clCreateKernel(program, kname, &err);
  CL_CHECK(err);

  // Argument order: kernel signature = all inputs (manifest order), then the output
  std::vector<cl_mem> bufs;
  int argi = 0;
  for (int k = 0; k < nin; ++k) {
    const char *path = argv[7 + 2 * k];
    int count = atoi(argv[8 + 2 * k]);
    float *data = read_bin(path, count);
    cl_mem b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count,
                              NULL, &err);
    CL_CHECK(err);
    CL_CHECK(clEnqueueWriteBuffer(queue, b, CL_TRUE, 0, sizeof(float) * count,
                                  data, 0, NULL, NULL));
    CL_CHECK(clSetKernelArg(kernel, argi++, sizeof(b), &b));
    bufs.push_back(b);
    free(data);
  }
  cl_mem ob = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                             sizeof(float) * out_count, NULL, &err);
  CL_CHECK(err);
  CL_CHECK(clSetKernelArg(kernel, argi++, sizeof(ob), &ob));

  CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gws, &lws, 0, NULL, NULL));
  CL_CHECK(clFinish(queue));

  std::vector<float> out(out_count);
  CL_CHECK(clEnqueueReadBuffer(queue, ob, CL_TRUE, 0, sizeof(float) * out_count,
                               out.data(), 0, NULL, NULL));
  FILE *fp = fopen(out_path, "wb");
  if (!fp) { printf("Failed to open %s\n", out_path); exit(-1); }
  fwrite(out.data(), sizeof(float), out_count, fp);
  fclose(fp);

  for (cl_mem b : bufs) clReleaseMemObject(b);
  clReleaseMemObject(ob);
  clReleaseKernel(kernel); clReleaseProgram(program);
  clReleaseCommandQueue(queue); clReleaseContext(context);
  printf("Done. %s gws=%zu lws=%zu -> %s (%d floats)\n", kname, gws, lws,
         out_path, out_count);
  return 0;
}
