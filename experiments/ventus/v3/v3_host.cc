// Ionia Ventus V3 whole-network host: launches the full kernel chain in
// manifest order. All kernel sources are compiled into a single cl_program;
// buffers are keyed by parameter name and created on first use: if
// data/<name>.bin exists it is loaded as an input, otherwise the buffer is
// an intermediate (device-resident, not read back). After each kernel, its
// out parameter is read back to data/v3out_<kname>_<pname>.bin for
// per-layer validation.
// Usage: v3_host <kernels_dir> <data_dir>
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <string>
#include <vector>

#define CL_CHECK(err)                                                        \
  if (err != CL_SUCCESS) {                                                   \
    printf("OpenCL Error: %d (line %d)\n", err, __LINE__);                   \
    exit(-1);                                                                \
  }

struct Param {
  bool is_out;
  std::string name;
  int count;
};

struct KSpec {
  std::string kname;
  size_t gws, lws;
  std::vector<Param> params;
};

static char *read_file(const char *path, size_t *size) {
  FILE *fp = fopen(path, "rb");
  if (!fp) { printf("Failed to open %s\n", path); exit(-1); }
  fseek(fp, 0, SEEK_END);
  *size = ftell(fp);
  rewind(fp);
  char *buf = (char *)malloc(*size);
  if (fread(buf, 1, *size, fp) != *size) { printf("Short read on %s\n", path); exit(-1); }
  fclose(fp);
  return buf;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("usage: %s kernels_dir data_dir\n", argv[0]);
    return 1;
  }
  std::string kdir = argv[1], ddir = argv[2];

  // ---- parse manifest ----
  std::vector<KSpec> specs;
  {
    std::string mpath = kdir + "/manifest.txt";
    FILE *fp = fopen(mpath.c_str(), "r");
    if (!fp) { printf("Failed to open %s\n", mpath.c_str()); return 1; }
    char line[4096];
    while (fgets(line, sizeof(line), fp)) {
      KSpec s;
      char tok[256];
      int gws, lws, n;
      if (sscanf(line, "%255s %d %d%n", tok, &gws, &lws, &n) != 3) continue;
      s.kname = tok; s.gws = (size_t)gws; s.lws = (size_t)lws;
      const char *cur = line + n;
      while (true) {
        char io[8]; char name2[256]; int cnt, adv;
        if (sscanf(cur, " %7[^:]:%255[^:]:%d%n", io, name2, &cnt, &adv) != 3) break;
        Param p; p.is_out = (strcmp(io, "out") == 0); p.name = name2; p.count = cnt;
        s.params.push_back(p);
        cur += adv;
      }
      specs.push_back(s);
    }
    fclose(fp);
  }
  printf("v3_host: %zu kernels in chain\n", specs.size());

  cl_platform_id platform_id;
  cl_device_id device_id;
  cl_int err;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
  cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  CL_CHECK(err);
  cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);
  CL_CHECK(err);

  // ---- concat all kernel sources into one program ----
  std::string all_src;
  for (auto &s : specs) {
    std::string p = kdir + "/" + s.kname + ".cl";
    size_t sz;
    char *src = read_file(p.c_str(), &sz);
    all_src.append(src, sz);
    all_src += "\n";
    free(src);
  }
  const char *srcp = all_src.c_str();
  size_t srcsz = all_src.size();
  cl_program program =
      clCreateProgramWithSource(context, 1, &srcp, &srcsz, &err);
  CL_CHECK(err);
  err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
    printf("Build log:\n%s\n", log.data());
    CL_CHECK(err);
  }

  // ---- buffer table ----
  std::map<std::string, std::pair<cl_mem, int> > bufs;
  for (auto &s : specs)
    for (auto &p : s.params) {
      if (bufs.count(p.name)) {
        if (bufs[p.name].second != p.count) {
          printf("buffer size mismatch for %s\n", p.name.c_str());
          return 1;
        }
        continue;
      }
      cl_mem b = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                sizeof(float) * p.count, NULL, &err);
      CL_CHECK(err);
      std::string bin = ddir + "/" + p.name + ".bin";
      if (FILE *fp = fopen(bin.c_str(), "rb")) {
        std::vector<float> data(p.count);
        if (fread(data.data(), sizeof(float), p.count, fp) != (size_t)p.count) {
          printf("Short read on %s\n", bin.c_str()); return 1;
        }
        fclose(fp);
        CL_CHECK(clEnqueueWriteBuffer(queue, b, CL_TRUE, 0,
                                      sizeof(float) * p.count, data.data(),
                                      0, NULL, NULL));
        printf("load %s <- %s (%d floats)\n", p.name.c_str(), bin.c_str(), p.count);
      } else {
        printf("alloc %s (%d floats, intermediate)\n", p.name.c_str(), p.count);
      }
      bufs[p.name] = std::make_pair(b, p.count);
    }

  // ---- launch chain ----
  for (auto &s : specs) {
    cl_kernel kernel = clCreateKernel(program, s.kname.c_str(), &err);
    CL_CHECK(err);
    int argi = 0;
    for (auto &p : s.params) {
      cl_mem b = bufs[p.name].first;
      CL_CHECK(clSetKernelArg(kernel, argi++, sizeof(b), &b));
    }
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &s.gws, &s.lws,
                                    0, NULL, NULL));
    CL_CHECK(clFinish(queue));
    printf("ran %-8s gws=%zu lws=%zu\n", s.kname.c_str(), s.gws, s.lws);
    // dump each out buffer for per-layer validation
    for (auto &p : s.params) {
      if (!p.is_out) continue;
      std::vector<float> data(p.count);
      CL_CHECK(clEnqueueReadBuffer(queue, bufs[p.name].first, CL_TRUE, 0,
                                   sizeof(float) * p.count, data.data(),
                                   0, NULL, NULL));
      std::string outp = ddir + "/v3out_" + s.kname + "_" + p.name + ".bin";
      FILE *fp = fopen(outp.c_str(), "wb");
      if (!fp) { printf("Failed to open %s\n", outp.c_str()); return 1; }
      fwrite(data.data(), sizeof(float), p.count, fp);
      fclose(fp);
    }
    clReleaseKernel(kernel);
  }

  for (auto &kv : bufs) clReleaseMemObject(kv.second.first);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  printf("v3_host: chain done\n");
  return 0;
}
