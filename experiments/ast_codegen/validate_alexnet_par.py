#!/usr/bin/env python3
"""Numerical validation + performance comparison of the OpenMP classic AlexNet
(AlexNet_Par.alexnet_par_gen_comm, 224x224 input, 5 conv + 3 fc).

Pipeline:
  1. Random float32 inputs/weights (seed=42, scale=1e-3) are staged as .bin
     files; a vectorized NumPy reference mirrors the Coq operator semantics
     (conv reduction order c,k1,k2; maxpool window max with 0-initialized
     accumulator; softmax without max-subtraction; flatten order (c,i,j)).
  2. Driver C files define all buffers, fread the .bin inputs, call the
     generated function (alexnet_par / alexnet_seq) and time it.
  3. gcc -O2 -std=gnu99 [-fopenmp]; outputs are compared against the NumPy
     reference and against each other; the parallel version is timed with
     1/2/4/8/16 threads.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
np.random.seed(42)
scale = 1e-3

INPUT = (scale * np.random.randn(1, 3, 224, 224)).astype(np.float32)
C1_W = (scale * np.random.randn(64, 3, 11, 11)).astype(np.float32)
C1_B = (scale * np.random.randn(64)).astype(np.float32)
C2_W = (scale * np.random.randn(192, 64, 5, 5)).astype(np.float32)
C2_B = (scale * np.random.randn(192)).astype(np.float32)
C3_W = (scale * np.random.randn(384, 192, 3, 3)).astype(np.float32)
C3_B = (scale * np.random.randn(384)).astype(np.float32)
C4_W = (scale * np.random.randn(256, 384, 3, 3)).astype(np.float32)
C4_B = (scale * np.random.randn(256)).astype(np.float32)
C5_W = (scale * np.random.randn(256, 256, 3, 3)).astype(np.float32)
C5_B = (scale * np.random.randn(256)).astype(np.float32)
F1_W = (scale * np.random.randn(4096, 9216)).astype(np.float32)
F1_B = (scale * np.random.randn(4096)).astype(np.float32)
F2_W = (scale * np.random.randn(4096, 4096)).astype(np.float32)
F2_B = (scale * np.random.randn(4096)).astype(np.float32)
F3_W = (scale * np.random.randn(1000, 4096)).astype(np.float32)
F3_B = (scale * np.random.randn(1000)).astype(np.float32)

TENSORS = {
    "input": INPUT,
    "c1_w": C1_W, "c1_b": C1_B,
    "c2_w": C2_W, "c2_b": C2_B,
    "c3_w": C3_W, "c3_b": C3_B,
    "c4_w": C4_W, "c4_b": C4_B,
    "c5_w": C5_W, "c5_b": C5_B,
    "f1_w": F1_W, "f1_b": F1_B,
    "f2_w": F2_W, "f2_b": F2_B,
    "f3_w": F3_W, "f3_b": F3_B,
}


def relu(x):
    return np.maximum(x, np.float32(0.0))


def conv4d(x, w, b, p, s):
    """Vectorized conv matching the Coq Conv4d semantics (zero padding p,
    stride s, reduction over (c, k1, k2), bias added at the end)."""
    from numpy.lib.stride_tricks import sliding_window_view
    N, Cin, H, W = x.shape
    Cout, _, K1, K2 = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
    win = sliding_window_view(xp, (K1, K2), axis=(2, 3))  # N,C,H',W',K1,K2
    win = win[:, :, ::s, ::s]                              # stride
    out = np.einsum("ncijkl,ockl->noij", win, w).astype(np.float32)
    return out + b.reshape(1, Cout, 1, 1)


def maxpool4d(x, k, s):
    """Window max; the Coq loopacc starts at 0, so mirror max(0, window)."""
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(x, (k, k), axis=(2, 3))[:, :, ::s, ::s]
    return np.maximum(win.max(axis=(4, 5)), np.float32(0.0))


def softmax(x):
    # No max-subtraction: matches the Coq semantics exactly.
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def numpy_reference():
    x = relu(conv4d(INPUT, C1_W, C1_B, 2, 4))
    x = maxpool4d(x, 3, 2)
    x = relu(conv4d(x, C2_W, C2_B, 2, 1))
    x = maxpool4d(x, 3, 2)
    x = relu(conv4d(x, C3_W, C3_B, 1, 1))
    x = relu(conv4d(x, C4_W, C4_B, 1, 1))
    x = relu(conv4d(x, C5_W, C5_B, 1, 1))
    x = maxpool4d(x, 3, 2)
    x = x.reshape(1, -1)                       # flatten order (c, i, j)
    x = relu(x @ F1_W.T + F1_B)
    x = relu(x @ F2_W.T + F2_B)
    x = x @ F3_W.T + F3_B
    return softmax(x).astype(np.float32)


BUFFER_DECLS = """
float input[1][3][224][224];
float c1_w[64][3][11][11]; float c1_b[64];
float c1[1][64][55][55]; float r1[1][64][55][55]; float p1[1][64][27][27];
float c2_w[192][64][5][5]; float c2_b[192];
float c2[1][192][27][27]; float r2[1][192][27][27]; float p2[1][192][13][13];
float c3_w[384][192][3][3]; float c3_b[384];
float c3[1][384][13][13]; float r3[1][384][13][13];
float c4_w[256][384][3][3]; float c4_b[256];
float c4[1][256][13][13]; float r4[1][256][13][13];
float c5_w[256][256][3][3]; float c5_b[256];
float c5[1][256][13][13]; float r5[1][256][13][13]; float p5[1][256][6][6];
float flat[1][9216];
float f1_w[4096][9216]; float f1_b[4096]; float fc1[1][4096]; float fc1r[1][4096];
float f2_w[4096][4096]; float f2_b[4096]; float fc2[1][4096]; float fc2r[1][4096];
float f3_w[1000][4096]; float f3_b[1000]; float fc3[1][1000];
float output[1][1000];
"""

LOADS = "\n".join(
    f'  load("{name}", {name}, {arr.size});' for name, arr in TENSORS.items())


def driver_source(func, wrapped, outbin):
    return f"""
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

{BUFFER_DECLS}

#include "{wrapped}"

static void load(const char* name, void* p, size_t n) {{
  char path[256];
  snprintf(path, sizeof(path), "alexnet_par_%s.bin", name);
  FILE* f = fopen(path, "rb");
  if (!f) {{ perror(path); exit(1); }}
  if (fread(p, sizeof(float), n, f) != n) {{
    fprintf(stderr, "short read: %s\\n", path); exit(1);
  }}
  fclose(f);
}}

static double now(void) {{
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}}

int main(int argc, char** argv) {{
  int reps = argc > 1 ? atoi(argv[1]) : 1;
{LOADS}
  {func}();  /* warmup */
  double best = 1e30;
  for (int t = 0; t < 3; t++) {{
    double t0 = now();
    for (int r = 0; r < reps; r++) {func}();
    double dt = (now() - t0) / reps;
    if (dt < best) best = dt;
  }}
  FILE* f = fopen("{outbin}", "wb");
  fwrite(output, sizeof(float), 1000, f);
  fclose(f);
  printf("%.3f\\n", best * 1e6);  /* microseconds per inference */
  return 0;
}}
"""


def run(cmd, **kw):
    subprocess.run(cmd, cwd=ROOT, check=True, **kw)


def run_timed(binary, reps, threads=None):
    env = dict(os.environ)
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
    r = subprocess.run([f"./{binary}", str(reps)], cwd=ROOT, check=True,
                       capture_output=True, text=True, env=env)
    return float(r.stdout.strip())


def main():
    only_numeric = "--numeric-only" in sys.argv

    # 1. Stage inputs/weights.
    for name, arr in TENSORS.items():
        (ROOT / f"alexnet_par_{name}.bin").write_bytes(arr.tobytes())

    # 2. Wrap the generated bodies (printer must have run already).
    run([sys.executable, "wrap_generated_c.py", "alexnet_par",
         "alexnet_par_streaming_output.c", "alexnet_par_wrapped.c"])
    run([sys.executable, "wrap_generated_c.py", "alexnet_seq",
         "alexnet_seq_streaming_output.c", "alexnet_seq_wrapped.c"])

    # 3. Drivers + compile.
    (ROOT / "test_alexnet_par.c").write_text(
        driver_source("alexnet_par", "alexnet_par_wrapped.c",
                      "alexnet_par_output.bin"))
    (ROOT / "test_alexnet_seq.c").write_text(
        driver_source("alexnet_seq", "alexnet_seq_wrapped.c",
                      "alexnet_seq_output.bin"))
    run(["gcc", "-O2", "-std=gnu99", "-fopenmp",
         "test_alexnet_par.c", "-o", "test_alexnet_par", "-lm"])
    run(["gcc", "-O2", "-std=gnu99",
         "test_alexnet_seq.c", "-o", "test_alexnet_seq", "-lm"])

    # 4. Numeric validation.
    print("== reference (NumPy, vectorized) ==")
    ref = numpy_reference()
    us_seq = run_timed("test_alexnet_seq", 1)
    us_par = run_timed("test_alexnet_par", 1, threads=4)
    out_seq = np.frombuffer((ROOT / "alexnet_seq_output.bin").read_bytes(),
                            dtype=np.float32).reshape(1, 1000)
    out_par = np.frombuffer((ROOT / "alexnet_par_output.bin").read_bytes(),
                            dtype=np.float32).reshape(1, 1000)
    err_seq = float(np.max(np.abs(out_seq - ref)))
    err_par = float(np.max(np.abs(out_par - ref)))
    err_x = float(np.max(np.abs(out_par - out_seq)))
    print("ref[:8]     :", ref[0][:8])
    print("seq[:8]     :", out_seq[0][:8])
    print("par[:8]     :", out_par[0][:8])
    print(f"max |seq - ref| = {err_seq:.3e}")
    print(f"max |par - ref| = {err_par:.3e}")
    print(f"max |par - seq| = {err_x:.3e}")
    ok = (np.allclose(out_par, ref, atol=1e-5, rtol=1e-5)
          and np.allclose(out_seq, ref, atol=1e-5, rtol=1e-5))
    print(f"numeric: {'PASSED' if ok else 'FAILED'} (atol=1e-5, rtol=1e-5)")
    if not ok:
        sys.exit(1)
    if only_numeric:
        return

    # 5. Performance.
    print("== performance (us/inference, best of 3 rounds) ==")
    t_seq = run_timed("test_alexnet_seq", 3)
    print(f"sequential        : {t_seq:12.3f} us  (1.00x)")
    for th in (1, 2, 4, 8, 16):
        t = run_timed("test_alexnet_par", 5, threads=th)
        print(f"OpenMP threads={th:2d} : {t:12.3f} us  ({t_seq / t:.2f}x)")


if __name__ == "__main__":
    main()
