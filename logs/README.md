# Raw benchmark logs

Raw measurement logs behind the performance numbers of the paper. All GPU
numbers are from the reference machine: NVIDIA RTX PRO 4000 Blackwell Laptop
GPU (60 SMs, sm_120), nvcc 12.9 (`-O2 -arch=native`), PyTorch 2.13.0+cu130,
cuDNN 9.2, `cudnn.benchmark=True`; batch 1, 224² input, kernel-chain time
excluding H2D/D2H transfers, min and median reported. Sessions of
2026-08-26 … 2026-08-29; cross-session comparisons are labeled as such in the
paper (§5.1 measurement discipline).

## ResNet-18 optimization rounds (§5.2, Tables 1–2)

| Log | Backs |
|---|---|
| `bench_resnet18_fused_final.log` | M10 naive 4.063/4.235 ms, G1 tiled 2.652/2.776 ms, E4 fused 2.026/2.202 ms (one session) |
| `bench_resnet18_tiled_final.log` | G1 per-kernel (KMIN) breakdown |
| `bench_r3_final.log` | E4-r3 (+ cooperative stage merge, float4 staging): 1.582/1.597 ms |
| `bench_igemm_final.log` | R4 implicit GEMM (generated): **0.715/0.855 ms**; +CUDA Graph replay 0.618/0.746 ms; E4 1.572 ms same session |
| `bench_warmed_baseline.log` | warmed-wrapper re-measurement of M10/G1/E4 (methodology rule 3) |
| `validate_igemm_r5_generated.log` | generated implicit-GEMM chain vs. FP64 reference: max abs err 2.587e-9 (paper: 2.59e-9), PASSED |
| `bench_igemm_r5_gen_vs_hw.log` | generated chain 0.713/0.848 ms vs. hand-written pilot 0.721/0.855 ms |

*Gaps, stated openly:* no archived raw log exists for the **E4-r2 row**
(Table 1: 1.600/1.608 ms) — it is an intermediate configuration between
the E4-r1 and E4-r3 logs above. Likewise, the **torch.compile same-session
re-measurement range** quoted in Table 2's footnote (1.133–1.151 min /
1.278–1.348 median) comes from runs whose raw logs were not archived; the
2026-09-18 re-run below (`bench_torch_baseline_rerun_20260918.log`) is a
fresh same-protocol measurement of that row.

## PyTorch baselines (§5.2–5.4)

| Log | Backs |
|---|---|
| `pytorch_res18_r4_rerun.log` | ResNet-18 eager strict-FP32 2.316/2.359 ms; TF32 1.106/1.296 ms (the e2e comparison frame) |
| `bench_torch_baseline.log` | eager vs. torch.compile baselines (vgg16, efficientnet_b0, mobilenet_v2) + ResNet-18 per-layer profiler breakdown (`implicit_convolve_sgemm`) |
| `bench_torch_baseline_rerun_20260918.log` | fresh re-run of `bench_torch_baseline.py` (2026-09-18, same machine): resnet18 eager (TF32 on) 1.095/1.199 ms, compiled 1.007/1.168 ms; vgg16, efficientnet_b0, mobilenet_v2 eager+compiled |

*Provenance note:* `pytorch_res18_r4_rerun.log` was produced by a
ResNet-18-only variant of the `bench_torch_baseline.py` protocol (same
batch 1 / 224² / warmup 20 / 50 reps / CUDA events / fixed seed), run in
session S4 with `torch.backends.cudnn.allow_tf32` toggled off and on;
`bench_torch_baseline.py` itself has no TF32 flag, so the one-off driver
was not archived. `bench_tf32_gap.py` reproduces the same TF32 on/off
end-to-end comparison (its own session measured 1.111 vs. 2.391 ms).

## The TF32 generation gap (§5.4)

| Log | Backs |
|---|---|
| `bench_tf32_kernels.log` | 4096³ matmul microbenchmark: **13.77 TFLOP/s strict FP32 vs. 42.05 TF32** (3.05×); cuDNN kernel-level profile (strict-FP32 `implicit_convolve_sgemm` vs. TF32 `cutlass_tensorop`) |
| `bench_tf32_gap.log` | e2e TF32 on/off: ResNet-18 1.111 vs. 2.391 ms (2.15×), VGG16 2.444 vs. 5.911 ms (2.42×); cuDNN per-shape strict-FP32 numbers of the cuDNN comparison table (l1 36.3 µs/6.37 TF, l2 48.4/4.77, l3 86.5/2.67, l4 161.6/1.43, down-sample 32.4/3.57); per-stage timings |

*Gap, stated openly:* the Ionia column of the cuDNN table (Table 3) quotes
down-sample convs at 22.4–27.6 µs, but the archived per-kernel logs support
21.1–27.6 µs — the provenance of the 22.4 µs lower bound was not saved.

## VGG16 CUDA (§5.4)

| Log | Backs |
|---|---|
| `vgg16p_cuda_validate.log` | naive generated CUDA chain: min 18.256 ms (paper: 18.26) |
| `bench_vgg16p_tiled_final.log` | tiled/fused chain: min 6.357 ms (paper: 6.36; 2.88×) + per-kernel breakdown |

## OpenMP backend — beyond the scope of the DATE paper (extension evidence)

| Log | Backs |
|---|---|
| `validate_resnet18_par_0001.log` | ResNet-18: 8 threads speedup 4.23×; par output bit-identical to seq |
| `vgg16s_validate.log` | VGG16 (32²): 8 threads 5.80× (1/2/4-thread rows too); bit-identical |
| `validate_minitransformer.log` | MiniTransformer: sequential 30.9 µs/inference; OpenMP loses at every thread count (0.62× at 8); par ≡ seq exactly; NumPy err 2.693e-7 |

*Gap, stated openly:* no archived raw log exists for the LeNet-5 2.60×
(4 threads) and AlexNet 5.01× (8 threads) rows, nor for the LeNet-5
16-thread 443× slowdown; they regenerate with
`experiments/ast_codegen/validate_lenet5_par.py` and
`validate_alexnet_par.py` (same protocol as the logs above).

## Triton backend — beyond the scope of the DATE paper (extension evidence)

| Log | Backs |
|---|---|
| `run_triton_alexnet.log` | generated AlexNet chain 39.7 µs min vs. PyTorch eager 80.7 µs (2.0×); err 2.235e-8 |
| `hand_triton_lenet5.log` | LeNet-5: generated 77.3 µs vs. hand-written Triton 70.1 µs (within noise); outputs identical |
| `run_triton_lenet5.log` | generated LeNet-5 chain validation (err 1.49e-8) |
| `run_triton_resnet18.log` | not-yet-fused 68-kernel ResNet-18 chain 2.975 ms min; err 1.024e-8 |

## TileLang feasibility study — beyond the scope of the DATE paper (extension evidence)

See `experiments/ast_codegen/tilelang/README.md` for the full mapping:
`tl_gemm_bench.log` (T.gemm 48.15 TFLOP/s vs. cuBLAS TF32 45.15; silent-TF32
rel err 7.62e-4), `tl_conv_gemm_bench.log` (implicit-GEMM conv 30.3 µs ≈
cuDNN TF32), `tl_conv_bench.log` (direct-tiled FP32 conv 198.3 µs, slowest
of four), `tl_lenet5.log` (hand-written LeNet-5 chain, err 1.49e-8),
`triton_same_shape_bench.log` (same-shape Triton controls).
