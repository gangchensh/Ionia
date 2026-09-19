# Artifact Evaluation Guide — Ionia

This guide reproduces the verified results of the companion paper
*"Ionia 3.0: A Rewriting-Based Synthesis Flow from Typed
Neural-Network Specifications to High-Performance GPU Code"* (DATE
submission) from a clean
checkout. All commands run from the release root. Measured times are from
our reference machine (16-core x86-64, Rocq 9.2.0 + rocq-stdlib 9.2.0 /
OCaml 4.14.1,
RTX PRO 4000 Blackwell Laptop GPU, nvcc 12.9); your numbers will vary with
hardware.

## Step 0 — Environment (≈ 10 min)

- Rocq 9.2.0 with the matching standard library (opam:
  `opam install rocq-core.9.2.0 rocq-stdlib.9.2.0`). Verify:
  `coqc --version` → `The Rocq Prover, version 9.2`.
  Coq 8.18/8.19 also works after `sed -i 's/From Stdlib/From Coq/' src/Ionia/*.v`.
- OCaml ≥ 4.14 with `ocamlc`; Python 3 with NumPy (`pip install numpy`).
- Optional: `gcc` (OpenMP), `nvcc` (CUDA ≥ 12), PyTorch+Triton (Triton
  backend validation only).

## Step 1 — Build the verified core (≈ 25–35 min)

```bash
coq_makefile -f _CoqProject -o Makefile.coq
make -f Makefile.coq -j$(nproc)
```

Expected: no errors; 68 `.vo` files under `src/Ionia/`.
Measured on the reference machine: **26 min 23 s wall (66 min CPU), exit 0**
for the 60-file core; the 8 newer files (`Ops_Transformer`, `Attention_DPIA`,
`MiniTransformer_*`, `Opt_HierPar`, `Opt_ApproxError*`) add roughly
5 minutes.
The single most expensive file is `Bridge_ResNet18.v` (≈ 13 min, sequential);
to skip it, comment it out in `_CoqProject` (the rest builds in ≈ 15 min).
Compilation prints deprecation warnings (`From Coq` → `From Stdlib`) and
extraction-opaque notices — these are benign.

Smoke subset, if you only want the paper's core claims quickly
(verified on the reference machine with Rocq 9.2.0, exit 0):

```bash
make -f Makefile.coq -j$(nproc) \
  src/Ionia/Proof_CodeGen.vo src/Ionia/DPIA_Rules.vo \
  src/Ionia/Parallel_Safety.vo src/Ionia/Opt_HierPar.vo \
  src/Ionia/Opt_BlockFusion.vo src/Ionia/Opt_ApproxError_Apply.vo \
  src/Ionia/LeNet5_DPIA.vo src/Ionia/AlexNet_DPIA.vo \
  src/Ionia/MiniTransformer_AST_Faithful.vo
```

## Step 2 — Axiom audit (≈ 2 min)

```bash
./scripts/print_assumptions.sh
```

Expected output (verified on the reference machine):

```
ClassicalDedekindReals.sig_forall_dec
ClassicalDedekindReals.sig_not_dec
Classical_Prop.classic
FunctionalExtensionality.functional_extensionality_dep
ProofIrrelevance.proof_irrelevance
OK: only standard library axioms are used.
```

These are the standard classical axioms of the Rocq standard library
(real-number decidability schemata, classical logic, proof irrelevance,
functional extensionality). The audit covers the **dependency closure of
the key theorems** listed in `scripts/print_assumptions.v`: none of them
depends on an `Admitted` proof or a custom axiom, and the script exits
non-zero if any appears.

For completeness: `src/Ionia` also contains a few historical `Axiom`
*declarations* left over from the route-B development that are **not in
the dependency closure of any audited theorem** (the audit above is the
evidence — any dependence would show up in its output):

- `src/Ionia/CNN_CodeGen.v` — `CGexpvar_gen_oob` and seven
  `writeNat_*` axioms: they encode a code-generation convention
  (symbolic index arithmetic in the emitted C syntax) for the route-B
  printer experiments, not a semantic fact used by the audited theorems.
- `src/Ionia/CNN_Examples_Base.v` — nine `*_equiv_concrete_*` /
  `Conv4d_p*_k*_s2_equiv` axioms: concrete-shape operator equivalences
  from the route-B examples, superseded by the proved general forms in
  `Ops_*.v` used by the audited theorems.
- `src/Ionia/CNN_Examples_LeNet5.v` — `lenet5_equiv`: the original
  route-B LeNet-5 statement, superseded by the proved
  `LeNet5_DPIA.A_LeNet5`.

## Step 3 — Tutorial demos (requires nvcc; ≈ 5 min)

```bash
cd demo && make all        # or: make demo0 / demo1 / demo2
```

Each demo compiles a small functional expression through the verified
chain, prints a CUDA kernel, runs it on the GPU and checks the result
against NumPy. Expected: `MATCH` / max-abs-diff ≈ 0 per demo
(see `demo/README.md` for exact expected outputs).

## Step 4 — Engineering pipeline: C/OpenMP/CUDA/Triton validation

```bash
cd experiments/ast_codegen
make && make run && make wrapped   # regenerate + gcc-check generated C
                                   # (LeNet-5 / AlexNet / ResNet-18)
make validate                      # CPU C vs NumPy for the three networks
make validate-minitransformer      # MiniTransformer seq + OpenMP C vs NumPy
python3 validate_cuda_vgg16p.py            # CUDA vs NumPy (requires GPU)
python3 validate_cuda_resnet18_tiled.py    # tiled/fused CUDA (requires GPU)
python3 validate_cuda_resnet18_igemm.py    # implicit-GEMM CUDA (requires GPU)
python3 bench_cuda_tiled.py vgg16p_tiled 12        # timing runs (12 reps)
```

The DATE paper evaluates the CUDA backend; the Triton, OpenCL (Ventus)
and OpenMP targets below are **extension capabilities beyond the scope
of the paper** — they are shipped and runnable, but their numbers are
not paper claims.

Triton backend (beyond paper scope; requires a PyTorch+Triton
environment):

```bash
make triton-lenet5     # or triton-alexnet / triton-resnet18
```

OpenCL (Ventus) printer smoke run (beyond paper scope) — writes
`<kname>.cl` + `manifest.txt`:

```bash
make comm_to_opencl && ./comm_to_opencl /tmp/ionia_opencl_out
```

Expected: all validators print a match within `1e-4` abs/rel tolerance.
Reference performance (see README §Performance for the full tables):
VGG16 18.26 → 6.36 ms (**2.88×**); ResNet-18 strict-FP32 kernel chain
4.235 → 0.855 ms median (**0.715 ms min**, implicit GEMM, generated);
generated Triton AlexNet 39.7 µs vs. PyTorch eager 80.7 µs (**2.0×**,
beyond paper scope).

## Step 4b — Evidence archives (not re-runnable here)

Four directories hold **evidence** whose execution dependencies are
external to this artifact; nothing in this step needs to be re-run, and
each directory's README quotes the numbers the evidence backs:

- `logs/` — the raw benchmark logs behind every performance table/figure
  of the paper (§5). `logs/README.md` maps each paper number to its log
  file(s).
- `experiments/ventus/` — generated OpenCL kernel chains, drivers, and
  spike-simulator result logs for the Ventus RISC-V GPGPU backend
  (LeNet-5 13 kernels, AlexNet 20/20, ResNet-18 69/69, all ~1e-9 vs. FP64
  references). **Beyond the scope of the DATE paper**: this backend is
  not covered by the paper and is archived here as an extension
  capability. Re-running requires the vendor Ventus toolchain
  (LLVM/pocl OpenCL + spike), which is not shipped; see
  `experiments/ventus/README.md`.
- `experiments/coralnpu/` — Coral NPU (RISC-V) example programs, compiled
  RV32 ELFs, the simulator driver, and toolchain build logs
  (scalar-FP32, integer-RVV, int8-MAC passing the vendor
  instruction-level simulator). **Beyond the scope of the DATE paper**
  (extension capability). Re-running requires the vendor Coral NPU source
  tree, prebuilt toolchain, and bazel-built simulator; see
  `experiments/coralnpu/README.md`.
- `experiments/ast_codegen/tilelang/` — hand-written TileLang kernels
  (48.15 TFLOP/s GEMM, silent-TF32 evidence). **Beyond the scope of the
  DATE paper** (extension capability). Re-running requires the specific
  GPU (sm_120) plus TileLang 0.1.13 and a pip-assembled CUDA toolkit; see
  the README there.

## Step 4c — Reproducing the paper's performance numbers (GPU required)

All numbers below are min/median kernel-chain times (CUDA events, no
H2D/D2H), batch 1, 224² input, measured on the reference machine (see
`logs/README.md`). Cross-session numbers carry the laptop clock
drift of ±5–8% stated in the paper's measurement discipline (§5.1);
expect your re-runs to land within that band of the quoted values.

**Table 1 (ResNet-18 optimization rounds).** Each round's binary is
built from the shipped ASTs by the matching generator target:

```bash
cd experiments/ast_codegen
make cuda-resnet18          # M10 naive chain (33 launches)
make cuda-resnet18-tiled    # G1 shared-memory tiled + split-c
make cuda-resnet18-fused    # E4 BasicBlock-fused
python3 bench_cuda_resnet18_fused.py 30   # benches all three, one session
```

Expected (one warmed session, 30 processes): M10 ≈ 4.06/4.24 ms, G1 ≈
2.65/2.78 ms, E4 ≈ 2.03/2.20 ms (cf. `logs/bench_resnet18_fused_final.log`).
The R4 row:

```bash
make resnet18_igemm        # generates + compiles the implicit-GEMM chain
python3 bench_cuda_resnet18_igemm.py 30
```

Expected: R4 implicit GEMM ≈ **0.72/0.86 ms**, +CUDA Graph ≈ 0.62/0.75 ms
(cf. `logs/bench_igemm_final.log`); the same run re-measures the E4 binary
for a same-session baseline. The E4-r2 row (1.600/1.608 ms) is an
intermediate configuration whose raw log was not archived — see
`logs/README.md`.

Twin check (no GPU needed): `python3 compare_igemm_twin.py` regenerates
the implicit-GEMM body from the AST and diffs it against the
hand-written pilot `resnet18_igemm_handwritten_ref.cu`. Expected: PASS —
all launch sites argument-identical and all shared kernel bodies
byte-identical (verified 2026-09-18: 16 init + 43 chain launch sites
identical; 60 shared kernels byte-identical; the pilot additionally
carries one never-launched kernel, `ast32_output_fused`; the paper's
"57 call sites" figure predates the final split-K configuration, which
has 59 launch sites). The paper's output bit-identity sub-claim
(md5-stable across runs) requires nvcc+GPU and is not re-checked here —
no archived raw log of those runs survives; what is checkable offline,
the md5 stability of the *generated source* across regeneration, holds
(`resnet18_igemm_cuda_output.cu` regenerates byte-identically).

**Table 2 (end-to-end vs. PyTorch).**

```bash
# run from the artifact root, so the results JSON lands in
# experiments/ast_codegen/:
python3 experiments/ast_codegen/bench_torch_baseline.py
```

The script measures PyTorch eager and
`torch.compile` (default mode) for ResNet-18/VGG16/EfficientNet-B0/
MobileNetV2 with the paper's protocol (20 warmup + 50 timed iterations,
CUDA events); with the default TF32-enabled cuDNN settings it reproduces
the TF32-frame rows (eager TF32 ≈ 1.11/1.30 ms; compile ≈ 1.09/1.28 ms,
cross-session; a fresh 2026-09-18 re-run is archived as
`logs/bench_torch_baseline_rerun_20260918.log`). The strict-FP32 eager
row (2.316/2.359 ms) comes from `logs/pytorch_res18_r4_rerun.log`, a
ResNet-18-only variant of the same protocol with
`torch.backends.cudnn.allow_tf32` toggled off/on in session S4 — the
script has no TF32 flag, so rerun it with that toggle set in a short
driver to reproduce the strict-FP32 row.

**Table 3 (per-shape vs. cuDNN FP32).** The cuDNN column:

```bash
python3 experiments/ast_codegen/bench_tf32_gap.py
```

(part 2 is the single-convolution microbenchmark: strict-FP32 cuDNN
timings for the four representative ResNet-18 shapes, ≈ 36.3/48.4/86.5/
161.6 µs for l1–l4; see `logs/bench_tf32_gap.log`). The Ionia column is
the per-kernel `KMIN` breakdown printed by the instrumented igemm binary
(`python3 bench_cuda_resnet18_igemm.py 30`, cf. `logs/bench_igemm_final.log`;
expect ≈ 30–37 µs for l1–l4 convs).

**TF32 generation gap (§5.4).**

```bash
python3 experiments/ast_codegen/bench_tf32_kernels.py
```

Expected: the 4096³ matmul microbenchmark reports ≈ 13.8 TFLOP/s strict
FP32 vs. ≈ 42 TFLOP/s TF32 (≈ 3.05× tensor-core advantage), and the
cuDNN kernel profile shows strict-FP32 `implicit_convolve_sgemm` vs.
TF32 `cutlass_tensorop` kernels (cf. `logs/bench_tf32_kernels.log`).

## Step 5 — Inspect the paper's claims

Open the files listed in README's claim↔evidence map, e.g.:

- `src/Ionia/DPIA_Rules.v` — all rewriting rules end in `Qed.`
- `src/Ionia/Parallel_Safety.v:415` — `Theorem parallel_safe`
- `src/Ionia/Opt_BlockFusion.v:199` — `Theorem A_BasicBlock_fused`
- `src/Ionia/Opt_HierPar.v:150` — `Theorem AmkvPar3` (3-level OpenCL parallelism)
- `src/Ionia/Opt_ApproxError.v:245` — `Theorem rsum_rnd_error`;
  `:361` — `Corollary conv_dot_error` (TF32 `u = 2^-11` at `:543`)
- `src/Ionia/ResNet18_DPIA.v:864` — `Theorem A_ResNet18`
- `src/Ionia/MiniTransformer_DPIA.v:500` — `Theorem A_MiniTransformer`
- `src/Ionia/MiniTransformer_AST_Faithful.v:1585` —
  `Theorem Bridge_MiniTransformer_faithful`
- `src/Ionia/Bridge_ResNet18.v` — `Theorem Bridge_ResNet18`

## Troubleshooting

- `Cannot find a physical path bound to logical path ... Stdlib` — you are
  on Coq ≤ 8.19; apply the `sed` from Step 0.
- `coqc: command not found` — put the opam switch bin directory on `PATH`
  (e.g. `export PATH="$HOME/.opam/<switch>/bin:$PATH"`).
- The numerical steps (3–4) are optional for the formal claims: the
  theorems audited in Step 2 are independent of the OCaml/Python/CUDA
  engineering pipeline (see README §Trust boundary).
