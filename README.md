# Ionia — Verified Neural-Network Code Generation in Rocq/Coq

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Ionia is a framework, formalized in the Rocq Prover (formerly Coq), that
generates high-performance CPU (OpenMP) and GPU (CUDA, Triton, OpenCL)
code for neural
networks from **functional** operator expressions, where every rewriting
step from the functional specification to the imperative `comm` AST is
backed by a machine-checked theorem. This repository is the open-source
artifact accompanying our DATE-submission paper, *"Ionia 3.0: A
Rewriting-Based Synthesis Flow from Typed Neural-Network
Specifications to High-Performance GPU Code"* — the system described
there is **Ionia 3.0** — and its companion technical report
(`docs/TECHNICAL_REPORT.md`).

## Directory layout

```
src/Ionia/            All verified Coq sources (68 files, see _CoqProject)
  finType.v vector.v easy_logic.v DataExp.v Proof_CodeGen.v
                      -- verification core: finite types, vectors, exp/comm AST,
                         semantics and the rewriting-rule infrastructure
  DPIA_Rules.v        -- all rewriting rules (the A/C rule families: the
                         acceptor/continuation lineage (a.1)-(a.15) /
                         (c.1)-(c.13) of Ma & Chen, rebuilt around mkvSeq
                         as A1-A14/C1-C14 in Ionia and extended to
                         A0-A16/C1-C16 in Ionia 2.0; in the current source
                         the rules are named, not numbered), all Qed
  Parallel_Safety.v   -- parallel-safety theorems (parfor introduction/scheduling)
  LeNet5_Utils/Functional/Imperative/DPIA.v
                      -- operator-layer methodology developed on LeNet-5
  Ops_*.v             -- operator library: elementwise, reduce, conv, pool/pad,
                         batchnorm, concat, grouped conv, transformer ops
                         (softmax, layer norm, GELU, matmul: Ops_Transformer.v)
  Opt_*.v             -- optimization rules: tiling, fusion, parallel granularity,
                         tiled conv, basic-block fusion, hierarchical
                         parallelism (Opt_HierPar.v, OpenCL work-group/wavefront
                         semantics, 45 Qed), rounding-error bounds over an
                         abstract rounding model (Opt_ApproxError{,_Apply}.v,
                         instantiated at u = 2^-11 for TF32)
  *_DPIA.v            -- 12 end-to-end networks: 11 CNNs (LeNet-5 ...
                         EfficientNet-B0) plus a GPT-style MiniTransformer
                         (Attention_DPIA.v, MiniTransformer_DPIA.v), each
                         closing with Theorem A_<Net>
  MiniTransformer_AST{,_Faithful}.v
                      -- AST-level faithful-layout compilation of
                         MiniTransformer; Bridge_MiniTransformer_faithful
                         (68 Qed) relates it to A_MiniTransformer
  Bridge_*.v          -- A/B bridges relating route-A (DPIA) networks to the
                         route-B (CNN_*) development, plus the route-B files
                         they depend on (CNN_Defs, CNN_OP, CNN_Model,
                         CNN_CodeGen_Core/Semantics/Combinators,
                         Operators_Sem_Equiv, FusionTactics, CNN_Examples_*)
  *_Par.v             -- parallel (parfor) versions: LeNet-5, AlexNet, VGG16,
                         ResNet-18
  CNN_CodeGen.v       -- CG_comm: the proved printer core (comm -> C string)
experiments/ast_codegen/
                      -- engineering pipeline: OCaml streaming printers
                         (comm_to_c*.ml, comm_to_cuda*.ml, comm_to_triton.ml,
                         comm_to_opencl.ml), Python drivers
                         (validate_*.py, bench_*.py, run_triton_*.py),
                         CUDA test harnesses (*.cu)
experiments/ast_codegen/tilelang/
                      -- hand-written TileLang kernels (extension
                         capability beyond the scope of the DATE paper;
                         see its README.md)
experiments/ventus/   -- Ventus (RISC-V GPGPU) spike-simulator validation:
                         generated OpenCL kernel chains, run scripts, and
                         result logs for LeNet-5 / AlexNet / ResNet-18
                         (extension capability beyond the scope of the
                         DATE paper; see its README.md)
experiments/coralnpu/ -- Coral NPU (RISC-V) toolchain validation: example
                         programs, compiled RV32 ELFs, simulator driver,
                         build logs (extension capability beyond the scope
                         of the DATE paper; see its README.md)
logs/                 -- raw benchmark logs behind the paper's performance
                         numbers (see logs/README.md for the claim->log map)
demo/                 -- tutorial: four runnable examples walking the whole
                         verification chain down to a GPU kernel (see demo/README.md)
scripts/print_assumptions.{v,sh}
                      -- batch axiom audit (Print Assumptions) for key theorems
docs/                 -- detailed technical report (English):
                         docs/TECHNICAL_REPORT.md
_CoqProject           -- build manifest for coq_makefile
ARTIFACT.md           -- artifact-evaluation style reproduction guide
LICENSE               -- MIT
```

## Quick start

### Requirements

- **Rocq 9.2.0 with rocq-stdlib 9.2.0** (reference toolchain; the full
  release compiles cleanly on it). Note for 9.x: the standard library is
  a **separate opam
  package** (`rocq-stdlib`, e.g. `rocq-stdlib.9.2.0` for `rocq-core 9.2.0`)
  and `.vo` files are strictly version-keyed — after any toolchain upgrade
  or downgrade, delete all `*.vo`/`*.glob`/`*.vos`/`*.vok` and rebuild from
  clean. The sources use
  `From Stdlib Require Import ...`. With **Coq 8.18/8.19**, rewrite the
  prefix once and the tree compiles unchanged:
  `sed -i 's/From Stdlib/From Coq/' src/Ionia/*.v`
- OCaml ≥ 4.14 (for the `experiments/ast_codegen` printers) and Python 3
  with NumPy (for the validation scripts).
- Optional, for the numerical backends: `gcc` with OpenMP
  (`-fopenmp`), NVIDIA `nvcc` (tested with CUDA 12.9), and a
  PyTorch+Triton Python environment for `make triton-<net>`.

### Build the verified core

```bash
coq_makefile -f _CoqProject -o Makefile.coq
make -f Makefile.coq -j$(nproc)
```

Expected wall time on a 16-core machine: roughly 25–35 minutes for the full
tree (see ARTIFACT.md for a measured breakdown; `Bridge_ResNet18.v` alone
takes ~13 minutes and can be commented out of `_CoqProject` for a quick
check — the core chain then builds in about 15 minutes).

### Axiom audit (trust base)

```bash
./scripts/print_assumptions.sh
```

This runs `Print Assumptions` on the key theorems (core rules, parallel
safety, all optimization rules including `Opt_HierPar` and the
error-bound family, all 12 network theorems, the MiniTransformer
faithful bridge, and the four A/B bridges)
and reports the axiom base. Expected result: only standard classical
axioms of the Rocq standard library (functional extensionality, proof
irrelevance, classical logic, real-number axioms). The audit covers the
**dependency closure** of those theorems: none of them relies on an
`Admitted` proof or a custom axiom. For completeness, `src/Ionia` also
contains a few historical `Axiom` declarations from the route-B
development (`CNN_CodeGen.v`'s `writeNat_*`/`CGexpvar_gen_oob`,
`CNN_Examples_Base.v`'s concrete-shape `*_equiv_*`,
`CNN_Examples_LeNet5.v`'s `lenet5_equiv`) that are outside every audited
closure — `scripts/print_assumptions.sh` lists them and the audit output
is the evidence.

### Numerical validation (OpenMP / CUDA)

See `ARTIFACT.md` §4 and `demo/README.md`. In short:

- `demo/` — `make all` walks four small examples from a functional
  expression in Coq down to a running CUDA kernel, checking outputs
  against NumPy.
- `experiments/ast_codegen/` — `make && make run && make wrapped`
  regenerates C for LeNet-5/AlexNet and checks it with gcc;
  `python3 validate_<net>.py` compares against a NumPy reference;
  `python3 validate_cuda_<net>.py` does the same for the CUDA backends.

## Paper claims ↔ evidence map

| Claim in the paper | File | Key theorem(s) |
|---|---|---|
| All rewriting rules (the A/C families — (a.1)–(a.15)/(c.1)–(c.13) lineage, A1–A14/C1–C14 in Ionia, A0–A16/C1–C16 in Ionia 2.0; named, not numbered, in the current source) are proved | `src/Ionia/DPIA_Rules.v` | `AmkvSeq`, `AmkvPar`, `Aif1`, `A_sum`, `A_max`, `Clet`, `Cif`, ... (all `Qed`) |
| Parallel safety of `parfor` introduction and scheduling | `src/Ionia/Parallel_Safety.v` | `parallel_safe`, `parallel_safe_A`, `seqfor_perm`, `schedule_safe` |
| Tiling / loop-splitting rule | `src/Ionia/Opt_Tiling.v` | `seqfor_split` |
| Operator fusion (Conv+ReLU, buffer elimination) | `src/Ionia/Opt_Fusion.v` | `A_Conv4d_ReLU4d`, `Conv4d_ReLU4d_buffer_elim` |
| Parallel-granularity coarsening | `src/Ionia/Opt_ParGranularity.v` | `parfor_coarsen`, `parallel_safe_coarsen` |
| Tiled convolution | `src/Ionia/Opt_TiledConv.v` | `A_Conv4d_tiled`, `A_Conv4d_BN_ReLU4d` |
| ResNet basic-block fusion theorem | `src/Ionia/Opt_BlockFusion.v` | `A_BasicBlock_fused` (line 199), `basicblock_id_comm_eq_fused` |
| Hierarchical (3-level) parallelism, OpenCL/Ventus path | `src/Ionia/Opt_HierPar.v` | `AmkvPar3`, `lds_reduce_correct` (45 `Qed`) |
| Rounding-error bounds (abstract rounding model, TF32 `u = 2^-11`) | `src/Ionia/Opt_ApproxError.v`, `Opt_ApproxError_Apply.v` | `rsum_rnd_error`, `dot_rnd_error`, `conv_dot_error` |
| End-to-end correctness of 12 networks | `src/Ionia/*_DPIA.v` | `A_LeNet5`, `A_AlexNet`, `A_VGG16`, `A_ResNet18`, `A_ResNet50`, `A_GoogLeNet`, `A_MobileNetV1`, `A_DenseNet121`, `A_SqueezeNet`, `A_ResNeXt50`, `A_EfficientNetB0`, `A_MiniTransformer` |
| MiniTransformer faithful-layout compilation | `src/Ionia/MiniTransformer_AST_Faithful.v` | `Bridge_MiniTransformer_faithful` (68 `Qed`) |
| A/B route equivalence (bridges) | `src/Ionia/Bridge_*.v` | `Bridge_LeNet5`, `Bridge_AlexNet`, `Bridge_VGG16`, `Bridge_ResNet18` |
| Parallel network versions | `src/Ionia/*_Par.v` | e.g. `lenet5_par_eval`, per-layer `*_par_eval` lemmas |
| Printer backend (`CG_comm`) | `src/Ionia/CNN_CodeGen.v` | `CG_comm` (proved printing core) |
| Triton / OpenCL (Ventus) printers — extension capability, not part of the DATE paper | `experiments/ast_codegen/comm_to_triton.ml`, `comm_to_opencl.ml` | `make triton-<net>`, `make comm_to_opencl` |
| OpenMP / CUDA numerical validation | `experiments/ast_codegen/` | `validate_*.py`, `validate_cuda_*.py` (see ARTIFACT.md) |
| TileLang kernels — extension capability, not part of the DATE paper: `T.gemm` GEMM 48.15 TFLOP/s > cuBLAS TF32 45.15; implicit-GEMM conv ≈ cuDNN TF32; silent TF32 lowering (rel. err. 7.6e-4); direct-tiled conv slowest in strict FP32 | `experiments/ast_codegen/tilelang/` | `tl_gemm_bench.py`, `tl_conv_gemm_bench.py`, `tl_conv_bench.py`; logs `logs/tl_*.log` (see tilelang/README.md) |
| Ventus spike-simulator validation — extension capability, not part of the DATE paper: LeNet-5 13 kernels err 7.5e-9, AlexNet 20/20 err 1.4e-9, ResNet-18 69/69 err 1.5e-9 | `experiments/ventus/` | `v3/`, `v4/` generated OpenCL chains + drivers; `experiments/ventus/logs/v3_spike_20260831.log`, `v4_spike_{alexnet,resnet18}_20260831.log` |
| Coral NPU toolchain operational — extension capability, not part of the DATE paper: scalar-FP32 / integer-RVV / int8-MAC pass the instruction-level simulator | `experiments/coralnpu/` | `examples/`, `build/*.elf`, `run_sim.py`, `logs/` (see its README.md) |
| Raw benchmark logs behind every performance number (§5) | `logs/` | `logs/README.md` maps each table/figure to its log(s) |

## Performance summary

Measured on an NVIDIA RTX PRO 4000 Blackwell Laptop GPU (60 SM, nominal
28.34 TFLOP/s FP32), nvcc 12.9, `-O2 -arch=native`, batch 1, 224²,
kernel-chain time without host transfers;
FLOP counted as 1 MAC = 2 FLOP. See `docs/TECHNICAL_REPORT.md` for the full methodology.

ResNet-18 strict-FP32 kernel chain across optimization rounds (each round
is the kernel realization of a proved rule family; min / median, ms):

| Round | Version | min | median |
|---|---|---|---|
| M10 | naive, per-operator | 4.063 | 4.235 |
| G1 | shared-mem tiled + split-c (`A_Conv4d_tiled`) | 2.652 | 2.776 |
| E4 | BasicBlock-fused (`A_BasicBlock_fused`) | 2.026 | 2.202 |
| E4-r3 | + cooperative stage merge, float4 staging | 1.582 | 1.597 |
| R4 | **implicit GEMM (generated)** | **0.715** | **0.855** |
| R4 | + CUDA Graph replay | 0.618 | 0.746 |

End-to-end, same warmed session: Ionia implicit GEMM (strict FP32)
**0.715 min / 0.855 med. ms** vs. PyTorch eager strict-FP32 2.316/2.359 ms
(**3.24×** / 2.76×) and vs. default-TF32 PyTorch 1.106/1.296 ms
(**1.55×** / 1.52×). The generated implicit-GEMM chain matches its
hand-written pilot at max abs err. 2.59e-9 vs. an FP64 reference.

Triton backend (same harness; **not part of the DATE paper** — an
extension capability): generated AlexNet chain 39.7 µs vs.
PyTorch eager 80.7 µs (**2.0×**); LeNet-5 matches hand-written Triton
within noise (77.3 vs. 70.1 µs); the not-yet-fused ResNet-18 Triton
chain (68 kernels) reaches 2.95 ms.

VGG16 (224²): naive generated CUDA 18.26 ms → tiled/fused 6.36 ms
(**2.88×**, 4.86 TFLOP/s effective).

The OpenMP (`parfor`) backends compile to `#pragma omp parallel for`
loops whose correctness is backed by `parallel_safe` /
`parallel_safe_coarsen`; per-network numbers are reproduced by
`experiments/ast_codegen/bench_*.py`.

## Trust boundary

- **Proved (trusted only on the Rocq kernel):** everything in
  `src/Ionia` — the functional↔imperative equations, the optimization
  rules, the end-to-end network theorems, and the `CG_comm` printer core.
- **Not proved (engineering, validated numerically):** the OCaml
  extraction-time patching (`build_network_sym.py`,
  `transform_to_symbolic.py`), the streaming printers
  (`comm_to_c*.ml`, `comm_to_cuda*.ml`, `comm_to_triton.ml`,
  `comm_to_opencl.ml`), and the kernel templates in the
  CUDA backends. These are heuristic and are validated by
  `validate_*.py` against NumPy references on random inputs.
- The formalization uses real numbers (`Reals`) as the scalar domain;
  floating-point rounding of the generated code is outside the formal
  model — the `Opt_ApproxError` theorems bound this gap a priori over an
  abstract rounding model (instantiated at `u = 2^-11` for TF32), and the
  numerical validation above covers it a posteriori.

## Documentation

- `ARTIFACT.md` — step-by-step reproduction guide (build, axiom audit,
  numerical validation).
- `docs/TECHNICAL_REPORT.md` — the full technical report (proof
  organization, per-file structure, evaluation methodology).
- `demo/README.md` — tutorial: four runnable examples walking the whole
  verification chain down to a GPU kernel.
- `logs/README.md` — maps every performance number in the paper to its
  raw benchmark log(s).

## Citation

If you use Ionia, please cite our papers (BibTeX placeholders — please
replace with the final publication data; see also `CITATION.cff`):

```bibtex
@inproceedings{ionia2026date,
  author    = {Anonymous},
  title     = {Ionia 3.0: A Rewriting-Based Synthesis Flow from Typed
               Neural-Network Specifications to High-Performance GPU Code},
  booktitle = {Proc. Design, Automation and Test in Europe (DATE)},
  year      = {2027},
  note      = {Artifact: this repository}
}

@misc{ionia2026artifact,
  author       = {Anonymous},
  title        = {{Ionia} 3.0 artifact: Coq development, code generators, and
                  benchmark logs},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {MIT license}
}
```

## License

MIT — see `LICENSE`.
