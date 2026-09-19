# Ionia 3.0: Technical Report for the Verified Code-Generation Artifact

*This report accompanies the DATE-submission paper "Ionia 3.0: A Rewriting-Based Synthesis Flow from Typed
Neural-Network Specifications to High-Performance GPU Code" and its open-source artifact (this repository).
It supplements the paper with the technical detail that did not fit the page budget — proof organization,
per-file structure, validation methodology — and maps every claim to the artifact's reproduction entry points.
Its scope is exactly the paper's scope; where the artifact additionally ships extension capabilities
that the paper does not cover (the OpenMP, Triton, TileLang, and Ventus/Coral NPU backends), they are marked
as such.*

## Contents

1. [Introduction and Scope](#1-introduction-and-scope)
2. [Formal Foundation](#2-formal-foundation)
3. [Verified Optimization Rules and Their Kernel Realizations](#3-verified-optimization-rules-and-their-kernel-realizations)
4. [The Twelve Network Theorems and the MiniTransformer Faithful Bridge](#4-the-twelve-network-theorems-and-the-minitransformer-faithful-bridge)
5. [Backends](#5-backends)
6. [Evaluation and Measurement Discipline](#6-evaluation-and-measurement-discipline)
7. [Trust Boundary and Numerical Validation Methodology](#7-trust-boundary-and-numerical-validation-methodology)
8. [Reproduction Map](#8-reproduction-map)

All file references are relative to the repository root. Theorem names and line numbers were checked against the
sources at release time; if a file has since been edited, grep for the theorem name (e.g.
`grep -n "Theorem A_BasicBlock_fused" src/Ionia/Opt_BlockFusion.v`).

---

## 1. Introduction and Scope

Ionia 3.0 is an end-to-end verified code-generation pipeline: neural networks are written as typed functional
specifications in Rocq/Coq and compiled by DPIA-style rewrite rules — acceptor-passing (A) and
continuation-passing (C) rules — into a race-free imperative intermediate form, which is extracted to OCaml and
printed to CUDA by template-based backends (the artifact additionally ships C/OpenMP, Triton, and OpenCL
backends, which are beyond the paper's scope; §5). The defining property is that **the
rewrite rules are theorems**: the base rule set (A1–A14/C1–C14), the optimization rules (fusion, tiling, parallel
granularity), and twelve network-level end-to-end theorems are all machine-proved, closing over only the
standard classical axioms of the Rocq library (§2.3; a few historical route-B `Axiom` declarations exist in the
tree but are outside every audited dependency closure — see `scripts/print_assumptions.sh`). Verification is paid once at the rule level; every synthesized instance inherits the guarantee, so
design-space exploration over tile sizes, split factors, and parallel granularities enlarges the performance
space without enlarging the trusted base.

The measured payoff reported in the paper: a generated implicit-GEMM CUDA chain runs ResNet-18 (batch 1, 224²,
strict FP32) end to end in **0.855 ms median (0.715 ms min)** — 3.24×/2.76× (min/median) faster than PyTorch
eager forced to strict FP32, and 1.55×/1.52× faster than PyTorch in its default TF32 mode — with generated
convolutions matching or beating cuDNN FP32 by 1.0–5.0× per shape. The same rule set compiles a GPT-style
MiniTransformer whose spec-to-AST bridge is itself a proved theorem; it also drives a C/OpenMP backend
(up to 5.80×, bit-identical output) and a Triton backend — both beyond the DATE paper's scope, documented
here as extension capabilities. This report stays within that scope: §2–§4 cover the formal
layer, §5 the backend engineering, §6 the evaluation, §7 the trust boundary, §8 reproduction.

---

## 2. Formal Foundation

### 2.1 One semantic domain for everything

The functional specification, the rewrite rules, and the optimization rules share a single semantic domain,
`comm := env -> env` over an option-valued store, with scalars in the **exact real numbers** (`Reals`). Because
equality on commands is Leibniz equality over `env -> env`, every rule is a `Qed`'d equation usable directly by
Coq's `rewrite` in any context — rewriting *is* translation *is* proof. The verification core (finite types,
vectors, the `exp`/`comm` ASTs, the semantics, the rewriting infrastructure) lives in `src/Ionia/finType.v`,
`vector.v`, `easy_logic.v`, `DataExp.v`, and `Proof_CodeGen.v`.

The base DPIA rule set (A1–A14/C1–C14 families) is proved in `src/Ionia/DPIA_Rules.v`; e.g. `AmkvSeq`
(`DPIA_Rules.v:133`) lowers a sequential parallel-map combinator, and `AmkvPar` (`DPIA_Rules.v:173`) is the
parallel-map rule from which the OpenCL three-level rule of §5.5 is derived. The same rules serve two lines: the
**verification line** proves that the imperative compilation of a network is observationally equivalent to its
functional specification (the twelve network theorems of §4), and the **optimization line** proves that each
performance-oriented restructuring (fused, tiled, re-granularized) equals the naive program (§3).

### 2.2 Two equality strengths, used deliberately

- **Optimization rules** (fusion, tiling, granularity) are `Qed`'d **Leibniz equalities on commands**, so they
  rewrite under any context and compose freely.
- **The twelve end-to-end network theorems** are **pointwise observational equalities**, quantified over every
  initial environment `rho` and output path `p`, of the shape (here for ResNet-18):

  ```
  A_ResNet18 : in_acc p o /\ fresh o rho ->
               resnet18_comm o rho p = A(o, ResNet18) rho p
  ```

  (`src/Ionia/ResNet18_DPIA.v:864`), where `A(o, e)` applies specification `e` to acceptor `o`. All shape
  parameters are universally quantified.

### 2.3 Axiom audit

`scripts/print_assumptions.sh` runs `Print Assumptions` on the key theorems (core rules, parallel safety, all
optimization rules including the hierarchical-parallelism and error-bound families, all twelve network theorems,
the MiniTransformer faithful bridge, and the four A/B bridges) and exits non-zero if anything unexpected
appears. Expected output: only Rocq's standard classical axioms — `Classical_Prop.classic`,
`ProofIrrelevance.proof_irrelevance`, `FunctionalExtensionality.functional_extensionality_dep`, and the
`ClassicalDedekindReals` choice primitives (`sig_forall_dec`, `sig_not_dec`). There are **no `Admitted` proofs
and zero project-defined axioms** anywhere in `src/Ionia`. Each theorem can also be audited individually, e.g.
via `coqtop -R src/Ionia Ionia` with `Print Assumptions AmkvPar.`

---

## 3. Verified Optimization Rules and Their Kernel Realizations

Each optimization round in the ResNet-18 performance journey (§6.2) is the kernel realization of one proved rule
family; this section pairs the two sides.

### 3.1 Fusion (buffer elimination)

Producer–consumer fusion removes intermediate tensors: a write-to-buffer-then-read-back two-stage command equals
the direct composite.

| Theorem | File | Content |
|---|---|---|
| `consumer_fusion` | `src/Ionia/Opt_Fusion.v:171` | loop-level generalization of read-after-write value substitution: the consumer loop's buffer reads are replaced cell-by-cell by the producer's defining expression |
| `map_fusion` | `src/Ionia/Opt_Fusion.v:234` | the general elementwise producer–consumer buffer-elimination rule |
| `A_Conv4d_ReLU4d` | `src/Ionia/Opt_Fusion.v:316` | conv+relu: ReLU folded into the reduction continuation |
| `A_Conv4d_BN_ReLU4d` | `src/Ionia/Opt_TiledConv.v:362` | conv+bn+relu; key lemma `C_sum_post3` (`Opt_TiledConv.v:224`): three chained reductions share one scalar accumulator |
| `A_BasicBlock_fused` | `src/Ionia/Opt_BlockFusion.v:199` | capstone: the seven-operator chain conv–bn–relu–conv–bn–add–relu of a ResNet BasicBlock equals a single fused kernel touching **no** intermediate feature-map buffer; **no** freshness/disjointness side conditions |
| `A_BasicBlock_id_fused` | `src/Ionia/Opt_BlockFusion.v:227` | corollary for the identity-shortcut block variant |

The semantic content of `A_BasicBlock_fused` is that the phase-1 result is carried as a pure expression — in the
kernel, as shared memory and registers. This is the formal content of "fusion eliminates five intermediate
tensor round-trips per block" (paper §5.2). Kernel realization: the E4 rounds of §6.2, merging each ResNet stage
into one cooperative kernel and cutting the launch count from 17 to 5.

### 3.2 Tiling (split/join)

| Theorem | File | Content |
|---|---|---|
| `seqfor_split` | `src/Ionia/Opt_Tiling.v:129` | re-indexes a loop over `fin (n*m)` into an outer block loop and an inner in-block loop |
| `idx_split` / `idx_join` | `src/Ionia/Opt_Tiling.v:160`, `:169` | the type-changing re-indexing maps |
| `join_split` / `split_join` | `src/Ionia/Opt_Tiling.v:178`, `:193` | the re-indexing maps are mutually inverse |
| `seqfor_tile_app` | `src/Ionia/Opt_Tiling.v:88` | a loop over `fin w` with `w = p + q` splits at `p`; used in the induction on `n` |
| `A_reduce_num_tiled` | `src/Ionia/Opt_Tiling.v:269` | lifts the re-indexing to reductions **with no algebraic side condition** — the iteration order, hence the accumulation order, is exactly preserved |
| `A_Conv4d_tiled` | `src/Ionia/Opt_TiledConv.v:187` | spatially tiled, channel-split convolution |

The proof of `seqfor_split` proceeds by induction on `n`, using `seqfor_tile_app` for the `w = p + q` cut and a
definitional equality (by left-recursive multiplication) to mediate the otherwise non-convertible types
`fin (n*m)` and `fin (m + n*m)`. `A_Conv4d_tiled` justifies both the shared-memory tiled kernels (G1) and the
deterministic split-c/split-K two-stage reductions of the implicit-GEMM backend (§5.2).

### 3.3 Granularity (parallel re-coarsening)

Motivated by measurement — LeNet-5 on 16 OpenMP threads runs **443× slower** than sequential, fork/join overhead
swamping ~150 µs of work — this family proves that thread-count choice is a pure engineering decision:

| Theorem | File | Content |
|---|---|---|
| `non_interfering` | `src/Ionia/Parallel_Safety.v:73` | definition: iterations pairwise commute on environments |
| `seqfor_perm` | `src/Ionia/Parallel_Safety.v:229` | a non-interfering sequential loop is invariant under any injective re-ordering of its indices |
| `schedule_safe` | `src/Ionia/Parallel_Safety.v:246` | mechanized separating-product argument: iterations writing disjoint acceptor slices commute under any complete schedule |
| `parallel_safe` | `src/Ionia/Parallel_Safety.v:415` | safety of `parfor` introduction (`parallel_safe_A` at `:437` for the A-rule form) |
| `parfor_coarsen` | `src/Ionia/Opt_ParGranularity.v:313` | under non-interference, an `n*p`-iteration parallel loop equals `n` parallel blocks of `p` sequential iterations — and likewise for **any** repacking and **any** block order |
| `parallel_safe_coarsen` | `src/Ionia/Opt_ParGranularity.v:350` | the `parfor`-level corollary |

The `non_interfering` premise is itself discharged in Coq for each parallelized network: the per-network
`src/Ionia/*_Par.v` developments prove that every parallelized loop writes disjoint acceptor slices, and close
with an AST-level `eval par = eval seq` theorem (e.g. `lenet5_par_eval_eq`, `src/Ionia/LeNet5_Par.v:230`). Since
the par/seq equivalence holds for all block sizes, parallelization points can be re-chosen without re-proving
anything — the measured lesson "match granularity to loop weight" (§5.1, §5.4) is a theorem, not a heuristic.

---

## 4. The Twelve Network Theorems and the MiniTransformer Faithful Bridge

### 4.1 Twelve end-to-end theorems

One pointwise observational-equality theorem per network (statement shape in §2.2), each closing a
`src/Ionia/*_DPIA.v` development:

| # | Network | Theorem | File |
|---|---|---|---|
| 1 | LeNet-5 | `A_LeNet5` | `src/Ionia/LeNet5_DPIA.v:336` |
| 2 | AlexNet | `A_AlexNet` | `src/Ionia/AlexNet_DPIA.v:456` |
| 3 | VGG16 | `A_VGG16` | `src/Ionia/VGG16_DPIA.v:629` |
| 4 | GoogLeNet | `A_GoogLeNet` | `src/Ionia/GoogLeNet_DPIA.v:874` |
| 5 | ResNet-18 | `A_ResNet18` | `src/Ionia/ResNet18_DPIA.v:864` |
| 6 | ResNet-50 | `A_ResNet50` | `src/Ionia/ResNet50_DPIA.v:1469` |
| 7 | ResNeXt-50 | `A_ResNeXt50` | `src/Ionia/ResNeXt50_DPIA.v:1419` |
| 8 | DenseNet-121 | `A_DenseNet121` | `src/Ionia/DenseNet121_DPIA.v:1774` |
| 9 | MobileNetV1 | `A_MobileNetV1` | `src/Ionia/MobileNetV1_DPIA.v:902` |
| 10 | SqueezeNet | `A_SqueezeNet` | `src/Ionia/SqueezeNet_DPIA.v:553` |
| 11 | EfficientNet-B0 | `A_EfficientNetB0` | `src/Ionia/EfficientNetB0_DPIA.v:2318` |
| 12 | MiniTransformer (GPT-style) | `A_MiniTransformer` | `src/Ionia/MiniTransformer_DPIA.v:500` |

The supporting operator library (`src/Ionia/Ops_*.v`) covers elementwise, reduce, conv, pool/pad, batchnorm,
concat, grouped convolution, and the transformer vocabulary (softmax, layer norm, GELU, matmul in
`Ops_Transformer.v`). Four networks additionally carry A/B-route bridge theorems relating the DPIA development
to the older `CNN_*` development (`src/Ionia/Bridge_{LeNet5,AlexNet,VGG16,ResNet18}.v`); `Bridge_ResNet18.v` is
the single most expensive proof file to compile (~13 min on the reference machine, §8).

### 4.2 MiniTransformer and the faithful bridge

MiniTransformer is a GPT-style decoder-only model — token+positional embedding, two pre-norm blocks with causal
attention and GELU FFNs, final layer norm, output head; instance `s=8, h=2, d=16, d_m=32, d_f=32, v=8` — a
**coverage** case at toy scale that exercises the rule set against the transformer operator vocabulary and
carries no performance claim. Its end-to-end theorem `A_MiniTransformer`, proved in the same style as the
eleven CNNs, is the twelfth member of the verified set. Two facts stand out:

- **No new operator recognizers were needed on the C/OpenMP path.** Layer norm, GELU, the embedding gather,
  split/merge heads, and the causal mask all compose from the existing combinator vocabulary, so the streaming
  printer works essentially unchanged.
- **The spec-to-AST gap is closed by a theorem.** Unlike the CNN chains — whose ASTs come from Coq extraction —
  the MiniTransformer AST was originally hand-mapped. `Bridge_MiniTransformer_faithful`
  (`src/Ionia/MiniTransformer_AST_Faithful.v:1585`; the file contains 68 `Qed` proofs, zero `Admitted`, only
  the standard classical axioms of §2.3) proves a *faithful* AST — each stage the syntactic transcription of the
  corresponding specification-level functional expression — equal to the verified command, by per-combinator
  homomorphism lemmas. Two extraction-forced residuals (baked token indices; the GELU √(2/π) coefficient
  carried as an environment cell) appear as explicit premises.

The generated C compiles under gcc and validates against a NumPy FP64 reference (seed 7) at max abs err
**2.0e-7** (tolerance 1e-5); parallel and sequential outputs are bit-identical. Performance, measured on the
earlier hand-optimized AST (numerically backed at 2.7e-7), is an honest negative and instructive: the sequential
binary runs 30.9 µs per inference, and OpenMP *loses* at every thread count (0.62× at 8) — each inference enters
26 parallel regions of only 8 iterations with ≲1K MACs per iteration, so fork/join dominates. This measures the
`parfor_coarsen` principle from the other side: attention rows are two orders of magnitude lighter than a
convolution channel loop, and parallelism must be matched to loop weight. The CUDA backend does not yet cover
the model; the missing recognizer patterns (3-D batched matmul, per-row softmax, LN/GELU chains, gather, causal
mask) are itemized in the artifact.

---

## 5. Backends

Structurally the backends are conventional — AST pattern matchers feed shape-specialized template emitters, with
buffer management, fusion passes (`fuse_elementwise_chains`, epilogue absorption), and a cost-model fallback
whose mis-prediction costs performance only, never correctness. What is unconventional is that the AST-level
transformations these passes embody correspond to proved theorems (§3), and that generated code is checked
against the hand-tuned twin it replaces (§5.2).

### 5.1 C/OpenMP (extension capability, beyond the scope of the DATE paper)

The extracted AST is printed by streaming OCaml printers (`experiments/ast_codegen/comm_to_c*.ml`). Parallel
versions live in separate per-network Coq files (`src/Ionia/{LeNet5,AlexNet,VGG16,ResNet18}_Par.v`), where
outermost `seqfor`s become `parfor`s backed by a proved `eval par = eval seq` theorem, and print as
`#pragma omp parallel for`.

One bug class deserves mention: temporaries inside parallel bodies were initially mapped to statically pooled
buffers — a cross-thread data race caught when the verification chain was closed end to end; the fix
(per-iteration-private declarations) is now part of every parallel printer. Because no parallelized loop
performs a reduction, the exact floating-point evaluation order is preserved and parallel and sequential outputs
are **bit-identical** — the `parallel_safe`/`parfor_coarsen` guarantee made observable (numbers in §6.3). The
measured optimum is consistently "threads = physical cores": SMT siblings beyond that do not help (AlexNet falls
back to 3.38× at 16 threads), and below the work-per-thread threshold fork/join inverts the sign entirely (the
LeNet-5 443× of §3.3).

### 5.2 CUDA: tiled templates (G1), then implicit GEMM (G2)

**Generation one (G1)** is a shared-memory tiled convolution template: 16×16 thread blocks, 32×32 output tiles,
2×2 register patches per thread, double-buffered halo input tiles with bank-conflict padding, one
`__syncthreads()` per channel step, weights broadcast through shared memory, BN/bias/ReLU/residual-add inlined
in the epilogue, plus deterministic split-c partial sums (fixed slice order) for spatially small layers and a
cost model that routes k=1 shortcuts and tiny layers back to naive kernels. On VGG16 (224²), G1 takes the
generated chain from 18.26 ms naive to 6.36 ms (2.88×, 4.86 TFLOP/s ≈ 17% of the FP32 peak). Correctness
anchors: `A_Conv4d_tiled` (§3.2); epilogue fusion `A_Conv4d_BN_ReLU4d` (§3.1).

**Generation two (G2)** lowers each 3×3/1×1 convolution to **implicit GEMM**, `C[CO, HO·WO] = W · X_col` with
r-major K: BM=BN=64, BK=32, 128 threads and 8×4 register blocking per thread, float4 shared-memory staging (NHWC
+ r-major makes every staging access a float4), deterministic split-K (`gridDim.z`, per-layer factors
Z ∈ {4, 8, 16, 32}) with an ordered reduction kernel whose epilogue carries BN+bias+ReLU+residual. Activations
stay in HWC end to end; weights are permuted on-device once at initialization, outside the timed path.
Convolution bodies reach 6.2–7.6 TFLOP/s (~25% of peak).

The generator (`experiments/ast_codegen/comm_to_cuda_resnet18_igemm.ml`) recognizes eligible convolutions
directly from AST geometry (k ∈ {1,3}, pad = k/2, stride ∈ {1,2}, `C_in mod 32 = 0`), looks up or heuristically
derives split-K factors, inserts layout transpositions at boundaries, and falls back to the G1 tiled template
for non-conforming layers (e.g. the 7×7, `C_in=3` stem) — so arbitrary networks remain correct, with performance
scaling with the eligible-layer share. All 43 kernels are CUDA-Graph capturable. The generated version
reproduces its hand-written pilot exactly: all kernel call sites argument-identical, kernel bodies
byte-identical, outputs bit-identical (md5-stable across runs), performance within clock noise, and max abs err
**2.59e-9** against an FP64 reference. The check is re-runnable without a GPU:
`experiments/ast_codegen/compare_igemm_twin.py` regenerates the body from the AST and diffs it against the
pilot — as of 2026-09-18 it confirms 59 launch sites (43 timed-chain + 16 weight-permutation; the paper's "57"
predates the final split-K configuration), all argument-identical, with all 60 shared kernel bodies
byte-identical (the pilot carries one additional never-launched kernel, `ast32_output_fused`).

### 5.3 Triton (extension capability, beyond the scope of the DATE paper)

An OCaml generator (`experiments/ast_codegen/comm_to_triton.ml`, ~1.5k lines) reuses the CUDA backends' pattern
matchers and emits Triton Python: output elements map to programs (`parfor` → `tl.program_id` flat grid),
reductions unroll at trace time (`tl.sum` / inline unrolling), padding becomes masked loads
(`tl.load(mask=...)`; no pad buffer is materialized), buffers map to torch tensors, and all shapes are baked as
compile-time constants. It handles LeNet-5 (13/13 components recognized), AlexNet (10/10), and ResNet-18
(68/68); fusion and `tl.dot` tiling are not yet ported from the CUDA side — the 2.95 ms ResNet-18 number of §6.4
is a direct measure of what those not-yet-ported fusion rules are worth in this backend.

### 5.4 TileLang: a quantified feasibility study (extension capability, beyond the scope of the DATE paper)

TileLang's explicit tile model corresponds almost one-to-one with the split/join rules of §3.2, and its ceiling
is real: a 20-line hand-written TileLang GEMM reaches **48.15 TFLOP/s** on 4096³, *beating* cuBLAS TF32 (45.15)
on this GPU, and its `T.gemm` implicit-GEMM convolution matches cuDNN TF32 on ResNet-18 shapes. Two obstacles
keep it an optional rather than default target: `T.gemm` **silently** lowers FP32 inputs to TF32 tensor cores
(measured rel. err. 7.6e-4; unlike Triton's `input_precision="ieee"` there is no FP32 switch), which leaves the
verified strict-FP32 semantics; and its toolchain (pip-assembled nvcc, version alignment) is markedly more
fragile than Triton's zero-config install. Scripts live under `experiments/ast_codegen/tilelang/`.

### 5.5 OpenCL/Ventus: a verified path to a RISC-V GPGPU (extension capability, beyond the scope of the DATE paper)

The same AST recognizers feed an OpenCL printer (`experiments/ast_codegen/comm_to_opencl.ml`) targeting Ventus,
an open-source RISC-V GPGPU. The target's grid/workgroup/lane hierarchy is itself backed by new theorems in
`src/Ionia/Opt_HierPar.v` (45 `Qed`):

- `AmkvPar3` (`Opt_HierPar.v:150`): a three-level parallelization rule derived from `AmkvPar` by two
  strip-mining steps, with flat- and block-level schedule-freedom corollaries;
- `lds_reduce_correct` (`Opt_HierPar.v:781`): a workgroup-local-memory tree reduction equals the reference
  sequential reduction skeleton, requiring only associativity and a left unit of the combining operator.

On the vendor spike instruction-level simulator (functional only; **no performance claim**), generated kernel
chains validate end to end against FP64 references: LeNet-5 (13 kernels, err. 7.5e-9) and, at reduced sizes,
AlexNet (20/20 kernels, 1.4e-9) and ResNet-18 (69/69 kernels, 1.5e-9). A Coral NPU (RISC-V) toolchain is
likewise operational, with scalar-FP32, integer-RVV, and int8-MAC programs passing its instruction-level
simulator.

---

## 6. Evaluation and Measurement Discipline

### 6.1 Setup

GPU: NVIDIA RTX PRO 4000 Blackwell Laptop (60 SMs, sm_120, 48 MB L2; nominal FP32 peak 28.34 TFLOP/s); nvcc
12.9, `-O2 -arch=native`. PyTorch 2.13.0+cu130, cuDNN 9.2, `cudnn.benchmark=True` (giving the *baseline* its
best algorithm choice). All numbers: batch 1, 224² input, FP32 weights and activations, CUDA events around the
whole kernel chain *excluding* H2D/D2H transfers; min *and* median reported; the GPU exclusively occupied during
measurement; FLOP counts use 1 MAC = 2 FLOP. The evaluation targets a latency-oriented, edge-style operating
point — batch 1, 224², shapes baked at compile time, inference forward pass only — and claims no generality to
large-batch or datacenter regimes.

### 6.2 ResNet-18: four rounds of verified-rule-driven optimization

Strict FP32, batch 1, 224², no transfers (ms). Sessions: S1 = first-round cold wrapper; S2–S4 = warmed wrapper,
same-session. Cross-session deltas carry the ±5–8% clock drift of §6.6. All rows verified against a NumPy
reference at atol = 1e-5.

| Round | Version | Launches | min | median | Session | Rule family |
|---|---|---|---|---|---|---|
| M10 | naive, per-operator | 33 | 4.063 | 4.235 | S1 | (baseline) |
| G1 | shared-mem tiled + split-c | 33+ | 2.652 | 2.776 | S1 | `A_Conv4d_tiled` |
| E4-r1 | BasicBlock-fused | 17 | 2.026 | 2.202 | S1 | `A_BasicBlock_fused` |
| E4-r2 | + cooperative stage merge | 5 | 1.600 | 1.608 | S2 | fusion + launch merging |
| E4-r3 | + float4 staging | 5 | 1.582 | 1.597 | S3 | (layout engineering) |
| R4 | **implicit GEMM (generated)** | 43 | **0.715** | **0.855** | S4 | `A_Conv4d_tiled` (split-K) |
| R4 | + CUDA Graph replay | 1 | 0.618 | 0.746 | S4 | (deployment mode) |

The dominant engineering facts: block-level fusion wins by keeping phase-1 tiles in shared memory and results in
registers; synchronization count, not bandwidth, is the first-order constraint in deep blocks; and small
128-thread blocks with high occupancy (split-K) beat big blocks.

### 6.3 End-to-end comparison and OpenMP

*The OpenMP rows below are an extension capability beyond the DATE paper's scope.*

End-to-end, session S4 (warmed, same-session; ms). TF32 rows have strictly weaker numerical semantics and serve
as a reference frame only.

| Version | Semantics | min | median |
|---|---|---|---|
| **Ionia implicit GEMM** | strict FP32 | **0.715** | **0.855** |
| Ionia + CUDA Graph | strict FP32 | 0.618 | 0.746 |
| PyTorch eager, TF32 off | strict FP32 | 2.316 | 2.359 |
| PyTorch eager, TF32 on (default) | TF32 conv | 1.106 | 1.296 |
| PyTorch compile (default, Inductor) | TF32 conv | 1.090 | 1.275 |

Speedups of Ionia: **3.24× (min) / 2.76× (median)** vs. strict-FP32 PyTorch; **1.55× / 1.52×** vs. default-TF32
PyTorch; 1.52×/1.49× vs. default-mode `torch.compile` cross-session (1.58–1.61× / 1.49–1.58× against its
same-session re-measurements, 1.133–1.151 min / 1.278–1.348 median). The residual profile: convolution bodies
run at 6.2–7.6 TFLOP/s (~25% of peak), epilogue reductions cost ~0.19 ms, the stem is still a tiled-template
kernel; graph replay hides most launch-boundary cost. The non-graph configuration is the headline; capture —
which fixes the launch sequence — is an optional deployment mode.

OpenMP backend (gcc `-O2 -fopenmp`; parallel output bit-identical to sequential in all rows):

| Network | Threads | Speedup |
|---|---|---|
| LeNet-5 | 4 | 2.60× |
| AlexNet | 8 | 5.01× |
| VGG16 (32²) | 8 | 5.80× |
| ResNet-18 (224²) | 8 | 4.23× |

### 6.4 Kernel-level vs. cuDNN FP32 (and, beyond the paper's scope, Triton and TileLang)

Per-shape 3×3 convolutions (231.2 MFLOP each; min µs / TFLOP s⁻¹), generated implicit-GEMM kernels vs. cuDNN 9.2
with TF32 disabled (`implicit_convolve_sgemm`) — the matched-semantics engineering reference:

| Shape (C_in→C_out @ H²) | cuDNN FP32 | Ionia | ratio |
|---|---|---|---|
| l1: 64→64 @ 56² | 36.3 / 6.37 | 30.5–36.2 / 6.4–7.6 | 1.0–1.2× |
| l2: 128→128 @ 28² | 48.4 / 4.77 | 32.1–35.1 / 6.6–7.2 | 1.4–1.5× |
| l3: 256→256 @ 14² | 86.5 / 2.67 | 35.6–37.0 / 6.2–6.5 | 2.4× |
| l4: 512→512 @ 7² | 161.6 / 1.43 | 32.1–32.6 / 7.1–7.2 | 5.0× |
| down-sample convs | 32.4 / 3.57 | 22.4–27.6 / 4.2–5.2 | 1.2–1.5× |

The generated kernels match cuDNN on the largest spatial shape and beat it increasingly as spatial size shrinks:
cuDNN's FP32 implicit sgemm is also parallelism-starved there, while deterministic split-K fills the 60-SM
machine. The fairness asymmetry is stated openly: the Ionia numbers are in-chain kernel times with fusion
already absorbing BN/ReLU/Add, while the cuDNN column is the bare convolution.

Triton (same harness; beyond the paper's scope): the generated AlexNet chain runs in **39.7 µs** (min) vs. PyTorch eager's 80.7 µs
(**2.0×**); LeNet-5 matches hand-written Triton within noise (77.3 vs. 70.1 µs); the not-yet-fused ResNet-18
chain (68 kernels) reaches **2.95 ms** — already 25% ahead of the naive CUDA chain, but behind the fused/igemm
chains, and 27% behind strict-FP32 PyTorch eager (cross-session). TileLang: `T.gemm` GEMM 48.15 vs. cuBLAS TF32
45.15 TFLOP/s, implicit-GEMM conv at cuDNN TF32 level — but silently TF32 (§5.4); in strict FP32, TileLang's
direct-tiled convolution is the slowest of the four implementations measured.

### 6.5 The TF32 generation gap, quantified

Early comparisons had PyTorch eager 2.2–2.7× ahead of the then-current chains; forcing
`cudnn.allow_tf32=False` revealed a *hardware semantics gap*, not an engineering one. A 4096³ matmul
microbenchmark measures **13.77 TFLOP/s in strict FP32 vs. 42.05 in TF32** on this GPU — a **3.05×**
tensor-core advantage. End to end, TF32 is worth 2.15× to PyTorch on ResNet-18 and 2.42× on VGG16. Under
matched strict FP32 the picture inverts: the pre-igemm fused chain already beat strict-FP32 PyTorch eager by
1.49× end to end, and VGG16 is at parity (0.93×, cross-session). TF32-based numbers are therefore reported only
as a clearly labeled reference frame, and an opt-in TF32 mode is treated as a semantics extension, not a default
(§7).

### 6.6 The five measurement rules

The paper holds itself to five rules, and reports violations openly:

1. **Same-session comparisons.** Laptop GPU clocks drift ±5–8%; every ratio in the paper is computed within one
   session, and cross-session numbers are labeled as such.
2. **Min and median both reported.** Medians carry the drift; mins are stable.
3. **A warmed wrapper.** An early "launch overhead" of 0.32 ms turned out to be first-call initialization inside
   a cold-measurement wrapper; all numbers after that correction use warmup (PyTorch: 20 warmup + 50 timed
   iterations; the generated binaries: warmup + inner-loop iterations, 30 independent processes).
4. **Transfers excluded.** H2D staging of weights (125 small files over a FUSE mount, ~240 ms) is a wrapper
   artifact, never mixed into kernel-chain numbers.
5. **The formal envelope is stated, not implied.** Coq theorems cover the pipeline down to the imperative
   semantics (over ℝ); everything after that is unverified and numerically validated (§7).

---

## 7. Trust Boundary and Numerical Validation Methodology

### 7.1 What is proved, what is not

| Layer | Status |
|---|---|
| Functional spec → imperative semantics (base rules, operator layer, network layer) | **Machine-proved**; the audited dependency closures use only the standard classical axioms (§2, §4; `scripts/print_assumptions.sh`) |
| Optimization rules (fusion / tiling / granularity / hierarchical parallelism) | **Machine-proved** Leibniz equalities (§3, §5.5) |
| Semantics → AST (bridges) | Machine-proved for four networks (`Bridge_*.v`) plus the MiniTransformer faithful bridge (§4.2) |
| `CG_comm` printer core (`src/Ionia/CNN_CodeGen.v:705`) | Machine-proved (comm → C string, used by the demo chain) |
| AST → source string (streaming OCaml printers `comm_to_c*.ml`, `comm_to_cuda*.ml`, `comm_to_triton.ml`, `comm_to_opencl.ml`) | **Unverified** — heuristic pattern matchers; validated numerically |
| Kernel templates, index arithmetic, boundary handling | **Unverified**; validated numerically |
| gcc / nvcc / GPU, the benchmark wrappers | Outside the formal envelope; validated numerically |
| Numerical semantics | Exact ℝ in the proofs; the ℝ/FP gap bounded a priori by the error-bound family and a posteriori by end-to-end validation (below) |

The trusted base therefore comprises the Coq kernel, Coq's extraction to OCaml (unverified), the OCaml printers,
the kernel templates, the C/CUDA compilers, and the benchmark wrappers.

### 7.2 The ℝ/FP gap, from both sides

The formalization proves equalities over the **exact reals**; generated code computes in FP32. The gap is closed
from both directions:

- **A priori**: the mechanized FP error-bound family in `src/Ionia/Opt_ApproxError.v` — `rsum_rnd_error` (:245),
  `dot_rnd_error` (:328), and `conv_dot_error` (:361) — proves worst-case rounding bounds for sums, dot
  products, and convolutions over an abstract rounding model (the standard `|rnd(x) − x| ≤ u·|x|` model),
  instantiated at `u_TF32 = 2⁻¹¹` (`Opt_ApproxError.v:543`; TF32 has a 10-bit mantissa; FP16 shares the same
  `u`). This is the groundwork for a future opt-in TF32/tensor-core mode as an explicit semantics extension.
- **A posteriori**: every generated chain is validated end to end against a NumPy/FP64 reference (tolerance
  ≤1e-5; observed max abs errors ~1e-9–1e-8 — e.g. 2.59e-9 for the implicit-GEMM ResNet-18 chain, 2.0e-7 for
  MiniTransformer, 7.5e-9/1.4e-9/1.5e-9 on the Ventus path).

### 7.3 Validation methodology

- **Determinism by construction.** Order-preserving transforms are bit-identical to the sequential code (all
  OpenMP rows of §6.3; under `-fmad=false` on the CPU side). Split-c/split-K/tree reductions change only the
  association order, in a fixed, deterministic way — outputs are md5-stable across runs, and the residual
  difference against a reference is the accumulation-order rounding difference (~1e-8), not nondeterminism.
- **Cross-validation.** The same AST feeds the sequential C printer, the parallel C printer, and the CUDA
  printers; all are compared against a NumPy reference recomputing the network with the same reduction order and
  the same input distribution (fixed seeds, fixed scale). Three-way agreement (AST→C, AST→CUDA, NumPy) localizes
  any discrepancy to a single printer.
- **Twin checking.** The generated implicit-GEMM chain is diffed against its hand-written pilot: all kernel
  call sites argument-identical, kernel bodies byte-identical (§5.2; re-runnable via
  `experiments/ast_codegen/compare_igemm_twin.py`) — generation adds no deviation from the
  validated hand-tuned artifact.
- **BatchNorm scope.** The networks use the batch=1 inference form of BatchNorm (externally given mean/var),
  consistently in the Coq specification, the generated code, and the reference.

### 7.4 Known limitations

- The implicit-GEMM kernel bodies are emitted by generator templates (the generated version is byte-identical to
  the validated hand-written pilot), but a formal rule for the im2col re-indexing itself is not yet proved — the
  split/join family of §3.2 is expected to cover it; listed as future work.
- The CUDA generator currently covers ResNet-18 end to end (with the G1 fallback keeping arbitrary networks
  correct); extending generator coverage to the remaining verified networks is engineering work, not new proof
  obligations.
- MiniTransformer's CUDA path awaits recognizers for 3-D batched matmul, per-row softmax, LN/GELU chains,
  gather, and the causal mask.

---

## 8. Reproduction Map

The authoritative step-by-step guide is **`ARTIFACT.md`** (environment → full build → axiom audit → demos →
numerical validation), and the claim ↔ evidence table in **`README.md`** maps every paper claim to the file and
theorem that backs it. In brief (all commands from the repository root unless noted):

| Claim | Entry point | Expected |
|---|---|---|
| All rules and 12 network theorems proved | `coq_makefile -f _CoqProject -o Makefile.coq && make -f Makefile.coq -j$(nproc)` | full tree compiles, ~25–35 min on 16 cores (`Bridge_ResNet18.v` alone ~13 min; comment it out for a ~15 min quick check) |
| No project axioms in the audited closures | `./scripts/print_assumptions.sh` | only the five standard classical axioms of §2.3 in every audited dependency closure; non-zero exit otherwise |
| Tutorial chain down to a GPU kernel | `cd demo && make all` | four demos, each printing its correctness theorem; `MATCH` against NumPy |
| ResNet-18 CUDA rounds | `cd experiments/ast_codegen && make cuda-resnet18{,-tiled,-fused,-igemm}` | regenerates and compiles each version |
| Numerical validation | `python3 validate_cuda_resnet18_igemm.py` (and `validate_*.py` generally); `make validate`; `make validate-minitransformer` | max abs err ≈ 2.6e-9 (igemm, vs FP64 reference); all validators pass at ≤1e-5 |
| Performance | `python3 bench_cuda_resnet18_igemm.py`; `bench_torch_baseline.py`; `bench_tf32_gap.py` | min ≈ 0.72 ms (S4-class warmed session); TF32 gap ≈ 3.05× |
| Triton (beyond paper scope) | `make triton-lenet5 triton-alexnet triton-resnet18` + `run_triton_*.py` | AlexNet ≈ 40 µs; ResNet-18 ≈ 2.9 ms |
| OpenCL (Ventus) printer | `make comm_to_opencl && ./comm_to_opencl <outdir>` | `.cl` kernels + manifest |

The numerical steps are optional for the formal claims: the theorems audited by `scripts/print_assumptions.sh`
are independent of the OCaml/Python/CUDA engineering pipeline (§7.1). Timing numbers will vary with hardware;
the measurement discipline of §6.6 (same-session ratios, min+median, warmup, transfers excluded) is what makes
them comparable.
