# Ionia tutorial: from functional neural-network expressions to GPU code

This tutorial walks through the entire Ionia verification chain with four
small runnable examples: write a **functional** operator expression in
Rocq/Coq, translate it via **rewriting rules** (each one a machine-checked
theorem) into an imperative `comm` AST, extract it to OCaml, print it as a
**CUDA kernel** with the backend printer, run it on a real GPU, and compare
the output element-wise against a NumPy reference.

All numbers below are real measured results from our reference machine
(2026-08-27, RTX PRO 4000 Blackwell Laptop GPU, nvcc 12.9.86); the random
seeds are fixed in the scripts, so the runs are reproducible.

## 1. What Ionia is: the verification chain at a glance

Ionia is a Coq/Rocq framework for neural-network code generation. The core
idea: **the equation between an operator's functional definition and its
imperative implementation is a Coq theorem**
(`A(output, f x) = f_comm x output`, Leibniz/observational equivalence), so
every layer of `comm` the generator emits is backed by the corresponding
operator equation.

```text
 functional expression (Coq, exp)              e.g. ReLU4d input, Conv4d p s bias w x
        │  rule rewriting: AmkvSeq, A_sum, Aif1, A_mul ... (src/Ionia/DPIA_Rules.v, all Qed)
        ▼
 imperative AST (Coq, comm)                    e.g. ReLU4d_comm, Conv4d_p1_k3_comm
        │  operator equations: A_ReLU4d, A_Conv4d, Linear2d_sem ... (machine-checked)
        ▼
 Rocq Extraction → OCaml AST             demo/coq/Demo_Comms.v → demo_comm.ml
        │  transform_to_symbolic.py (symbolic nat, keeps loop-variable names)
        ▼
 OCaml backend: pattern matchers + kernel templates   demo/ocaml/demo_to_cuda.ml
        │  (heuristic, unproved — see §6 trust boundary)
        ▼
 CUDA C (__global__ kernel + host launch)  demo/build/*_cuda_output.cu
        │  wrap_generated_cuda.py (memory/IO wrapper, unverified)
        ▼
 nvcc → GPU executable → element-wise comparison against NumPy
```

## 2. Environment setup

| Tool | Version | Notes |
|---|---|---|
| Rocq (coqc) | 9.2.0 (+ rocq-stdlib 9.2.0) | compiles `src/Ionia` and the demo `.v`; put the opam switch's bin directory on `PATH` (see `../ARTIFACT.md` Step 0). A system Coq 8.18 **cannot** be used. |
| ocamlc | ≥ 4.14 | builds the OCaml backend printers |
| nvcc | CUDA ≥ 12 (tested: 12.9.86) | resolved via `$NVCC`, then `PATH`, then `/tmp/cuda-12.9/bin/nvcc` (see `util.py`) |
| python3 + numpy | tested: numpy 1.26.4 | data generation and numerical comparison |

`src/Ionia` must have been built with Rocq 9.2.0 first (see
`../ARTIFACT.md` Step 1). Demo 3 reuses the ready-made pipeline in
`experiments/ast_codegen/` and needs no extra preparation.

## 3. Quick start

```bash
cd demo
make demo0    # demo 0: dot product
make demo1    # demo 1: ReLU
make demo2    # demo 2: small convolution
make demo3    # demo 3: end-to-end AlexNet (reuses experiments/ast_codegen)
make all      # all four
make clean    # remove all generated artifacts (build/, extracted .ml, .cmo, ...)
```

A single `make demo0` runs the whole chain: coqc compiles
`coq/Demo_Comms.v` and extracts → `transform_to_symbolic.py` symbolizes →
ocamlc builds the printer → CUDA is generated → nvcc compiles → the kernel
runs on the GPU → NumPy comparison. All generated artifacts land in
`demo/build/` (excluded via `.gitignore`).

## 4. Demo 0: dot product `dot x y = sum (mkvSeq (fun i => mul (x|[i]) (y|[i])))`

The dot product is the smallest example that exercises the whole chain. In
the functional style, `mkvSeq f` builds an array by index and `sum` is a
reduction; its translation to imperative code uses three rules
(`src/Ionia/DPIA_Rules.v`, all Qed):

```text
A(output, dot x y)
  = A(output, sum (mkvSeq (fun i => mul x|[i] y|[i])))   -- unfold dot
  = suml (mkvSeq ...) (fun r => A(output, r)) output     -- A_sum (DPIA_Rules.v:466)
  … mkvSeq is unrolled by AmkvSeq (:133) into an element-wise loop,
    mul is handled by A_mul (:113)
```

This demo instantiates the dot product as `Linear2d` (batch=1, m=8,
outnum=1) — the Linear2d cell is exactly "dot product + bias":
`add (sum (mkvSeq (fun i => mul x|[i] w|[i]))) b`. The Coq-side definition
is in `coq/Demo_Comms.v` (`demo_dot_comm`), instantiated from
`CNN_Examples_Base_Ops.Linear2d_comm`.

The generated kernel (`build/dot_demo_cuda_output.cu`, a real generated
artifact):

```cuda
__global__ void demo_output()
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  ...
  float acc = 0.0f;
  for (int i = 0; i < 8; i += 1)
    acc += x[i0][i] * w[i1][i];
  output[i0][i1] = acc + b[i1];
}
```

Expected output (seed 202608270, float32):

```text
np.dot(x,w)+b: -1.94688380e+00   (NumPy, pairwise summation)
sequential : -1.94688380e+00   (kernel accumulation order)
CUDA output: -1.94688392e+00
max abs error: 1.192e-07
max rel error: 6.123e-08
dot_demo: PASSED (atol=1e-05, rtol=1e-05)
```

> **Correctness basis**: `Operators_Sem_Equiv.Linear2d_sem` (:812) and
> `Bridge_AlexNet.Linear2d_sem_single` (:951) prove
> `eval (A(output, Linear2d w b input)) = eval (Linear2d_comm w b input output)`;
> the proof skeleton is `AmkvSeq` (unrolling mkvSeq) + `C_sum_post_acc`
> (the `A_sum` family: reduction becomes a sequential accumulator loop).
> The 1e-07-scale residual comes from nvcc's default fmad contraction and
> the float32 accumulation order, not from a semantic error.

## 5. Demo 1: ReLU (element-wise)

`ReLU4d`: four nested `mkvSeq` layers, the cell is
`if x <= 0 then 0 else x` (`Ops_Elementwise.v:26`). The rewriting uses
`AmkvSeq`×4 + `Aif1` (DPIA_Rules.v:185) + `A_num` (:97). Note: the `comm`
consumed by the pipeline is the CG world's
`CNN_Examples_Base_Ops.ReLU4d_comm` (the same one the AlexNet pipeline
uses); `Ops_Elementwise.ReLU4d_comm` is the route-A twin definition, and
the two are structurally identical.

Expected output (seed 202608271, shape 1×2×4×4; `input[0,0]` shown):

```text
NumPy ref  : [[1.6586955  0.         0.         0.        ]
              [0.         1.1207734  0.92978925 0.        ]
              [0.         0.         0.         0.        ]
              [0.69420934 0.         0.55631363 0.        ]]
CUDA output: (element-wise identical)
max abs error: 0.000e+00
relu_demo: PASSED (atol=1e-05, rtol=1e-05)
```

> **Correctness basis**: `Ops_Elementwise.A_ReLU4d` (:45, the route-A
> Leibniz equation `A(output, ReLU4d input) = ReLU4d_comm input output`);
> the CG-world counterpart is `CNN_Examples_Base_Ops.ReLU4d_equiv_sem`
> (:74). ReLU has no accumulation, so the floating-point results are
> bit-identical and the error is exactly 0.

## 6. Demo 2: small convolution Conv2d (reduction)

`Conv4d` p=1, k=3, s=1: input 1×1×8×8, two 3×3 kernels, output 1×2×8×8.
Each output cell is a c×k1×k2 = 1×3×3 reduction plus bias. This demo uses
`CNN_Examples_Base.Conv4d_p1_k3_comm` (:1122) — the same "materialized
padding buffer" variant the AlexNet pipeline uses: the AST first
allocates (`new`) a pad array and writes the zero padding, then reduces.
The OCaml pattern matcher recognizes this structure, and the kernel
template switches to **implicit boundary checks** (see
`pi < 1 || pi >= 9 … ? 0.0f : …` in `build/conv_demo_cuda_output.cu`).

Expected output (seed 202608272; top-left 2×2 of the first two channels):

```text
NumPy ref [0,:, :2, :2]  : [[[-0.16845727  3.1387584 ] [ 1.2508518  1.9910562]]
                            [[-2.884582    0.7280201 ] [-1.9517119  1.3395951]]]
CUDA output[0,:, :2, :2] : [[[-0.16845727  3.1387582 ] [ 1.2508518  1.9910562]]
                            [[-2.884582    0.72801983] [-1.9517119  1.3395951]]]
max abs error: 9.537e-07
max rel error: 1.552e-06
conv_demo: PASSED (atol=1e-05, rtol=1e-05)
```

> **Correctness basis**: the general form `Ops_Conv.A_Conv4d` (:194, the
> `pad_idx2d`-based `Conv4d_comm`, rule skeleton `AmkvSeq`×4 + `A_add_post`
> + `C_sum_post_acc`) and `Bridge_AlexNet.Conv4d_sem_single` (:557, the
> `Conv4d_ast_comm` variant) are both proved (`Qed`).
> **Full disclosure**: for the "materialized padding" variant
> `Conv4d_p1_k3_comm` actually consumed by the pipeline, the functional
> equation exists in route B only as an Axiom statement for the stride-2
> sibling (`CNN_Examples_Base.v:1446`); the p1/k3 variant's equivalence
> currently rests on structural correspondence with the proved variants
> above plus numerical validation (the same trust boundary as in the root
> `README.md`, §Trust boundary).

## 7. Demo 3: end-to-end network (AlexNet, reusing the existing pipeline)

`make demo3` (i.e. `python3 run_demo3.py`) adds no new generator code; it
directly drives the AlexNet CUDA validation chain in
`experiments/ast_codegen/`: `alexnet_comm` (extracted from
`CNN_Examples_AlexNet.v`) → `comm_to_cuda_alexnet.ml` (pattern matching +
Conv+ReLU fusion, 10 components → 8 kernels) → wrap → nvcc → run → NumPy
comparison. Network: 1×3×8×8 → Conv+ReLU+Pool ×2 → Flatten → Linear×2 →
Softmax.

Expected output (seed 123, scale 1e-3):

```text
NumPy reference: [0.09978928 0.10000253 0.09982363 0.10011248 0.10015771
                  0.09997978 0.10023294 0.10001698 0.09994307 0.09994157]
CUDA output:     [0.09978929 0.10000253 0.09982362 0.10011248 0.10015772
                  0.09997979 0.10023293 0.10001698 0.09994308 0.09994157]
max abs error: 1.490e-08
max rel error: 1.488e-07
alexnet_test: PASSED (atol=1e-5, rtol=1e-5)
```

> **Correctness basis**: the end-to-end theorem `AlexNet_DPIA.A_AlexNet`
> (:456, the functional↔imperative Leibniz equation for full-size AlexNet);
> the per-operator bridge theorems are in `src/Ionia/Bridge_AlexNet.v`
> (`bridge_ReLU4d_buf` :391, `Conv4d_sem_single` :557,
> `Linear2d_sem_single` :951, `Softmax2d_sem_single` :1099, etc.).

## 8. Comparison with other approaches & "generation is proof"

Numerical comparison (measured on the reference machine, float32, fixed
seeds; error = max |CUDA − NumPy reference|):

| Demo | Operator | Reference | max abs err | max rel err |
|---|---|---|---|---|
| 0 | dot product (Linear2d 1×8→1) | `np.dot(x,w)+b` (a same-accumulation-order reference agrees to 1e-7) | 1.192e-07 | 6.123e-08 |
| 1 | ReLU4d 1×2×4×4 | `np.maximum(x,0)` | 0 (bit-identical) | 0 |
| 2 | Conv4d p1k3, 1×1×8×8→1×2×8×8 | NumPy same-order reduction conv | 9.537e-07 | 1.552e-06 |
| 3 | AlexNet whole net (8 kernels) | NumPy layer-by-layer same-order reference | 1.490e-08 | 1.488e-07 |

The proof behind each demo (where "generation is proof" lands):

| Demo | Functional definition | Operator equation / bridge theorem (file:line) |
|---|---|---|
| 0 | `CNN_OP.Linear2d` | `Operators_Sem_Equiv.Linear2d_sem` :812; `Bridge_AlexNet.Linear2d_sem_single` :951 |
| 1 | `Ops_Elementwise.ReLU4d` | `Ops_Elementwise.A_ReLU4d` :45; `CNN_Examples_Base_Ops.ReLU4d_equiv_sem` :74 |
| 2 | `CNN_OP.Conv4d` | `Ops_Conv.A_Conv4d` :194; `Bridge_AlexNet.Conv4d_sem_single` :557 (see also the §6 disclosure) |
| 3 | `AlexNet_DPIA` network function | `AlexNet_DPIA.A_AlexNet` :456; per-operator bridge lemmas in `src/Ionia/Bridge_AlexNet.v` |

**What "generation is proof" means, and its boundary**: every rewriting
step from the functional expression to `comm` is a Coq theorem (the rules
`A_sum`, `AmkvSeq`, `A_mul` themselves are also `Qed`, see
`src/Ionia/DPIA_Rules.v`), so "the imperative implementation of this layer
computes the same mathematical function" is machine-checked. The
**unproved parts, backed by numerical testing** (the trust base): the
OCaml pattern matchers and kernel templates (the heuristic AST→CUDA
translation), the IO wrapper `wrap_generated_cuda.py`, and nvcc itself.
That is exactly why the numerical comparisons above exist: the proved part
guarantees the semantics, and the unproved part is covered by element-wise
comparison.

## 9. File inventory and provenance

```text
demo/
  README.md            this document
  Makefile             demo0..demo3 / all / clean
  coq/Demo_Comms.v     concrete instantiations of the three operators + Extraction
  ocaml/demo_to_cuda.ml CUDA printer (copied from
                        experiments/ast_codegen/comm_to_cuda_alexnet.ml,
                        only the open module and main changed; the file
                        header records the provenance)
  ocaml/demo_sym.ml    (generated) demo_comm.ml after transform_to_symbolic.py
  run_demo0/1/2.py     data generation + run + comparison for the three demos (fixed seeds)
  run_demo3.py         drives the AlexNet CUDA validation of experiments/ast_codegen
  util.py              shared: nvcc resolution, wrapping, compiling, running, error report
  build/               (generated) *_cuda_output.cu, *.cu, executables, .bin
```

The only file outside `demo/` that the demos rely on having demo-specific
entries is `experiments/ast_codegen/wrap_generated_cuda.py`, whose
`TENSOR_SPECS` already ships with the `dot_demo` / `relu_demo` /
`conv_demo` entries. No proof file under `src/Ionia/` is modified by the
demos.
