> **Release note (2026-09):** this README describes the pipeline's design.
> In this release the extracted/patched `*_sym.ml` ASTs are **shipped
> pre-generated**, so `make`, `make run`, `make wrapped`, `make validate`,
> `make validate-minitransformer` and the Triton/OpenCL targets work
> standalone. The regeneration script `extract_and_print.sh` and
> `src/Ionia/Extract_Comms.v` live in the main development repository and
> are not part of this release. See `../../ARTIFACT.md` for the
> authoritative quick start and the list of validated networks.

# Streaming OCaml code generator for Ionia `comm` AST

This directory contains the extraction-based pipeline:

1. Extract the unevaluated `comm` AST from Coq (not the C string).
2. Patch the extracted OCaml with a symbolic runtime and path-indexed
   `vwt`/`vrd` so array indices are preserved.
3. Run a streaming OCaml printer that walks the AST and writes C code to disk.
4. Wrap the generated body with `extern` declarations so it compiles as a C
   translation unit.

## Files

* `alexnet_sym.ml`, `lenet5_sym.ml`, `lenet5_par_sym.ml`,
  `alexnet_par_sym.ml`, `resnet18_sym.ml`, `resnet18_par_sym.ml`,
  `vgg16s_sym.ml`, `vgg16p_sym.ml`, `minitransformer_*_sym.ml` — patched
  ASTs for the shipped networks, pre-generated (steps 1–2 already done).
* `comm_to_c*.ml` / `comm_to_cuda*.ml` / `comm_to_triton.ml` /
  `comm_to_opencl.ml` — streaming printers, one per network and backend.
* `externs/*.h` — `extern` declarations for inputs, weights and intermediate
  activation buffers of each network.
* `wrap_generated_c.py` — wraps a generated body with the matching `externs/*.h`
  and a `void network()` function.
* `wrap_generated_cuda.py` — same for the CUDA backends (memory/IO wrapper,
  not verified).
* `build_network_sym.py` — builds a patched `*_sym.ml` from a pure Coq-extracted
  `*_comm.ml` and the fully patched `alexnet_sym.ml` base.
* `validate_*.py` / `validate_cuda_*.py` — numerical comparison of the
  generated code against NumPy references.
* `bench_*.py` — performance benchmarks (see `../../ARTIFACT.md`).

## Quick start

From this directory:

```bash
make          # builds the default sequential C printers
make run      # generates *_streaming_output.c
make wrapped  # generates *_wrapped.c and checks each with gcc -std=c99 -c
make validate # numerical check of LeNet-5 / AlexNet / ResNet-18 vs NumPy
```

Other useful targets (see the `Makefile` for the full list):

```bash
make validate-minitransformer   # MiniTransformer sequential + OpenMP C vs NumPy
make triton-lenet5              # Triton backend (also: triton-alexnet, triton-resnet18)
make comm_to_opencl             # OpenCL (Ventus) printer
make cuda-alexnet               # CUDA backends: cuda-<net>[-tiled|-fused|-igemm]
```

The end-to-end flow (Coq extraction → patched OCaml → streaming C
generation → wrapped compilable C) is driven from the repository root;
see `../../ARTIFACT.md`.

## Results

| Network | Input | Generated body bytes | Wrapped C compiles? | Numerical match |
|---|---|---|---|---|
| alexnet | 1×3×8×8 | 4232 | yes | yes |
| lenet5 | 1×32×32 | 3071 | yes | yes |

Both generated `*_wrapped.c` files compile with `gcc -std=c99 -c`, and
both produce output that matches a Python reference implementation on
random inputs/weights (within `1e-4` absolute/relative tolerance).
ResNet-18, VGG16 and MiniTransformer validate the same way via their
`validate_*.py` scripts; see `../../ARTIFACT.md` for the current numbers.

Peak memory for the streaming printer stays at a few megabytes regardless of
output size (only AST depth + 64 KiB buffer), in contrast to the old Coq
`vm_compute` pipeline which expanded the full output string inside Coq and
exhausted 19 GiB RAM + 3.4 GiB swap on WSL.

## Adding a new network

1. Extract the network's `comm` AST from the Coq development (the
   extraction driver lives in the main repository, not in this release;
   see `../../ARTIFACT.md`).
2. Add an entry to `build_network_sym.py` with the line number where
   network-specific helpers begin in the extracted `.ml`.
3. Add `externs/my_network_externs.h` declaring inputs, weights and all
   intermediate buffers.
4. Create a printer driver `comm_to_c_my_network.ml` (copy and tweak an
   existing one).
5. Extend the `NETWORKS` list in the `Makefile`.

## Remaining work

* Tighter integration with the root `Makefile.coq` so extraction and printing
  happen in one `make` invocation.
* Generate `externs/*.h` automatically from the `comm` AST instead of
  hand-writing declarations.
