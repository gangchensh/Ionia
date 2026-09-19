# Ventus (RISC-V GPGPU) spike-simulator validation — paper §4.5

Evidence for the paper's Ventus claim: the OpenCL kernel chains emitted by
Ionia's `comm_to_opencl` printer (backed by the proved three-level
parallelization rules of `src/Ionia/Opt_HierPar.v`) were executed
**end to end on the vendor spike instruction-level simulator** and validated
layer by layer against FP64 NumPy references:

| Network | Kernels | Result | Final softmax max abs err | Log |
|---|---|---|---|---|
| LeNet-5 | 13 | 13/13 PASS | 7.451e-9 (paper: 7.5e-9) | `logs/v3_spike_20260831.log` |
| AlexNet (reduced sizes) | 20 | 20/20 PASS | 1.445e-9 (paper: 1.4e-9) | `logs/v4_spike_alexnet_20260831.log` |
| ResNet-18 (reduced sizes) | 69 | 69/69 PASS | 1.470e-9 (paper: 1.5e-9) | `logs/v4_spike_resnet18_20260831.log`, `logs/v4_resnet18_offline_verify_20260901.log` |

The simulator is functional only (RV32 `rv32gcv_zfh`, vlen 1024); no
performance claim is made for this target.

## Layout

- `v2a/` — single-operator bring-up: 6 generated OpenCL kernels
  (relu1d/add1d/sum256/sum4096/max256/conv2d) + `manifest.txt`, a generic
  OpenCL host (`v2a_host.cc`), and the driver `run_v2a.py` that generates
  seeded inputs, runs each kernel on spike, and compares against NumPy
  (6/6 PASS; log `logs/v2a_spike_20260831.log`).
- `v3/` — full LeNet-5: 13 generated kernels, chained host (`v3_host.cc`,
  all kernels in one `cl_program`, intermediate buffers device-resident),
  driver `run_v3.py` using the same RNG protocol as the CPU validator
  (`experiments/ast_codegen/validate_lenet5_par.py`, seed 7).
- `v4/` — AlexNet and ResNet-18 at reduced sizes (AlexNet: input 3x48x48,
  channels /4, fc 1024/1024/256, seed 42; ResNet-18: input 3x40x40,
  channels /4, fc 256, seed 18), generated kernel sets
  `kernels_alexnet/` (20), `kernels_resnet18/` (69),
  `kernels_resnet18_b/` (second spike segment), chained host `v4_host.cc`,
  drivers `run_v4_alexnet.py` / `run_v4_resnet18.py` and the offline
  re-verifier `run_v4_resnet18_verify.py`.
- `logs/` — captured end-to-end run logs, including the V1 toolchain
  smoke tests (`probe_spike.log`: float4 / `__local`+barrier / 2D-grid
  feature probes; `vecadd_*_spike.log`, `vfma32_spike.log`: first kernels
  executed on the real toolchain).

## How the validation works

1. `comm_to_opencl` (OCaml printer, built from
   `experiments/ast_codegen/comm_to_opencl.ml`) emits one `.cl` kernel per
   network layer plus a `manifest.txt` (kernel name, global/local work size,
   typed parameter list).
2. A small C++ OpenCL host compiles the kernels with the Ventus pocl-based
   OpenCL implementation and launches them in manifest order; the vendor
   runtime lowers OpenCL C to RISC-V (`-triple=riscv32
   -target-cpu ventus-gpgpu`) and executes it on the spike simulator.
3. After each kernel the host dumps the output buffer; the Python driver
   compares every layer against an FP64 NumPy reference that mirrors the
   Coq semantics (zero-padding conv, bias added last, softmax without max
   subtraction, (c,i,j) row-major flatten). PASS threshold: max abs diff
   <= 1e-5; observed diffs are ~1e-10 per layer, ~1.5e-9 at the final
   softmax.

## Re-running (external dependency)

The Ventus toolchain itself (LLVM/pocl OpenCL implementation, spike
simulator, driver runtime) is **not** part of this artifact; install the
vendor toolchain separately and point `env.sh` at it:

```bash
export VENTUS_INSTALL_PREFIX=/path/to/ventus/toolchain/install   # or edit env.sh
experiments/ventus/v2a/run_v2a.sh        # 6 single operators
experiments/ventus/v3/run_v3.sh          # LeNet-5 chain
experiments/ventus/v4/run_v4.sh all      # AlexNet + ResNet-18 chains
```

The scripts rebuild `comm_to_opencl` from `experiments/ast_codegen` (paths
are relative to the artifact root, so they work from this layout), compile
the host against `$VENTUS_INSTALL_PREFIX`, run spike, and tee a log into
`experiments/ventus/logs/`. Also required: OCaml (printer build), clang++,
Python 3 + NumPy.

Note: the AlexNet log ends with an offline re-verification section — the
first spike run completed all 20 layers (412.49 s) but the validation pass
crashed on a Python bug in the pooling *reference*; the fixed reference was
re-checked against the already-dumped spike outputs without re-running spike.
Input data and spike dumps (`.data`/`.bin`, regenerated deterministically
from the seeds above) and the multi-hundred-MB per-instruction spike traces
are not shipped.
