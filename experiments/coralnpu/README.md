# Coral NPU (RISC-V) toolchain validation — paper §4.5

Evidence for the paper's claim: *"A Coral NPU (RISC-V) toolchain is likewise
operational, with scalar-FP32, integer-RVV, and int8-MAC programs passing its
instruction-level simulator."*

The target is Google's open-source Coral NPU (RV32IMF + RVV
`zve32f`/`zve32x`, bf16 extensions). The simulator is the vendor's mpact-based
instruction-level simulator (`CoralNPUV2Simulator`, Python pybind wrapper
`coralnpu_v2_sim_pybind.so`), built from the vendor source tree with
Bazel 8.6.0; programs are compiled with the vendor's prebuilt RISC-V
toolchain (LLVM/clang 20.1.0 + gcc 16.1 multilib, newlib;
`-march=rv32imf_zve32f_zicsr_zifencei_zbb_zfbfmin_zvfbfmin_zvfbfwma -mabi=ilp32`).

## What was run, and the results

Five small bare-metal programs (`examples/`), each loaded into the simulator
by `run_sim.py`, which writes inputs to the `.data`-section buffers, runs to
halt, reads back the output buffer, and checks it:

| Program | ISA path exercised | Result | Cycles |
|---|---|---|---|
| `hello_world_add_floats` | scalar FP32 (`flw`/`fadd.s`) | PASS | 179 |
| `vecadd32_rvv` | integer RVV intrinsics (`vsetvli`/`vle32`/`vadd.vv`/`vse32`) | PASS | 245 |
| `vcopy32` | pure vector load/store (no arithmetic) | PASS | 221 |
| `vwmac8` | int8 widening MAC (`vwmacc.vv`, e8m1 → e16m2) | PASS | 157 |
| `vecadd32_fp32_rvv` | vector FP32 (`zve32f`) | **FAIL (expected)** — the mpact v2 ISA model has no vector-FP slot; the instruction traps to ebreak. Vector FP32 needs the RTL/Verilator path. |

Conclusion recorded from the runs: scalar FP32 (F extension), integer RVV,
and int8 `vwmacc` all work under the instruction-level simulator; bf16 is
available at compile time (`zvfbfmin`/`zvfbfwma`).

**Honesty note.** The simulator console transcripts of these five runs were
not preserved as log files; the PASS/FAIL results and cycle counts above are
the recorded outcomes (see also the compiled ELFs below, whose disassembly
shows the expected `vsetvli`/`vle32`/`vadd.vv`/`vse32`, `flw`/`fadd.s`, and
`vwmacc.vv` instructions). The `logs/` directory holds the Bazel build logs
of the simulator bring-up itself.

## Layout

- `examples/` — the five test programs (C/C++, RVV intrinsics).
- `build/` — the compiled RV32 ELF binaries (evidence; inspect with any
  RISC-V `readelf`/`objdump`).
- `build_examples.sh` — builds the examples to ELF with the prebuilt
  toolchain (documents the exact flags mirroring the vendor bazel rules).
- `scripts/build_elf.sh` — the generic single-file variant of the above.
- `run_sim.py` — loads an ELF into `CoralNPUV2Simulator`, drives the
  input/output buffers, prints PASS/FAIL and the cycle count.
- `env/` — TCM linker scripts (`coralnpu_tcm_highmem.ld`: ITCM 1MB @ 0x0,
  DTCM 1MB @ 0x100000, matching the simulator's default highmem map).
- `logs/` — bazel build logs for the simulator pybind module.

## Re-running (external dependencies)

The Coral NPU source tree, prebuilt toolchain, and the bazel-built simulator
are **not** part of this artifact. To reproduce: unpack the vendor
`toolchain_coralnpu_v2` tarball, check out the vendor `coralnpu` source tree,
build `//sw/coralnpu_sim:coralnpu_v2_sim_pybind` with bazel (the shipped logs
document the flag fixes needed on a gcc-only host), then:

```bash
bash build_examples.sh                 # writes build/*.elf
python3 run_sim.py hello_world_add_floats   # or vecadd32_rvv / vcopy32 / vwmac8
```

`run_sim.py` expects the simulator `.so` under `build/` or the bazel output
tree, a `coralnpu_ws/` checkout for `coralnpu_v2_sim_utils.py`, and a
python3.11 environment with numpy + pyelftools.
