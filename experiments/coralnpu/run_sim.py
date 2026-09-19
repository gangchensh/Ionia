#!/usr/bin/env python3
"""run_sim.py — run the examples on CoralNPUV2Simulator (mpact
instruction-level simulator) and verify the results.

Usage:
  python3 run_sim.py hello_world_add_floats
  python3 run_sim.py vecadd32_rvv

Dependencies:
  - a built coralnpu_v2_sim_pybind.so (bazel output; see the SIM_SO search paths)
  - coralnpu_ws/sw/coralnpu_sim/coralnpu_v2_sim_utils.py
  - numpy + pyelftools (python3.11 matching the .so ABI)

Verification protocol:
  hello_world_add_floats: host writes input1[i]=i*0.5, input2[i]=1.25; expects output=input1+input2 (fp32)
  vecadd32_rvv:           host writes input1[i]=i, input2[i]=1000-i; expects output[i]=1000 (int32)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.join(HERE, "coralnpu_ws")

def _find_sim():
    cands = [
        os.path.join(HERE, "build", "coralnpu_v2_sim_pybind.so"),
        os.path.join(WS, "bazel-out", "k8-fastbuild", "bin", "sw", "coralnpu_sim",
                     "coralnpu_v2_sim_pybind.so"),
    ]
    # bazel-out is a symlink and may be broken on some filesystems; also scan the default output_base
    import glob
    cands += glob.glob(os.path.expanduser(
        "~/.cache/bazel/_bazel_*/*/execroot/coralnpu_hw/bazel-out/*/bin/"
        "sw/coralnpu_sim/coralnpu_v2_sim_pybind.so"))
    for c in cands:
        if os.path.isfile(c):
            return os.path.realpath(c)
    raise SystemExit("coralnpu_v2_sim_pybind.so not found; run the bazel build first (see README.md)")

sim_dir = os.path.dirname(_find_sim())
sys.path.insert(0, sim_dir)
sys.path.insert(0, os.path.join(WS, "sw", "coralnpu_sim"))

import numpy as np
from coralnpu_v2_sim_utils import CoralNPUV2Simulator


def main():
    prog = sys.argv[1] if len(sys.argv) > 1 else "hello_world_add_floats"
    elf_path = os.path.join(HERE, "build", f"{prog}.elf")
    if not os.path.isfile(elf_path):
        raise SystemExit(f"missing {elf_path}; run build_examples.sh first")

    sim = CoralNPUV2Simulator()  # highmem memory map, matching the linker script
    if prog == "hello_world_add_floats":
        syms = ["input1", "input2", "output"]
    else:
        syms = ["input1", "input2", "output"]
    entry, symbol_map = sim.get_elf_entry_and_symbol(elf_path, syms)
    sim.load_program(elf_path, entry)

    if prog == "hello_world_add_floats":
        in1 = (np.arange(8, dtype=np.float32) * 0.5)
        in2 = np.full(8, 1.25, dtype=np.float32)
        sim.write_memory(symbol_map["input1"], in1)
        sim.write_memory(symbol_map["input2"], in2)
        expected = in1 + in2
        n_bytes = 32
        dtype = np.float32
    elif prog == "vecadd32_rvv":
        in1 = np.arange(32, dtype=np.int32)
        in2 = (1000 - np.arange(32)).astype(np.int32)
        sim.write_memory(symbol_map["input1"], in1)
        sim.write_memory(symbol_map["input2"], in2)
        expected = in1 + in2
        n_bytes = 128
        dtype = np.int32
    elif prog == "vcopy32":  # pure vector copy: isolates vload/vstore from vfadd
        in1 = (np.arange(32, dtype=np.int32) * 7 + 3)
        sim.write_memory(symbol_map["input1"], in1)
        expected = in1
        n_bytes = 128
        dtype = np.int32
    elif prog == "vwmac8":  # int8 widening MAC (vwmacc; probes the int8 accumulation path)
        in1 = np.arange(-8, 8, dtype=np.int8)
        in2 = np.full(16, 6, dtype=np.int8)
        sim.write_memory(symbol_map["input1"], in1)
        sim.write_memory(symbol_map["input2"], in2)
        expected = (in1.astype(np.int16) * in2.astype(np.int16))
        n_bytes = 32
        dtype = np.int16
    else:  # vecadd32_fp32_rvv (zve32f vector-fp probe)
        in1 = (np.arange(32, dtype=np.float32) * 0.25)
        in2 = np.full(32, 0.5, dtype=np.float32)
        sim.write_memory(symbol_map["input1"], in1)
        sim.write_memory(symbol_map["input2"], in2)
        expected = in1 + in2
        n_bytes = 128
        dtype = np.float32

    sim.run()
    sim.wait()

    out = sim.read_memory(symbol_map["output"], n_bytes)
    out = np.frombuffer(out.tobytes() if not isinstance(out, bytes) else out,
                        dtype=dtype)
    ok = np.array_equal(out, expected)
    try:
        print(f"[{prog}] cycles = {sim.get_cycle_count()}")
    except Exception:
        pass
    print(f"[{prog}] output = {out.tolist()}")
    print(f"[{prog}] expected= {expected.tolist()}")
    print(f"[{prog}] {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
