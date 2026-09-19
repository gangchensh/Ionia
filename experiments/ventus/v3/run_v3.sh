#!/usr/bin/env bash
# Ionia Ventus V3 end to end: OCaml printer (--lenet5) -> 13 OpenCL kernels
# -> chained host build -> spike simulation -> per-layer NumPy check
# Usage: experiments/ventus/v3/run_v3.sh  (log written to experiments/ventus/logs/)
set -euo pipefail

V3="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENTUS_DIR="$(cd "$V3/.." && pwd)"
PROJ_ROOT="$(cd "$VENTUS_DIR/../.." && pwd)"
LOG_DIR="$VENTUS_DIR/logs"
mkdir -p "$LOG_DIR" "$V3/kernels" "$V3/data" "$V3/build"
LOG="$LOG_DIR/v3_spike_20260831.log"

{
  source "$VENTUS_DIR/env.sh"

  echo "== 1. Build the comm_to_opencl printer =="
  cd "$PROJ_ROOT/experiments/ast_codegen"
  ocamlc -g -c lenet5_sym.ml -o lenet5_sym.cmo
  ocamlc -g -o "$V3/build/comm_to_opencl" unix.cma lenet5_sym.cmo comm_to_opencl.ml

  echo "== 2. Generate the LeNet-5 OpenCL kernel chain =="
  "$V3/build/comm_to_opencl" "$V3/kernels" --lenet5
  ls -la "$V3/kernels"

  echo "== 3. Build the chained host =="
  clang++ -g -O2 -std=c++11 -ffp-contract=off "$V3/v3_host.cc" \
    -o "$V3/build/v3_host" \
    -I"$VENTUS_INSTALL_PREFIX/include" -L"$VENTUS_INSTALL_PREFIX/lib" -lOpenCL

  echo "== 4. spike simulation + per-layer NumPy check =="
  cd "$V3"
  VENTUS_BACKEND=spike python3 run_v3.py
} 2>&1 | tee "$LOG"
