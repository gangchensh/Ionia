#!/usr/bin/env bash
# Ionia Ventus V2a end to end: OCaml printer -> OpenCL kernels -> host build -> spike simulation -> NumPy check
# Usage: experiments/ventus/v2a/run_v2a.sh  (log written to experiments/ventus/logs/)
set -euo pipefail

V2A="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENTUS_DIR="$(cd "$V2A/.." && pwd)"
PROJ_ROOT="$(cd "$VENTUS_DIR/../.." && pwd)"
LOG_DIR="$VENTUS_DIR/logs"
mkdir -p "$LOG_DIR" "$V2A/kernels" "$V2A/data" "$V2A/build"
LOG="$LOG_DIR/v2a_spike_20260831.log"

{
  source "$VENTUS_DIR/env.sh"

  echo "== 1. Build the comm_to_opencl printer =="
  cd "$PROJ_ROOT/experiments/ast_codegen"
  ocamlc -g -c lenet5_sym.ml -o lenet5_sym.cmo
  ocamlc -g -o "$V2A/build/comm_to_opencl" unix.cma lenet5_sym.cmo comm_to_opencl.ml

  echo "== 2. Generate OpenCL kernels =="
  "$V2A/build/comm_to_opencl" "$V2A/kernels"
  ls -la "$V2A/kernels"

  echo "== 3. Build the generic host =="
  clang++ -g -O2 -std=c++11 -ffp-contract=off "$V2A/v2a_host.cc" \
    -o "$V2A/build/v2a_host" \
    -I"$VENTUS_INSTALL_PREFIX/include" -L"$VENTUS_INSTALL_PREFIX/lib" -lOpenCL

  echo "== 4. spike simulation + NumPy check =="
  cd "$V2A"
  VENTUS_BACKEND=spike python3 run_v2a.py
} 2>&1 | tee "$LOG"
