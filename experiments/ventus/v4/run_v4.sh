#!/usr/bin/env bash
# Ionia Ventus V4 end to end: OCaml printer (--alexnet/--resnet18) -> OpenCL
# kernel chains -> chained host build -> spike simulation -> per-layer NumPy check
# Usage: experiments/ventus/v4/run_v4.sh [alexnet|resnet18|all] (default: all;
# log written to experiments/ventus/logs/)
set -euo pipefail

V4="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENTUS_DIR="$(cd "$V4/.." && pwd)"
PROJ_ROOT="$(cd "$VENTUS_DIR/../.." && pwd)"
LOG_DIR="$VENTUS_DIR/logs"
mkdir -p "$LOG_DIR" "$V4/kernels_alexnet" "$V4/kernels_resnet18" \
         "$V4/data_alexnet" "$V4/data_resnet18" "$V4/build"
WHICH="${1:-all}"
LOG="$LOG_DIR/v4_spike_${WHICH}_20260831.log"

{
  source "$VENTUS_DIR/env.sh"

  echo "== 1. Build the comm_to_opencl printer =="
  cd "$PROJ_ROOT/experiments/ast_codegen"
  ocamlc -g -c lenet5_sym.ml -o lenet5_sym.cmo
  ocamlc -g -o "$V4/build/comm_to_opencl" unix.cma lenet5_sym.cmo comm_to_opencl.ml

  echo "== 2. Generate the OpenCL kernel chains =="
  "$V4/build/comm_to_opencl" "$V4/kernels_alexnet" --alexnet
  "$V4/build/comm_to_opencl" "$V4/kernels_resnet18" --resnet18

  echo "== 3. Build the chained host =="
  clang++ -g -O2 -std=c++11 -ffp-contract=off "$V4/v4_host.cc" \
    -o "$V4/build/v4_host" \
    -I"$VENTUS_INSTALL_PREFIX/include" -L"$VENTUS_INSTALL_PREFIX/lib" -lOpenCL

  echo "== 4. spike simulation + per-layer NumPy check =="
  cd "$V4"
  if [ "$WHICH" = "alexnet" ] || [ "$WHICH" = "all" ]; then
    echo "-- AlexNet --"
    VENTUS_BACKEND=spike python3 run_v4_alexnet.py
  fi
  if [ "$WHICH" = "resnet18" ] || [ "$WHICH" = "all" ]; then
    echo "-- ResNet-18 --"
    VENTUS_BACKEND=spike python3 run_v4_resnet18.py
  fi
} 2>&1 | tee "$LOG"
