#!/bin/bash
# build_examples.sh — compile the examples to RV32 ELF with the Coral NPU
# prebuilt RISC-V toolchain.
#
# Mirrors the compile flags of the bazel rules rules/coralnpu_v2.bzl +
# toolchain/cc_toolchain_config.bzl:
#   -march=rv32imf_zve32f_zicsr_zifencei_zbb_zfbfmin_zvfbfmin_zvfbfwma -mabi=ilp32
#   -mcmodel=medany -nostdlib + newlib-nano + crt (toolchain/crt)
# Two deliberate deviations from bazel:
#   1. -isystem ordering puts lib/clang/20/include before the gcc includes —
#      otherwise <riscv_vector.h> resolves to the GCC header and no RVV
#      intrinsic is available;
#   2. linking does not use --specs=nano.specs (the clang driver ignores
#      specs for baremetal riscv); explicit -B/-L point at the rv32imf
#      multilib newlib-nano/libgcc instead.
# Linker script: env/coralnpu_tcm_highmem.ld (ITCM 1MB@0x0 / DTCM
#   1MB@0x100000, matching the CoralNPUV2Simulator default highmem map)
#
# Usage: bash build_examples.sh [hello_world_add_floats|vecadd32_rvv]  (default: all)
set -euo pipefail
cd "$(dirname "$0")"

TC=toolchain/toolchain_coralnpu_v2
CRT=coralnpu_ws/toolchain/crt
MULTI=rv32imf_zicsr_zifencei_zmmul_zbb/ilp32
mkdir -p build

CXX="$TC/bin/clang++"
READELF="$TC/bin/riscv64-unknown-elf-readelf"
[ -x "$TC/bin/clang++" ] || { echo "ERROR: prebuilt toolchain not unpacked ($TC)" >&2; exit 1; }

ARCH="-march=rv32imf_zve32f_zicsr_zifencei_zbb_zfbfmin_zvfbfmin_zvfbfwma -mabi=ilp32 -mcmodel=medany"
INC="-nostdinc -isystem $TC/lib/clang/20/include \
 -isystem $TC/riscv64-unknown-elf/include/c++/16.1.0 \
 -isystem $TC/riscv64-unknown-elf/include/c++/16.1.0/backward \
 -isystem $TC/riscv64-unknown-elf/include/c++/16.1.0/riscv64-unknown-elf \
 -isystem $TC/lib/gcc/riscv64-unknown-elf/16.1.0/include \
 -isystem $TC/riscv64-unknown-elf/include"
CFLAGS="$ARCH -nostdlib $INC -O2 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti"
LDFLAGS="$ARCH -nostdlib -nostartfiles \
 -B$TC/riscv64-unknown-elf/lib/$MULTI -B$TC/riscv64-unknown-elf/lib \
 -B$TC/lib/gcc/riscv64-unknown-elf/16.1.0/$MULTI \
 -L$TC/riscv64-unknown-elf/lib/$MULTI -L$TC/lib/gcc/riscv64-unknown-elf/16.1.0/$MULTI \
 -Wl,--gc-sections -Wl,-T,env/coralnpu_tcm_highmem.ld"

# 1) CRT (add -DSKIP_HTIF_SYMBOLS for the non-semihosting variant)
CRT_OBJS=""
for f in coralnpu_start.S crt.S coralnpu_exceptions.cc coralnpu_gloss.cc cxx_guards.cc; do
  o="build/crt_$(basename "$f").o"
  $CXX $CFLAGS -DSKIP_HTIF_SYMBOLS -c "$CRT/$f" -o "$o"
  CRT_OBJS="$CRT_OBJS $o"
done

build_one() {  # $1 = example name
  $CXX $CFLAGS -c "examples/$1.cc" -o "build/$1.o"
  $CXX $LDFLAGS "build/$1.o" $CRT_OBJS \
      -Wl,--start-group -lstdc++ -lm -lc -lgcc -Wl,--end-group \
      -o "build/$1.elf"
  echo "built build/$1.elf"
}

progs="hello_world_add_floats vecadd32_rvv vcopy32 vwmac8"
[ $# -ge 1 ] && progs="$@"
for p in $progs; do build_one "$p"; done

$READELF -h build/hello_world_add_floats.elf | grep -E "Class|Machine|Entry" || true
echo done
