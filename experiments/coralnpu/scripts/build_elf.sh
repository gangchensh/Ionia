#!/bin/bash
# build_elf.sh — compile C/C++ directly to a CoralNPU RV32 ELF with the
# Coral NPU prebuilt toolchain (LLVM/clang 20 + gcc16 multilib, newlib).
# Mirrors the compile/link flags of toolchain/cc_toolchain_config.bzl.
#
# Usage: bash build_elf.sh <src.c|src.cc> [output name]
# Output: build/<name>.elf
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"          # experiments/coralnpu
WS=${CORALNPU_WS:-$HOME/ws/coralnpu}             # local workspace (unpacked toolchain + source tree)
TC=$WS/toolchain_coralnpu_v2                     # prebuilt toolchain (GCS tarball)
SRC_TREE=$WS/coralnpu-main                       # local coralnpu source-tree copy (only its toolchain/crt is referenced)
CRT=$SRC_TREE/toolchain/crt
BUILD=$HERE/build
mkdir -p "$BUILD" env

CXX=$TC/bin/clang++
# Link via the GCC driver: the clang driver does no directory selection for
# this newlib multilib (it would only use the base rv64 libraries); the
# multilib reuse rules of riscv64-unknown-elf-gcc/g++ 16.1 correctly select
# rv32imf.../ilp32.
LINKER=$TC/bin/riscv64-unknown-elf-g++
ARCH="-march=rv32imf_zve32f_zicsr_zifencei_zbb_zfbfmin_zvfbfmin_zvfbfwma -mabi=ilp32 -mcmodel=medany"
CFLAGS="$ARCH -nostdlib --sysroot=$TC/riscv64-unknown-elf --gcc-toolchain=$TC -O3 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -Wall"
LINKFLAGS="$ARCH -nostdlib --specs=nano.specs -nostartfiles -Wl,--gc-sections -Wl,--start-group -lstdc++ -lm -lc -lgcc -Wl,--end-group"

# Linker script: default instantiation of toolchain/coralnpu_tcm.ld.tpl (ITCM 8K@0x0, DTCM 32K@0x10000)
LD=$HERE/env/coralnpu_tcm.ld
if [ ! -f "$LD" ]; then
  sed -e 's/@@ITCM_LENGTH@@/8/' \
      -e 's/@@DTCM_LENGTH@@/32/' \
      -e 's/@@DTCM_ORIGIN@@/0x00010000/' \
      -e 's/@@STACK_SIZE@@/128/' \
      -e 's/@@HEAP_LOCATION@@/DTCM/' \
      -e 's/@@HEAP_SIZE_SPEC@@/. = ORIGIN(DTCM) + LENGTH(DTCM) - STACK_SIZE;/' \
      -e 's/@@STACK_START_SPEC@@//' \
      "$SRC_TREE/toolchain/coralnpu_tcm.ld.tpl" > "$LD"
fi

# CRT (the crt target of toolchain/crt/BUILD, -DSKIP_HTIF_SYMBOLS): compiled once and cached
CRTOBJS=()
for f in coralnpu_start.S crt.S coralnpu_gloss.cc coralnpu_exceptions.cc cxx_guards.cc; do
  obj="$BUILD/$(basename "$f").o"
  if [ ! -f "$obj" ]; then
    $CXX $CFLAGS -DSKIP_HTIF_SYMBOLS -c "$CRT/$f" -o "$obj"
  fi
  CRTOBJS+=("$obj")
done

src="$1"
name="$(basename "${2:-$(basename "$src")}")"; name="${name%.*}"
obj="$BUILD/$name.o"
$CXX $CFLAGS -c "$src" -o "$obj"
$LINKER $obj "${CRTOBJS[@]}" $LINKFLAGS \
  -Wl,-T,"$LD" -o "$BUILD/$name.elf"
echo "built $BUILD/$name.elf"
$TC/bin/riscv64-unknown-elf-readelf -h "$BUILD/$name.elf" | grep -E "Class|Machine|Entry"
