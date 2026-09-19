#!/usr/bin/env bash
# Environment for the Ventus OpenCL runs.
# Set VENTUS_INSTALL_PREFIX to the install prefix of the Ventus toolchain
# (the vendor RISC-V GPGPU toolchain: llvm/pocl OpenCL implementation, spike
# simulator, and driver — it is an external dependency and is NOT part of
# this artifact; see README.md).
# Example:
#   export VENTUS_INSTALL_PREFIX=/opt/ventus/toolchain/install
: "${VENTUS_INSTALL_PREFIX:?Set VENTUS_INSTALL_PREFIX to the Ventus toolchain install prefix}"
export VENTUS_INSTALL_PREFIX
