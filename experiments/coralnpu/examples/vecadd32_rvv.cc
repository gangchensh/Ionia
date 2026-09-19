// vecadd32_rvv: 32-element int32 vector add, RVV intrinsics (vsetvl strip-mining loop)
// Intrinsic usage follows coralnpu-main/examples/rvv_add_intrinsic.cc (Apache-2.0)
// The host writes input1/input2 (32 x int32 each) and reads output after halt.

#include <riscv_vector.h>
#include <stdint.h>

#define N 32

int32_t input1[N] __attribute__((section(".data")));
int32_t input2[N] __attribute__((section(".data")));
int32_t output[N] __attribute__((section(".data")));

int main() {
  size_t i = 0;
  while (i < N) {
    size_t vl = __riscv_vsetvl_e32m1(N - i);
    vint32m1_t va = __riscv_vle32_v_i32m1(input1 + i, vl);
    vint32m1_t vb = __riscv_vle32_v_i32m1(input2 + i, vl);
    __riscv_vse32_v_i32m1(output + i, __riscv_vadd_vv_i32m1(va, vb, vl), vl);
    i += vl;
  }
  return 0;
}
