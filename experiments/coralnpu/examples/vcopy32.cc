// vcopy32: 32x int32 pure vector copy (vle32+vse32, no arithmetic) — isolates vload/vstore from vfadd

#define N 32

int32_t input1[N] __attribute__((section(".data")));
int32_t input2[N] __attribute__((section(".data")));
int32_t output[N] __attribute__((section(".data")));

int main() {
  size_t i = 0;
  while (i < N) {
    size_t vl = __riscv_vsetvl_e32m1(N - i);
    vint32m1_t va = __riscv_vle32_v_i32m1(input1 + i, vl);
    __riscv_vse32_v_i32m1(output + i, va, vl);
    i += vl;
  }
  return 0;
}
