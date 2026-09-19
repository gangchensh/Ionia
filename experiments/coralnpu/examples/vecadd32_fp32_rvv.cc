// vecadd32_fp32_rvv: 32-element fp32 vector add (zve32f vector-fp intrinsic probe)
// The host writes input1/input2 (32 x fp32 each) and reads output after halt.

#define N 32

float input1[N] __attribute__((section(".data")));
float input2[N] __attribute__((section(".data")));
float output[N] __attribute__((section(".data")));

int main() {
  size_t i = 0;
  while (i < N) {
    size_t vl = __riscv_vsetvl_e32m1(N - i);
    vfloat32m1_t va = __riscv_vle32_v_f32m1(input1 + i, vl);
    vfloat32m1_t vb = __riscv_vle32_v_f32m1(input2 + i, vl);
    __riscv_vse32_v_f32m1(output + i, __riscv_vfadd_vv_f32m1(va, vb, vl), vl);
    i += vl;
  }
  return 0;
}
