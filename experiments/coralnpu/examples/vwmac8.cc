// vwmac8: 16-element int8 widening MAC (vwmacc.vv; probes the zve32x integer vector multiply-add path)
// acc(e16m2) += sext(a_e8m1) * sext(b_e8m1); the host writes input1/input2 and reads output after halt.
// VLEN=128: e8m1 processes 16 int8 per beat, widened to e16m2.

#define N 16

int8_t input1[N] __attribute__((section(".data")));
int8_t input2[N] __attribute__((section(".data")));
int16_t output[N] __attribute__((section(".data")));

int main() {
  size_t vl8 = __riscv_vsetvl_e8m1(N);
  vint8m1_t a = __riscv_vle8_v_i8m1(input1, vl8);
  vint8m1_t b = __riscv_vle8_v_i8m1(input2, vl8);
  vint16m2_t acc = __riscv_vmv_v_x_i16m2(0, vl8);
  acc = __riscv_vwmacc_vv_i16m2(acc, a, b, vl8);
  __riscv_vse16_v_i16m2(output, acc, vl8);
  return 0;
}
