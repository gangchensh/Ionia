// HelloWorld (tutorial program): 8-element scalar fp32 addition
// Source: coralnpu-main/examples/hello_world_add_floats.cc (Apache-2.0)
// The host writes input1/input2 before the run and reads output after halt.

float input1[8] __attribute__((section(".data")));
float input2[8] __attribute__((section(".data")));
float output[8] __attribute__((section(".data")));

int main() {
  for (int i = 0; i < 8; i++) {
    output[i] = input1[i] + input2[i];
  }
  return 0;
}
