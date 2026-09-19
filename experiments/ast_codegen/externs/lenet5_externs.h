#ifndef LENET5_EXTERNS_H
#define LENET5_EXTERNS_H

/* External arrays referenced by lenet5_comm (1x32x32 LeNet-5). */

extern float input[1][32][32];

extern float kc1[6][1][5][5];
extern float bc1[6];
extern float c1[6][28][28];
extern float r1[6][28][28];
extern float p1[6][14][14];

extern float kc2[16][6][5][5];
extern float bc2[16];
extern float c2[16][10][10];
extern float r2[16][10][10];
extern float p2[16][5][5];

extern float flat[400];   /* 16*5*5 */

extern float fc1_w[120][400];
extern float fc1_b[120];
extern float fc1[120];
extern float fc1_relu[120];

extern float fc2_w[84][120];
extern float fc2_b[84];
extern float fc2[84];
extern float fc2_relu[84];

extern float fc3_w[10][84];
extern float fc3_b[10];
extern float fc3[10];

extern float output[10];

#endif
