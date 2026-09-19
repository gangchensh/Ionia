#ifndef ALEXNET_PAR_EXTERNS_H
#define ALEXNET_PAR_EXTERNS_H

/* External arrays referenced by the classic AlexNet AST
   (Bridge_AlexNet.alexnet_gen_comm / AlexNet_Par.alexnet_par_gen_comm):
   1x3x224x224 input, 5 conv + 3 fc, batch = 1.
   Inputs, weights, and all intermediate activation buffers are pre-allocated
   by the caller. */

extern float input[1][3][224][224];

extern float c1_w[64][3][11][11];
extern float c1_b[64];
extern float c1[1][64][55][55];
extern float r1[1][64][55][55];
extern float p1[1][64][27][27];

extern float c2_w[192][64][5][5];
extern float c2_b[192];
extern float c2[1][192][27][27];
extern float r2[1][192][27][27];
extern float p2[1][192][13][13];

extern float c3_w[384][192][3][3];
extern float c3_b[384];
extern float c3[1][384][13][13];
extern float r3[1][384][13][13];

extern float c4_w[256][384][3][3];
extern float c4_b[256];
extern float c4[1][256][13][13];
extern float r4[1][256][13][13];

extern float c5_w[256][256][3][3];
extern float c5_b[256];
extern float c5[1][256][13][13];
extern float r5[1][256][13][13];
extern float p5[1][256][6][6];

extern float flat[1][9216];

extern float f1_w[4096][9216];
extern float f1_b[4096];
extern float fc1[1][4096];
extern float fc1r[1][4096];

extern float f2_w[4096][4096];
extern float f2_b[4096];
extern float fc2[1][4096];
extern float fc2r[1][4096];

extern float f3_w[1000][4096];
extern float f3_b[1000];
extern float fc3[1][1000];

extern float output[1][1000];

#endif
