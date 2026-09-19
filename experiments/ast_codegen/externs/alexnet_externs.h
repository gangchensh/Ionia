#ifndef ALEXNET_EXTERNS_H
#define ALEXNET_EXTERNS_H

/* External arrays referenced by alexnet_comm (CIFAR-style 1x3x8x8).
   Inputs, weights, and all intermediate activation buffers are pre-allocated
   by the caller. */

extern float input[1][3][8][8];

extern float conv1_w[8][3][3][3];
extern float conv1_b[8];
extern float conv1[1][8][8][8];
extern float conv1_relu[1][8][8][8];
extern float conv1_pool[1][8][4][4];

extern float conv2_w[16][8][3][3];
extern float conv2_b[16];
extern float conv2[1][16][4][4];
extern float conv2_relu[1][16][4][4];
extern float conv2_pool[1][16][2][2];

extern float flat[1][64];

extern float fc1_w[16][64];
extern float fc1_b[16];
extern float fc1[1][16];

extern float fc2_w[10][16];
extern float fc2_b[10];
extern float fc2[1][10];

extern float output[1][10];

#endif
