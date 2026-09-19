#ifndef RESNET18_EXTERNS_H
#define RESNET18_EXTERNS_H

/* Input / output */
extern float input[1][3][224][224];
extern float output[1][1000];

/* Stem: conv1 + BN + ReLU + maxpool */
extern float conv1_w[64][3][7][7];
extern float conv1_b[64];
extern float bn1_mean[64];
extern float bn1_var[64];
extern float bn1_gamma[64];
extern float bn1_beta[64];
extern float bn_eps;
extern float conv1_out[1][64][112][112];
extern float bn1_out[1][64][112][112];
extern float relu1_out[1][64][112][112];
extern float pool_out[1][64][56][56];

/* Layer 1: 2 identity blocks, 64 channels, 56x56 */
extern float l1_b1_w1[64][64][3][3];
extern float l1_b1_b1[64];
extern float l1_b1_w2[64][64][3][3];
extern float l1_b1_b2[64];
extern float l1_b1_bn1_mean[64];
extern float l1_b1_bn1_var[64];
extern float l1_b1_bn1_gamma[64];
extern float l1_b1_bn1_beta[64];
extern float l1_b1_bn2_mean[64];
extern float l1_b1_bn2_var[64];
extern float l1_b1_bn2_gamma[64];
extern float l1_b1_bn2_beta[64];
extern float l1_b1_conv1_out[1][64][56][56];
extern float l1_b1_bn1_out[1][64][56][56];
extern float l1_b1_relu1_out[1][64][56][56];
extern float l1_b1_conv2_out[1][64][56][56];
extern float l1_b1_bn2_out[1][64][56][56];
extern float l1_b1_add_out[1][64][56][56];

extern float l1_b2_w1[64][64][3][3];
extern float l1_b2_b1[64];
extern float l1_b2_w2[64][64][3][3];
extern float l1_b2_b2[64];
extern float l1_b2_bn1_mean[64];
extern float l1_b2_bn1_var[64];
extern float l1_b2_bn1_gamma[64];
extern float l1_b2_bn1_beta[64];
extern float l1_b2_bn2_mean[64];
extern float l1_b2_bn2_var[64];
extern float l1_b2_bn2_gamma[64];
extern float l1_b2_bn2_beta[64];
extern float l1_b2_conv1_out[1][64][56][56];
extern float l1_b2_bn1_out[1][64][56][56];
extern float l1_b2_relu1_out[1][64][56][56];
extern float l1_b2_conv2_out[1][64][56][56];
extern float l1_b2_bn2_out[1][64][56][56];
extern float l1_b2_add_out[1][64][56][56];

/* Layer 2: projection + identity, 128 channels, 28x28 */
extern float l2_b1_w1[128][64][3][3];
extern float l2_b1_b1[128];
extern float l2_b1_w2[128][128][3][3];
extern float l2_b1_b2[128];
extern float l2_b1_bn1_mean[128];
extern float l2_b1_bn1_var[128];
extern float l2_b1_bn1_gamma[128];
extern float l2_b1_bn1_beta[128];
extern float l2_b1_bn2_mean[128];
extern float l2_b1_bn2_var[128];
extern float l2_b1_bn2_gamma[128];
extern float l2_b1_bn2_beta[128];
extern float l2_b1_shortcut_w[128][64][1][1];
extern float l2_b1_shortcut_b[128];
extern float l2_b1_shortcut_mean[128];
extern float l2_b1_shortcut_var[128];
extern float l2_b1_shortcut_gamma[128];
extern float l2_b1_shortcut_beta[128];
extern float l2_b1_shortcut_out[1][128][28][28];
extern float l2_b1_conv1_out[1][128][28][28];
extern float l2_b1_bn1_out[1][128][28][28];
extern float l2_b1_relu1_out[1][128][28][28];
extern float l2_b1_conv2_out[1][128][28][28];
extern float l2_b1_bn2_out[1][128][28][28];
extern float l2_b1_add_out[1][128][28][28];

extern float l2_b2_w1[128][128][3][3];
extern float l2_b2_b1[128];
extern float l2_b2_w2[128][128][3][3];
extern float l2_b2_b2[128];
extern float l2_b2_bn1_mean[128];
extern float l2_b2_bn1_var[128];
extern float l2_b2_bn1_gamma[128];
extern float l2_b2_bn1_beta[128];
extern float l2_b2_bn2_mean[128];
extern float l2_b2_bn2_var[128];
extern float l2_b2_bn2_gamma[128];
extern float l2_b2_bn2_beta[128];
extern float l2_b2_conv1_out[1][128][28][28];
extern float l2_b2_bn1_out[1][128][28][28];
extern float l2_b2_relu1_out[1][128][28][28];
extern float l2_b2_conv2_out[1][128][28][28];
extern float l2_b2_bn2_out[1][128][28][28];
extern float l2_b2_add_out[1][128][28][28];

/* Layer 3: projection + identity, 256 channels, 14x14 */
extern float l3_b1_w1[256][128][3][3];
extern float l3_b1_b1[256];
extern float l3_b1_w2[256][256][3][3];
extern float l3_b1_b2[256];
extern float l3_b1_bn1_mean[256];
extern float l3_b1_bn1_var[256];
extern float l3_b1_bn1_gamma[256];
extern float l3_b1_bn1_beta[256];
extern float l3_b1_bn2_mean[256];
extern float l3_b1_bn2_var[256];
extern float l3_b1_bn2_gamma[256];
extern float l3_b1_bn2_beta[256];
extern float l3_b1_shortcut_w[256][128][1][1];
extern float l3_b1_shortcut_b[256];
extern float l3_b1_shortcut_mean[256];
extern float l3_b1_shortcut_var[256];
extern float l3_b1_shortcut_gamma[256];
extern float l3_b1_shortcut_beta[256];
extern float l3_b1_shortcut_out[1][256][14][14];
extern float l3_b1_conv1_out[1][256][14][14];
extern float l3_b1_bn1_out[1][256][14][14];
extern float l3_b1_relu1_out[1][256][14][14];
extern float l3_b1_conv2_out[1][256][14][14];
extern float l3_b1_bn2_out[1][256][14][14];
extern float l3_b1_add_out[1][256][14][14];

extern float l3_b2_w1[256][256][3][3];
extern float l3_b2_b1[256];
extern float l3_b2_w2[256][256][3][3];
extern float l3_b2_b2[256];
extern float l3_b2_bn1_mean[256];
extern float l3_b2_bn1_var[256];
extern float l3_b2_bn1_gamma[256];
extern float l3_b2_bn1_beta[256];
extern float l3_b2_bn2_mean[256];
extern float l3_b2_bn2_var[256];
extern float l3_b2_bn2_gamma[256];
extern float l3_b2_bn2_beta[256];
extern float l3_b2_conv1_out[1][256][14][14];
extern float l3_b2_bn1_out[1][256][14][14];
extern float l3_b2_relu1_out[1][256][14][14];
extern float l3_b2_conv2_out[1][256][14][14];
extern float l3_b2_bn2_out[1][256][14][14];
extern float l3_b2_add_out[1][256][14][14];

/* Layer 4: projection + identity, 512 channels, 7x7 */
extern float l4_b1_w1[512][256][3][3];
extern float l4_b1_b1[512];
extern float l4_b1_w2[512][512][3][3];
extern float l4_b1_b2[512];
extern float l4_b1_bn1_mean[512];
extern float l4_b1_bn1_var[512];
extern float l4_b1_bn1_gamma[512];
extern float l4_b1_bn1_beta[512];
extern float l4_b1_bn2_mean[512];
extern float l4_b1_bn2_var[512];
extern float l4_b1_bn2_gamma[512];
extern float l4_b1_bn2_beta[512];
extern float l4_b1_shortcut_w[512][256][1][1];
extern float l4_b1_shortcut_b[512];
extern float l4_b1_shortcut_mean[512];
extern float l4_b1_shortcut_var[512];
extern float l4_b1_shortcut_gamma[512];
extern float l4_b1_shortcut_beta[512];
extern float l4_b1_shortcut_out[1][512][7][7];
extern float l4_b1_conv1_out[1][512][7][7];
extern float l4_b1_bn1_out[1][512][7][7];
extern float l4_b1_relu1_out[1][512][7][7];
extern float l4_b1_conv2_out[1][512][7][7];
extern float l4_b1_bn2_out[1][512][7][7];
extern float l4_b1_add_out[1][512][7][7];

extern float l4_b2_w1[512][512][3][3];
extern float l4_b2_b1[512];
extern float l4_b2_w2[512][512][3][3];
extern float l4_b2_b2[512];
extern float l4_b2_bn1_mean[512];
extern float l4_b2_bn1_var[512];
extern float l4_b2_bn1_gamma[512];
extern float l4_b2_bn1_beta[512];
extern float l4_b2_bn2_mean[512];
extern float l4_b2_bn2_var[512];
extern float l4_b2_bn2_gamma[512];
extern float l4_b2_bn2_beta[512];
extern float l4_b2_conv1_out[1][512][7][7];
extern float l4_b2_bn1_out[1][512][7][7];
extern float l4_b2_relu1_out[1][512][7][7];
extern float l4_b2_conv2_out[1][512][7][7];
extern float l4_b2_bn2_out[1][512][7][7];
extern float l4_b2_add_out[1][512][7][7];

/* Classifier */
extern float avgpool_out[1][512];
extern float fc_w[1000][512];
extern float fc_b[1000];

#endif
