#ifndef MINITRANSFORMER_FAITHFUL_EXTERNS_H
#define MINITRANSFORMER_FAITHFUL_EXTERNS_H

/* External arrays referenced by minitransformer_faithful_comm
   (b=1, s=8, h=2, d=16, dm=32, df=32, v=8 mini GPT-style decoder,
   faithful AST: no score buffer, GELU coefficient read from gelu_c). */

/* GELU sqrt(2/pi) coefficient (route A reads it from the environment). */
extern float gelu_c;

/* Embedding tables. */
extern float tok_tab[8][32];
extern float pos_tab[8][32];

/* Layer-1 parameters. */
extern float l1_ln1_g[32];
extern float l1_ln1_b[32];
extern float l1_ln2_g[32];
extern float l1_ln2_b[32];
extern float l1_wq[32][32];
extern float l1_wk[32][32];
extern float l1_wv[32][32];
extern float l1_wo[32][32];
extern float l1_w1[32][32];
extern float l1_w2[32][32];

/* Layer-2 parameters. */
extern float l2_ln1_g[32];
extern float l2_ln1_b[32];
extern float l2_ln2_g[32];
extern float l2_ln2_b[32];
extern float l2_wq[32][32];
extern float l2_wk[32][32];
extern float l2_wv[32][32];
extern float l2_wo[32][32];
extern float l2_w1[32][32];
extern float l2_w2[32][32];

/* Final LayerNorm and output head. */
extern float lnf_g[32];
extern float lnf_b[32];
extern float whead[32][8];

/* Network-level buffers. */
extern float l1[1][8][32];
extern float l2[1][8][32];
extern float l3[1][8][32];
extern float l4[1][8][32];

/* Block-level buffers (shared by both layers, cf. route A). */
extern float bln1[1][8][32];
extern float bmha[1][8][32];
extern float br1[1][8][32];
extern float bln2[1][8][32];
extern float btf1[1][8][32];
extern float btg[1][8][32];
extern float btf2[1][8][32];

/* MHA buffers (no mtscore: the masked score is fused into mtprob). */
extern float mtq[1][8][32];
extern float mtk[1][8][32];
extern float mtv[1][8][32];
extern float mtmg[1][8][32];
extern float mtqs[1][2][8][16];
extern float mtks[1][2][8][16];
extern float mtvs[1][2][8][16];
extern float mtatt[1][2][8][16];
extern float mtprob[1][2][8][8];

extern float output[1][8][8];

#endif
