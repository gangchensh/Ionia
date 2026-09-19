#!/usr/bin/env python3
"""Numerical validation + performance comparison of the MiniTransformer AST
code generation (milestone T5).

Compares the generated C (sequential and parfor/OpenMP variants) against a
NumPy reference implementing the same computation as
MiniTransformer_DPIA.MiniTransformer at b=1, s=8, h=2, d=16, dm=32, df=32,
v=8 with the baked token sequence [3,1,4,1,5,2,6,0], then times both.
"""

import os
import subprocess
import numpy as np

np.random.seed(7)

B, S, H, D, DM, DF, V = 1, 8, 2, 16, 32, 32, 8
EPS = 1e-5
NEGINF = -1e30
TOKS = [3, 1, 4, 1, 5, 2, 6, 0]

# Weights (same values are injected into the C driver below).
tok_tab = (0.5 * np.random.randn(V, DM)).astype(np.float64)
pos_tab = (0.5 * np.random.randn(S, DM)).astype(np.float64)

def ln_params():
    g = 1.0 + 0.1 * np.random.randn(DM)
    b = 0.1 * np.random.randn(DM)
    return g, b

params = {}
for layer in ('l1', 'l2'):
    params[f'{layer}_ln1_g'], params[f'{layer}_ln1_b'] = ln_params()
    params[f'{layer}_ln2_g'], params[f'{layer}_ln2_b'] = ln_params()
    for w in ('wq', 'wk', 'wv', 'wo', 'w1', 'w2'):
        params[f'{layer}_{w}'] = (0.1 * np.random.randn(DM, DM)).astype(np.float64)
params['lnf_g'], params['lnf_b'] = ln_params()
params['whead'] = (0.1 * np.random.randn(DM, V)).astype(np.float64)


def layernorm(x, g, b):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(var + EPS) + b


def gelu(x):
    t = np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))
    return 0.5 * x * (1.0 + t)


def bmm(x, w):
    return x @ w


def block(x, p):
    ln1 = layernorm(x, p['ln1_g'], p['ln1_b'])
    q = bmm(ln1, p['wq']).reshape(B, S, H, D).transpose(0, 2, 1, 3)
    k = bmm(ln1, p['wk']).reshape(B, S, H, D).transpose(0, 2, 1, 3)
    v = bmm(ln1, p['wv']).reshape(B, S, H, D).transpose(0, 2, 1, 3)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(16.0)
    mask = np.tril(np.ones((S, S), dtype=bool))
    scores = np.where(mask, scores, NEGINF)
    # softmax without max subtraction, matching the generated code
    e = np.exp(scores)
    prob = e / e.sum(axis=-1, keepdims=True)
    att = np.matmul(prob, v)                      # (B,H,S,D)
    merged = att.transpose(0, 2, 1, 3).reshape(B, S, DM)
    mha = bmm(merged, p['wo'])
    r1 = x + mha
    ln2 = layernorm(r1, p['ln2_g'], p['ln2_b'])
    f = gelu(bmm(ln2, p['w1']))
    return r1 + bmm(f, p['w2'])


def ref_forward():
    x = tok_tab[TOKS] + pos_tab                   # (S, DM)
    x = x.reshape(B, S, DM)
    l1p = {k[3:]: params[f'l1_{k[3:]}'] for k in params if k.startswith('l1_')}
    l2p = {k[3:]: params[f'l2_{k[3:]}'] for k in params if k.startswith('l2_')}
    x = block(x, l1p)
    x = block(x, l2p)
    x = layernorm(x, params['lnf_g'], params['lnf_b'])
    return bmm(x, params['whead'])                # (B, S, V)


ref_output = ref_forward()


def arr2c(a):
    return ', '.join(f'{v:.9g}' for v in np.asarray(a, dtype=np.float64).flatten())


def array_defs(init=True):
    """C definitions of every global array (with initializers if init)."""
    lines = []
    def emit(name, shape, arr=None):
        dims = ''.join(f'[{d}]' for d in shape)
        if init and arr is not None:
            lines.append(f'float {name}{dims} = {{{arr2c(arr)}}};')
        else:
            lines.append(f'float {name}{dims};')
    emit('tok_tab', (V, DM), tok_tab)
    emit('pos_tab', (S, DM), pos_tab)
    for layer in ('l1', 'l2'):
        for w in ('ln1_g', 'ln1_b', 'ln2_g', 'ln2_b'):
            emit(f'{layer}_{w}', (DM,), params[f'{layer}_{w}'])
        for w in ('wq', 'wk', 'wv', 'wo', 'w1', 'w2'):
            emit(f'{layer}_{w}', (DM, DM), params[f'{layer}_{w}'])
    emit('lnf_g', (DM,), params['lnf_g'])
    emit('lnf_b', (DM,), params['lnf_b'])
    emit('whead', (DM, V), params['whead'])
    for name in ('l1', 'l2', 'l3', 'l4', 'bln1', 'bmha', 'br1', 'bln2',
                 'btf1', 'btg', 'btf2', 'mtq', 'mtk', 'mtv', 'mtmg'):
        emit(name, (B, S, DM))
    for name in ('mtqs', 'mtks', 'mtvs', 'mtatt'):
        emit(name, (B, H, S, D))
    for name in ('mtscore', 'mtprob'):
        emit(name, (B, H, S, S))
    emit('output', (B, S, V))
    return '\n'.join(lines)


def build_and_run(wrapped, fname, flags, tag):
    c_code = f'''
#include <math.h>
#include <stdio.h>

{array_defs(init=True)}

#include "{wrapped}"

int main() {{
  {fname}();
  for (int s = 0; s < {S}; s++)
    for (int v = 0; v < {V}; v++) printf("%.9g\\n", output[0][s][v]);
  return 0;
}}
'''
    src = f'test_mt_{tag}.c'
    with open(src, 'w') as f:
        f.write(c_code)
    subprocess.run(['gcc', '-O2', '-std=gnu99', *flags, src, '-o',
                    f'test_mt_{tag}', '-lm'], check=True)
    result = subprocess.run([f'./test_mt_{tag}'], capture_output=True,
                            text=True, check=True)
    return np.array([float(x) for x in result.stdout.split()],
                    dtype=np.float64).reshape(B, S, V)


seq_out = build_and_run('minitransformer_ast_wrapped.c',
                        'minitransformer_ast', [], 'seq')
par_out = build_and_run('minitransformer_ast_par_wrapped.c',
                        'minitransformer_ast_par', ['-fopenmp'], 'par')

max_err_ref = float(np.max(np.abs(ref_output - par_out)))
max_err_seq = float(np.max(np.abs(seq_out - par_out)))
print("Reference output[0,0,:]: ", ref_output[0, 0, :])
print("Sequential C out[0,0,:]: ", seq_out[0, 0, :])
print(f"Max |seq - numpy_ref| = {float(np.max(np.abs(ref_output - seq_out))):.3e}")
print(f"Max |par - numpy_ref| = {max_err_ref:.3e}")
print(f"Max |par - seq|       = {max_err_seq:.3e}")
print("Match (numpy, 1e-5):", bool(np.allclose(ref_output, par_out,
                                               atol=1e-5, rtol=0)))
print("Match (seq, exact): ", np.array_equal(seq_out, par_out))

# ---- timing ----
timing_c = '''
#include <stdio.h>
#include <time.h>
%s
#include "%s"
static double now(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}
int main() {
  int reps = 2000;
  %s();  /* warmup */
  double best = 1e30;
  for (int t = 0; t < 3; t++) {
    double t0 = now();
    for (int r = 0; r < reps; r++) %s();
    double dt = (now() - t0) / reps;
    if (dt < best) best = dt;
  }
  printf("%%.3f\\n", best * 1e6);  /* microseconds per inference */
  return 0;
}
'''
with open('time_mt_seq.c', 'w') as f:
    f.write(timing_c % (array_defs(init=False),
                        'minitransformer_ast_wrapped.c',
                        'minitransformer_ast', 'minitransformer_ast'))
with open('time_mt_par.c', 'w') as f:
    f.write(timing_c % (array_defs(init=False),
                        'minitransformer_ast_par_wrapped.c',
                        'minitransformer_ast_par', 'minitransformer_ast_par'))
subprocess.run(['gcc', '-O2', '-std=gnu99', 'time_mt_seq.c', '-o',
                'time_mt_seq', '-lm'], check=True)
subprocess.run(['gcc', '-O2', '-std=gnu99', '-fopenmp', 'time_mt_par.c', '-o',
                'time_mt_par', '-lm'], check=True)
t_seq = float(subprocess.run(['./time_mt_seq'], capture_output=True,
                             text=True, check=True).stdout.strip())
print(f"Sequential:            {t_seq:.3f} us/inference")
for nthreads in (1, 2, 4, 8):
    env = dict(os.environ, OMP_NUM_THREADS=str(nthreads))
    t_par = float(subprocess.run(['./time_mt_par'], capture_output=True,
                                 text=True, check=True, env=env).stdout.strip())
    print(f"OpenMP ({nthreads} threads):    {t_par:.3f} us/inference"
          f"  (speedup {t_seq / t_par:.2f}x)")
