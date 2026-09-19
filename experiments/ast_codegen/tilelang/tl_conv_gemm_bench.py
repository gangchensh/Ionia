# TileLang im2col-free implicit-GEMM conv: C[co, hw] = sum_{ci,kh,kw} Wt[co, k] * Apad[ci, h+kh, w+kw]
# k = ci*9+kh*3+kw indexes the reduction; B_sh tile gathered in-kernel (no host im2col).
# Output written directly in NCHW-flat (CO, H*W). T.gemm on fp32 -> TF32 tensor cores on sm_120.
import os
os.environ.setdefault("CUDA_HOME", os.path.expanduser("~/torch_env/lib/python3.12/site-packages/nvidia/cu13"))  # pip-assembled CUDA 13.3 toolkit (see README)
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T

CI, CO, H, W, K = 128, 128, 28, 28, 3
HP = H + 2
KR = CI * 9          # reduction dim = 1152
HW = H * W           # 784
FLOPS = 2 * CO * HW * KR


def conv_gemm(BM, BN, BK, threads, stages):
    @T.prim_func
    def main(
        Wt: T.Tensor((CO, KR), "float32"),       # weights pre-flattened (co, ci*9+kh*3+kw)
        A: T.Tensor((CI, HP, HP), "float32"),    # padded input
        Bias: T.Tensor((CO,), "float32"),
        Out: T.Tensor((CO, HW), "float32"),
    ):
        with T.Kernel(T.ceildiv(HW, BN), T.ceildiv(CO, BM), threads=threads) as (bx, by):
            A_sh = T.alloc_shared((BM, BK), "float32")
            B_sh = T.alloc_shared((BK, BN), "float32")
            C_lo = T.alloc_fragment((BM, BN), "float32")
            T.clear(C_lo)
            for k0 in T.Pipelined(KR // BK, num_stages=stages):
                # A_sh: weight tile (BM, BK)
                T.copy(Wt[by * BM, k0 * BK], A_sh)
                # B_sh: gathered input tile (BK, BN) with implicit im2col
                for k, n in T.Parallel(BK, BN):
                    gk = k0 * BK + k
                    gn = bx * BN + n
                    ci = gk // 9
                    r = gk % 9
                    kh = r // 3
                    kw = r % 3
                    h = gn // W
                    w = gn % W
                    B_sh[k, n] = A[ci, h + kh, w + kw]
                T.gemm(A_sh, B_sh, C_lo)
            for i, j in T.Parallel(BM, BN):
                C_lo[i, j] += Bias[by * BM + i]
            T.copy(C_lo, Out[by * BM, bx * BN])

    return main


def bench(fn, warmup=20, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def main():
    torch.manual_seed(0)
    x = torch.randn(CI, H, W, device="cuda")
    w = torch.randn(CO, CI, K, K, device="cuda") * 0.05
    b = torch.randn(CO, device="cuda") * 0.05
    torch.backends.cudnn.allow_tf32 = False
    ref = F.conv2d(x.unsqueeze(0), w, b, padding=1).squeeze(0)
    xp = F.pad(x.unsqueeze(0), (1, 1, 1, 1)).squeeze(0).contiguous()
    wt = w.reshape(CO, KR).contiguous()

    for (BM, BN, BK, th, st) in [(64, 112, 32, 128, 2), (32, 112, 32, 128, 2),
                                 (64, 112, 32, 256, 2), (128, 112, 32, 256, 2)]:
        name = f"BM{BM}_BN{BN}_BK{BK}_t{th}_s{st}"
        try:
            ker = tilelang.compile(conv_gemm(BM, BN, BK, th, st), out_idx=[3], target="cuda")
            out = ker(wt, xp, b)
            err = (out.reshape(CO, H, W) - ref).abs().max().item()
            mn, md = bench(lambda: ker(wt, xp, b))
            print(f"{name}: min {mn*1e3:.1f} us, med {md*1e3:.1f} us, "
                  f"{FLOPS/(mn*1e-3)/1e12:.2f} TFLOP/s, max_err {err:.2e}")
        except Exception as ex:
            print(f"{name}: FAILED {str(ex).splitlines()[0][:150]}")


if __name__ == "__main__":
    main()
