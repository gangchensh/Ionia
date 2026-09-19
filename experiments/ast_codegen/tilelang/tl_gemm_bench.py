# TileLang GEMM benchmark:
#  A) fc GEMV: M=1, K=25088, N=4096 (VGG16 fc1 shape, memory-bound; our CUDA fc1 = 0.75 ms)
#  B) compute-bound GEMM 4096^3, T.gemm tensor-core path, numeric check reveals TF32 vs FP32.
import os, json
os.environ.setdefault("CUDA_HOME", os.path.expanduser("~/torch_env/lib/python3.12/site-packages/nvidia/cu13"))  # pip-assembled CUDA 13.3 toolkit (see README)
import torch
import tilelang
import tilelang.language as T


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


# ---------- A) GEMV  y[n] = sum_k x[k] * W[k, n] ----------
def gemv_kernel(K, N, BN, BK, threads):
    @T.prim_func
    def main(
        X: T.Tensor((K,), "float32"),
        W: T.Tensor((K, N), "float32"),
        Y: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(N // BN, threads=threads) as bx:
            acc = T.alloc_fragment((BN,), "float32")
            x_sh = T.alloc_shared((BK,), "float32")
            T.clear(acc)
            for k0 in T.Pipelined(K // BK, num_stages=2):
                T.copy(X[k0 * BK], x_sh)
                for n in T.Parallel(BN):
                    for kk in range(BK):
                        acc[n] += x_sh[kk] * W[k0 * BK + kk, bx * BN + n]
            T.copy(acc, Y[bx * BN])

    return main


def run_gemv():
    K, N = 25088, 4096
    x = torch.randn(K, device="cuda")
    w = torch.randn(K, N, device="cuda") * 0.01
    ref = x @ w
    for (BN, BK, threads) in [(128, 8, 128), (256, 4, 256), (128, 16, 128), (512, 4, 256)]:
        name = f"gemv_BN{BN}_BK{BK}_t{threads}"
        try:
            ker = tilelang.compile(gemv_kernel(K, N, BN, BK, threads), out_idx=[2], target="cuda")
            out = ker(x, w)
            err = (out - ref).abs().max().item()
            mn, md = bench(lambda: ker(x, w))
            gbps = (K * N * 4) / (mn * 1e-3) / 1e9
            print(f"{name}: min {mn*1e3:.1f} us, med {md*1e3:.1f} us, {gbps:.0f} GB/s, err {err:.2e}")
        except Exception as ex:
            print(f"{name}: FAILED {str(ex)[:150]}")
    # torch cublas
    torch.backends.cuda.matmul.allow_tf32 = False
    mn, md = bench(lambda: x @ w)
    print(f"torch gemv (cublas fp32): min {mn*1e3:.1f} us, {(K*N*4)/(mn*1e-3)/1e9:.0f} GB/s")


# ---------- B) GEMM 4096^3 ----------
def gemm_kernel(M, N, K, BM, BN, BK, threads, stages):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float32"),
        B: T.Tensor((K, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=threads) as (bx, by):
            A_sh = T.alloc_shared((BM, BK), "float32")
            B_sh = T.alloc_shared((BK, BN), "float32")
            C_lo = T.alloc_fragment((BM, BN), "float32")
            T.clear(C_lo)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=stages):
                T.copy(A[by * BM, k * BK], A_sh)
                T.copy(B[k * BK, bx * BN], B_sh)
                T.gemm(A_sh, B_sh, C_lo)
            T.copy(C_lo, C[by * BM, bx * BN])

    return main


def run_gemm():
    M = N = K = 4096
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda") * 0.01
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = a @ b
    for (BM, BN, BK, th, st) in [(128, 128, 32, 256, 2), (128, 64, 32, 128, 3), (64, 64, 32, 128, 3)]:
        name = f"gemm_BM{BM}_BN{BN}_BK{BK}_t{th}_s{st}"
        try:
            ker = tilelang.compile(gemm_kernel(M, N, K, BM, BN, BK, th, st), out_idx=[2], target="cuda")
            out = ker(a, b)
            err = (out - ref).abs().max().item()
            rel = ((out - ref).abs().max() / ref.abs().max()).item()
            mn, md = bench(lambda: ker(a, b))
            tf = 2 * M * N * K / (mn * 1e-3) / 1e12
            print(f"{name}: min {mn:.3f} ms, {tf:.2f} TFLOP/s, max_err {err:.2e} rel {rel:.2e}")
        except Exception as ex:
            print(f"{name}: FAILED {str(ex)[:150]}")
    for tf32 in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        mn, md = bench(lambda: a @ b)
        print(f"torch matmul tf32={tf32}: {mn:.3f} ms, {2*M*N*K/(mn*1e-3)/1e12:.2f} TFLOP/s")


if __name__ == "__main__":
    run_gemv()
    run_gemm()
