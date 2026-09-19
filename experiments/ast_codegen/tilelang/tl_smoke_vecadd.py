# Minimal TileLang vector-add kernel to verify sm_120 (Blackwell) compatibility.
import torch
import tilelang
import tilelang.language as T


def vecadd_kernel(N, block_M=256, dtype="float32"):
    num_blocks = (N + block_M - 1) // block_M

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(num_blocks, threads=block_M) as bx:
            for i in T.Parallel(block_M):
                idx = bx * block_M + i
                if idx < N:
                    C[idx] = A[idx] + B[idx]

    return main


N = 1 << 20
jit = tilelang.compile(vecadd_kernel(N), out_idx=[2], target="cuda")
a = torch.randn(N, device="cuda", dtype=torch.float32)
b = torch.randn(N, device="cuda", dtype=torch.float32)
c = jit(a, b)
err = (c - (a + b)).abs().max().item()
print("max err:", err)
print("OK" if err < 1e-6 else "FAIL")
