# Triton same-shape baselines for the four-way comparison:
#  1) conv 128->128 @28x28 k3p1 (kernel style copied from generated resnet18_triton.py, BLOCK=128)
#  2) gemv K=25088 N=4096 (generated-generator style: per-output program, sequential K)
#  3) matmul 4096^3 tl.dot, input_precision ieee vs tf32
import torch, triton, triton.language as tl
import torch.nn.functional as F


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


@triton.jit
def conv128_kernel(x_ptr, w_ptr, b_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < 100352
    i3 = offs % 28
    t2 = offs // 28
    i2 = t2 % 28
    i1 = t2 // 28
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for ci in range(128):
        for di in range(3):
            for dj in range(3):
                ih = i2 + di - 1
                iw = i3 + dj - 1
                xm = mask & (ih >= 0) & (ih < 28) & (iw >= 0) & (iw < 28)
                xv = tl.load(x_ptr + (ci * 784 + (ih * 28 + iw)), mask=xm, other=0.0)
                wv = tl.load(w_ptr + (i1 * 1152 + (ci * 9 + (di * 3 + dj))), mask=mask, other=0.0)
                acc += xv * wv
    acc += tl.load(b_ptr + i1, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc, mask=mask)


@triton.jit
def gemv_kernel(x_ptr, w_ptr, out_ptr, K: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(K):
        xv = tl.load(x_ptr + k)
        wv = tl.load(w_ptr + k * N + offs, mask=mask, other=0.0)
        acc += xv * wv
    tl.store(out_ptr + offs, acc, mask=mask)


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  IEEE: tl.constexpr):
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
        if IEEE:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, b, acc, input_precision="tf32")
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


def main():
    torch.manual_seed(0)
    # 1) conv
    x = torch.randn(128, 28, 28, device="cuda")
    w = torch.randn(128, 128, 3, 3, device="cuda") * 0.05
    b = torch.randn(128, device="cuda") * 0.05
    torch.backends.cudnn.allow_tf32 = False
    ref = F.conv2d(x.unsqueeze(0), w, b, padding=1).squeeze(0)
    out = torch.empty_like(ref)
    grid = (triton.cdiv(128 * 28 * 28, 128),)
    conv128_kernel[grid](x, w, b, out, BLOCK=128)
    err = (out - ref).abs().max().item()
    mn, md = bench(lambda: conv128_kernel[grid](x, w, b, out, BLOCK=128))
    fl = 2 * 128 * 28 * 28 * 128 * 9
    print(f"triton conv128@28: min {mn*1e3:.1f} us, med {md*1e3:.1f} us, {fl/(mn*1e-3)/1e12:.2f} TFLOP/s, err {err:.2e}")

    # 2) gemv
    K, N = 25088, 4096
    xv = torch.randn(K, device="cuda")
    wv = torch.randn(K, N, device="cuda") * 0.01
    torch.backends.cuda.matmul.allow_tf32 = False
    refv = xv @ wv
    outv = torch.empty(N, device="cuda")
    g = (triton.cdiv(N, 256),)
    gemv_kernel[g](xv, wv, outv, K, N, BLOCK=256)
    errv = (outv - refv).abs().max().item()
    mn, md = bench(lambda: gemv_kernel[g](xv, wv, outv, K, N, BLOCK=256))
    print(f"triton gemv 25088x4096: min {mn*1e3:.1f} us, {(K*N*4)/(mn*1e-3)/1e9:.0f} GB/s, err {errv:.2e}")

    # 3) matmul 4096^3
    M = N2 = K2 = 4096
    a = torch.randn(M, K2, device="cuda")
    bb = torch.randn(K2, N2, device="cuda") * 0.01
    refm = a @ bb
    for ieee in (True, False):
        c = torch.empty(M, N2, device="cuda")
        BM, BN, BK = 128, 128, 32
        gm = (N2 // BN, M // BM)
        matmul_kernel[gm](a, bb, c, M, N2, K2, BM, BN, BK, ieee, num_warps=8, num_stages=3)
        errm = (c - refm).abs().max().item()
        mn, md = bench(lambda: matmul_kernel[gm](a, bb, c, M, N2, K2, BM, BN, BK, ieee,
                                                 num_warps=8, num_stages=3))
        tf = 2 * M * N2 * K2 / (mn * 1e-3) / 1e12
        print(f"triton matmul4096 {'ieee' if ieee else 'tf32'}: {mn:.3f} ms, {tf:.2f} TFLOP/s, err {errm:.2e}")


if __name__ == "__main__":
    main()
