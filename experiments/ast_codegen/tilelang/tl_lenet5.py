# TileLang LeNet-5 end-to-end (shapes from validate_lenet5.py, seed 7, scale 1e-3):
# conv1 6@28^2 k5 -> relu -> pool -> conv2 16@10^2 k5 -> relu -> pool -> fc 400->120 -> relu
# -> fc 120->84 -> relu -> fc 84->10 -> softmax.  One T.Kernel per op, tiny-grid style
# (idx = bx*threads + i with bounds predication, local accumulation via T.alloc_local).
import os
os.environ.setdefault("CUDA_HOME", os.path.expanduser("~/torch_env/lib/python3.12/site-packages/nvidia/cu13"))  # pip-assembled CUDA 13.3 toolkit (see README)
import numpy as np
import torch
import tilelang
import tilelang.language as T


def conv_kernel(CI, CO, H, W, K):  # no pad, stride 1; input (CI,H+K-1,W+K-1)
    HI, WI = H + K - 1, W + K - 1
    N_OUT = CO * H * W

    @T.prim_func
    def main(
        A: T.Tensor((CI, HI, WI), "float32"),
        Wt: T.Tensor((CO, CI, K, K), "float32"),
        Bias: T.Tensor((CO,), "float32"),
        Out: T.Tensor((CO, H, W), "float32"),
    ):
        with T.Kernel(T.ceildiv(N_OUT, 128), threads=128) as bx:
            for i in T.Parallel(128):
                idx = bx * 128 + i
                if idx < N_OUT:
                    co = idx // (H * W)
                    hw = idx % (H * W)
                    h = hw // W
                    w = hw % W
                    acc = T.alloc_local([1], "float32")
                    acc[0] = Bias[co]
                    for ci in range(CI):
                        for kh in range(K):
                            for kw in range(K):
                                acc[0] += A[ci, h + kh, w + kw] * Wt[co, ci, kh, kw]
                    Out[co, h, w] = acc[0]

    return main


def relu_kernel(N_OUT, C, H, W):
    @T.prim_func
    def main(X: T.Tensor((C, H, W), "float32"), Y: T.Tensor((C, H, W), "float32")):
        with T.Kernel(T.ceildiv(N_OUT, 128), threads=128) as bx:
            for i in T.Parallel(128):
                idx = bx * 128 + i
                if idx < N_OUT:
                    c = idx // (H * W)
                    hw = idx % (H * W)
                    Y[c, hw // W, hw % W] = T.max(X[c, hw // W, hw % W], 0.0)

    return main


def pool_kernel(C, H, W):  # k2 s2, input (C, 2H, 2W)
    N_OUT = C * H * W

    @T.prim_func
    def main(X: T.Tensor((C, 2 * H, 2 * W), "float32"), Y: T.Tensor((C, H, W), "float32")):
        with T.Kernel(T.ceildiv(N_OUT, 128), threads=128) as bx:
            for i in T.Parallel(128):
                idx = bx * 128 + i
                if idx < N_OUT:
                    c = idx // (H * W)
                    hw = idx % (H * W)
                    h = hw // W
                    w = hw % W
                    m = T.alloc_local([1], "float32")
                    m[0] = X[c, 2 * h, 2 * w]
                    m[0] = T.max(m[0], X[c, 2 * h, 2 * w + 1])
                    m[0] = T.max(m[0], X[c, 2 * h + 1, 2 * w])
                    m[0] = T.max(m[0], X[c, 2 * h + 1, 2 * w + 1])
                    Y[c, h, w] = m[0]

    return main


def linear_kernel(IN, OUT):
    @T.prim_func
    def main(
        X: T.Tensor((IN,), "float32"),
        Wt: T.Tensor((OUT, IN), "float32"),
        Bias: T.Tensor((OUT,), "float32"),
        Y: T.Tensor((OUT,), "float32"),
    ):
        with T.Kernel(T.ceildiv(OUT, 128), threads=128) as bx:
            for i in T.Parallel(128):
                idx = bx * 128 + i
                if idx < OUT:
                    acc = T.alloc_local([1], "float32")
                    acc[0] = Bias[idx]
                    for k in range(IN):
                        acc[0] += X[k] * Wt[idx, k]
                    Y[idx] = acc[0]

    return main


def softmax_kernel(N):
    @T.prim_func
    def main(X: T.Tensor((N,), "float32"), Y: T.Tensor((N,), "float32")):
        with T.Kernel(1, threads=1) as bx:
            s = T.alloc_local([1], "float32")
            s[0] = 0.0
            for i in range(N):
                s[0] += T.exp(X[i])
            for i in range(N):
                Y[i] = T.exp(X[i]) / s[0]

    return main


def main():
    np.random.seed(7)
    scale = 1e-3
    d = lambda a: torch.tensor(a, device="cuda")
    input_ = (scale * np.random.randn(1, 32, 32)).astype(np.float32)
    kc1 = (scale * np.random.randn(6, 1, 5, 5)).astype(np.float32)
    bc1 = (scale * np.random.randn(6)).astype(np.float32)
    kc2 = (scale * np.random.randn(16, 6, 5, 5)).astype(np.float32)
    bc2 = (scale * np.random.randn(16)).astype(np.float32)
    fc1_w = (scale * np.random.randn(120, 400)).astype(np.float32)
    fc1_b = (scale * np.random.randn(120)).astype(np.float32)
    fc2_w = (scale * np.random.randn(84, 120)).astype(np.float32)
    fc2_b = (scale * np.random.randn(84)).astype(np.float32)
    fc3_w = (scale * np.random.randn(10, 84)).astype(np.float32)
    fc3_b = (scale * np.random.randn(10)).astype(np.float32)

    k_conv1 = tilelang.compile(conv_kernel(1, 6, 28, 28, 5), out_idx=[3], target="cuda")
    k_relu1 = tilelang.compile(relu_kernel(6 * 28 * 28, 6, 28, 28), out_idx=[1], target="cuda")
    k_pool1 = tilelang.compile(pool_kernel(6, 14, 14), out_idx=[1], target="cuda")
    k_conv2 = tilelang.compile(conv_kernel(6, 16, 10, 10, 5), out_idx=[3], target="cuda")
    k_relu2 = tilelang.compile(relu_kernel(16 * 10 * 10, 16, 10, 10), out_idx=[1], target="cuda")
    k_pool2 = tilelang.compile(pool_kernel(16, 5, 5), out_idx=[1], target="cuda")
    k_fc1 = tilelang.compile(linear_kernel(400, 120), out_idx=[3], target="cuda")
    k_fc2 = tilelang.compile(linear_kernel(120, 84), out_idx=[3], target="cuda")
    k_fc3 = tilelang.compile(linear_kernel(84, 10), out_idx=[3], target="cuda")
    k_sm = tilelang.compile(softmax_kernel(10), out_idx=[1], target="cuda")

    x0 = d(input_.reshape(1, 32, 32))
    w1, b1 = d(kc1), d(bc1)
    w2, b2 = d(kc2), d(bc2)
    w3, b3 = d(fc1_w), d(fc1_b)
    w4, b4 = d(fc2_w), d(fc2_b)
    w5, b5 = d(fc3_w), d(fc3_b)

    # compile small relu kernels for 120 and 84
    k_relu120 = tilelang.compile(relu_kernel(120, 120, 1, 1), out_idx=[1], target="cuda")
    k_relu84 = tilelang.compile(relu_kernel(84, 84, 1, 1), out_idx=[1], target="cuda")

    def forward():
        x1 = k_pool1(k_relu1(k_conv1(x0, w1, b1)))
        x2 = k_pool2(k_relu2(k_conv2(x1, w2, b2)))
        x3 = k_relu120(k_fc1(x2.reshape(-1), w3, b3).reshape(120, 1, 1))
        x4 = k_relu84(k_fc2(x3.reshape(-1), w4, b4).reshape(84, 1, 1))
        x5 = k_fc3(x4.reshape(-1), w5, b5)
        return k_sm(x5)

    out = forward()

    # numpy reference (same as validate_lenet5.py)
    def relu(x):
        return np.maximum(x, 0)

    def conv(x, w, b):
        N, Cin, H, W = x.shape
        Cout, _, K, _ = w.shape
        out = np.zeros((N, Cout, H - K + 1, W - K + 1), dtype=np.float32)
        for n in range(N):
            for co in range(Cout):
                for i in range(out.shape[2]):
                    for j in range(out.shape[3]):
                        acc = b[co]
                        for ci in range(Cin):
                            for ki in range(K):
                                for kj in range(K):
                                    acc += x[n, ci, i + ki, j + kj] * w[co, ci, ki, kj]
                        out[n, co, i, j] = acc
        return out

    def pool(x):
        N, C, H, W = x.shape
        out = np.zeros((N, C, H // 2, W // 2), dtype=np.float32)
        for n in range(N):
            for c in range(C):
                for i in range(H // 2):
                    for j in range(W // 2):
                        out[n, c, i, j] = max(x[n, c, 2*i, 2*j], x[n, c, 2*i, 2*j+1],
                                              x[n, c, 2*i+1, 2*j], x[n, c, 2*i+1, 2*j+1])
        return out

    xr = input_.reshape(1, 1, 32, 32)
    xr = pool(relu(conv(xr, kc1, bc1)))
    xr = pool(relu(conv(xr, kc2, bc2)))
    xf = xr.reshape(-1)
    xf = relu(fc1_w @ xf + fc1_b)
    xf = relu(fc2_w @ xf + fc2_b)
    xf = fc3_w @ xf + fc3_b
    ref = np.exp(xf) / np.exp(xf).sum()

    err = np.abs(out.cpu().numpy() - ref).max()
    print("max abs err vs numpy ref:", err)

    # timing: whole chain
    for _ in range(20):
        forward()
    torch.cuda.synchronize()
    ts = []
    for _ in range(50):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); forward(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    print(f"TileLang LeNet-5 chain: min {ts[0]*1e3:.1f} us, median {ts[len(ts)//2]*1e3:.1f} us")


if __name__ == "__main__":
    main()
