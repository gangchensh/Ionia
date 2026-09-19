# TileLang tiled conv2d benchmark: ResNet18 stage2 shape 128->128 @ 28x28, k3 s1 p1, batch=1, FP32.
# Multi-config sweep; correctness vs torch F.conv2d (TF32 off); timing: CUDA events, warmup 20 + 50 iters, min/median.
import os, sys, json
os.environ.setdefault("CUDA_HOME", os.path.expanduser("~/torch_env/lib/python3.12/site-packages/nvidia/cu13"))  # pip-assembled CUDA 13.3 toolkit (see README)
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

CI, CO, H, W, K, P = 128, 128, 28, 28, 3, 1
HP = H + 2 * P
FLOPS = 2 * CO * H * W * CI * K * K


def conv_kernel(CI, CO, HP, H, W, BC, BH, BW, CT, threads, stages):
    @T.prim_func
    def main(
        A: T.Tensor((CI, HP, HP), "float32"),
        Wt: T.Tensor((CO, CI, K, K), "float32"),
        Bias: T.Tensor((CO,), "float32"),
        Out: T.Tensor((CO, H, W), "float32"),
    ):
        with T.Kernel(CO // BC, H // BH, W // BW, threads=threads) as (bz, by, bx):
            A_sh = T.alloc_shared((CT, BH + 2, BW + 2), "float32")
            W_sh = T.alloc_shared((BC, CT, K, K), "float32")
            acc = T.alloc_fragment((BC, BH, BW), "float32")
            T.clear(acc)
            for p in T.Pipelined(CI // CT, num_stages=stages):
                T.copy(A[p * CT, by * BH, bx * BW], A_sh)
                T.copy(Wt[bz * BC, p * CT, 0, 0], W_sh)
                for co, h, w in T.Parallel(BC, BH, BW):
                    for ci in range(CT):
                        for kh in range(K):
                            for kw in range(K):
                                acc[co, h, w] += A_sh[ci, h + kh, w + kw] * W_sh[co, ci, kh, kw]
            for co, h, w in T.Parallel(BC, BH, BW):
                acc[co, h, w] += Bias[bz * BC + co]
            T.copy(acc, Out[bz * BC, by * BH, bx * BW])

    return main


def bench(fn, args, warmup=20, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(*args); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[0], times[len(times) // 2]


def main():
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(1, CI, H, W, device=dev)
    w = torch.randn(CO, CI, K, K, device=dev) * 0.05
    b = torch.randn(CO, device=dev) * 0.05
    xp = F.pad(x, (P, P, P, P)).squeeze(0).contiguous()  # (CI, HP, HP)
    ref = F.conv2d(x, w, b, padding=P).squeeze(0)

    configs = [
        # (BC, BH, BW, CT, threads, stages); BC*BH*BW must be divisible by threads
        (32, 14, 14, 16, 128, 1),
        (32, 14, 14, 32, 128, 1),
        (16, 14, 14, 16, 64, 1),
        (32, 7, 28, 16, 128, 1),
        (16, 14, 14, 16, 64, 2),
        (32, 14, 14, 16, 128, 2),
    ]
    results = []
    for cfg in configs:
        BC, BH, BW, CT, threads, stages = cfg
        name = f"BC{BC}_BH{BH}_BW{BW}_CT{CT}_t{threads}_s{stages}"
        try:
            ker = tilelang.compile(conv_kernel(CI, CO, HP, H, W, *cfg), out_idx=[3], target="cuda")
            out = ker(xp, w, b)
            err = (out - ref).abs().max().item()
            rel = ((out - ref).abs().max() / ref.abs().max()).item()
            mn, md = bench(ker, (xp, w, b))
            results.append(dict(cfg=name, min_us=mn * 1e3, med_us=md * 1e3,
                                tflops=FLOPS / (mn * 1e-3) / 1e12, max_err=err, rel=rel))
            print(f"{name}: min {mn*1e3:.1f} us, med {md*1e3:.1f} us, "
                  f"{FLOPS/(mn*1e-3)/1e12:.2f} TFLOP/s, max_err {err:.2e}")
        except Exception as ex:
            msg = str(ex).replace("\n", " ")[:200]
            results.append(dict(cfg=name, error=msg))
            print(f"{name}: FAILED {msg}")

    # PyTorch / cuDNN same-session, TF32 off and on
    torch.backends.cudnn.benchmark = True
    for tf32 in (False, True):
        torch.backends.cudnn.allow_tf32 = tf32
        mn, md = bench(lambda a, b2, c: F.conv2d(a, b2, c, padding=P), (x, w, b))
        print(f"cuDNN tf32={tf32}: min {mn*1e3:.1f} us, med {md*1e3:.1f} us, "
              f"{FLOPS/(mn*1e-3)/1e12:.2f} TFLOP/s")
        results.append(dict(cfg=f"cudnn_tf32_{tf32}", min_us=mn * 1e3, med_us=md * 1e3,
                            tflops=FLOPS / (mn * 1e-3) / 1e12))
    torch.backends.cudnn.allow_tf32 = False

    with open(os.path.join(os.path.dirname(__file__), "conv_results.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
