#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
E3: PyTorch performance baseline measurement script (2026-08-27).

Baselines for: Ionia's generated naive CUDA kernel chain / G1 optimized
kernel chain.
Measurement protocol: batch=1, 224x224, FP32, eval mode, random weights
(fixed seed), CUDA events timing, warmup >= 20 iterations, 50 measured
iterations, median / min reported.
Two protocols: a) pure GPU compute (input already resident on GPU, matching
               our kernel-chain protocol that excludes transfers)
               b) including H2D input + D2H output (end-to-end, weights
                  resident on GPU)
Modes: eager and torch.compile (if available).
Also: torch.profiler per-layer breakdown of ResNet18 conv / BN / ReLU / Add
time shares.

Run: python3 experiments/ast_codegen/bench_torch_baseline.py
"""
import argparse
import copy
import json
import time

import torch
import torch.nn as nn
import torchvision.models as tvm

SEED = 20260827
WARMUP = 20
REPS = 50
BATCH = 1
HW = 224

MODELS = {
    "resnet18": tvm.resnet18,
    "vgg16": tvm.vgg16,
    "efficientnet_b0": tvm.efficientnet_b0,
    # torchvision has no MobileNetV1; use MobileNetV2 and note the difference
    "mobilenet_v2": tvm.mobilenet_v2,
}


def bench_gpu_only(fn, inp, warmup=WARMUP, reps=REPS):
    """Pure-GPU compute latency: CUDA events around one forward pass (input already on GPU, output stays on GPU)."""
    for _ in range(warmup):
        fn(inp)
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        start.record()
        fn(inp)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return {"min_ms": times[0], "median_ms": times[len(times) // 2],
            "p90_ms": times[int(len(times) * 0.9)], "max_ms": times[-1]}


def bench_e2e(fn, cpu_inp, warmup=WARMUP, reps=REPS):
    """End-to-end latency: perf_counter around H2D + forward + D2H + synchronize."""
    gpu_inp = cpu_inp.cuda()
    for _ in range(warmup):
        out = fn(gpu_inp)
        _ = out.cpu()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        gpu_inp = cpu_inp.cuda(non_blocking=False)
        out = fn(gpu_inp)
        _ = out.cpu()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {"min_ms": times[0], "median_ms": times[len(times) // 2]}


def profile_resnet18(model, inp):
    """torch.profiler per-layer breakdown of ResNet18: conv / BN / ReLU / Add / other."""
    from torch.profiler import profile, ProfilerActivity

    # grouping keywords (aten op names)
    groups = {
        "conv": ["convolution", "conv2d", "cudnn_convolution", "conv_"],
        "bn": ["batch_norm", "native_batch_norm"],
        "relu": ["relu", "threshold", "clamp_min"],
        "add": ["add", "add_"],
        "pool": ["max_pool", "adaptive_avg_pool", "avg_pool"],
        "fc": ["linear", "addmm", "mm"],
    }
    for _ in range(10):
        model(inp)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(REPS):
            model(inp)
        torch.cuda.synchronize()
    agg = {k: 0.0 for k in groups}
    agg["other"] = 0.0
    total = 0.0
    for ev in prof.key_averages():
        # only count aten:: op rows; raw CUDA kernel rows double-count the same time, skip
        if not ev.key.startswith("aten::"):
            continue
        t = ev.self_device_time_total / 1000.0 / REPS  # ms/iter, self GPU time
        if t <= 0:
            continue
        total += t
        name = ev.key.lower()
        placed = False
        for g, keys in groups.items():
            if any(k in name for k in keys):
                agg[g] += t
                placed = True
                break
        if not placed:
            agg["other"] += t
    return agg, total, prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--skip-profile", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True  # fixed 224² input; allow autotune (matches real inference deployment)

    dev = torch.device("cuda")
    print(f"# device: {torch.cuda.get_device_name(0)}")
    print(f"# torch {torch.__version__}, cudnn {torch.backends.cudnn.version()}")
    print(f"# batch={BATCH}, {HW}x{HW}, FP32, eval, random weights (seed={SEED})")
    print(f"# warmup={WARMUP}, reps={REPS}, CUDA events")

    results = {}
    for name, ctor in MODELS.items():
        torch.manual_seed(SEED)
        model = ctor(weights=None).eval().to(dev)
        inp = torch.randn(BATCH, 3, HW, HW, device=dev)
        cpu_inp = inp.cpu()
        with torch.inference_mode():
            eager_gpu = bench_gpu_only(model, inp)
            eager_e2e = bench_e2e(model, cpu_inp)
        entry = {"eager": {"gpu_only": eager_gpu, "e2e": eager_e2e}}
        print(f"\n## {name} eager:")
        print(f"  gpu_only  min={eager_gpu['min_ms']:.3f} ms  median={eager_gpu['median_ms']:.3f} ms"
              f"  p90={eager_gpu['p90_ms']:.3f}  max={eager_gpu['max_ms']:.3f}")
        print(f"  e2e(h2d+d2h) min={eager_e2e['min_ms']:.3f} ms  median={eager_e2e['median_ms']:.3f} ms")

        if not args.no_compile:
            try:
                cm = torch.compile(copy.deepcopy(model))
                with torch.inference_mode():
                    comp_gpu = bench_gpu_only(cm, inp)
                    comp_e2e = bench_e2e(cm, cpu_inp)
                entry["compile"] = {"gpu_only": comp_gpu, "e2e": comp_e2e}
                print(f"  compiled gpu_only  min={comp_gpu['min_ms']:.3f} ms  median={comp_gpu['median_ms']:.3f} ms")
                print(f"  compiled e2e       min={comp_e2e['min_ms']:.3f} ms  median={comp_e2e['median_ms']:.3f} ms")
            except Exception as e:
                entry["compile_error"] = repr(e)
                print(f"  torch.compile FAILED: {e!r}")
        results[name] = entry

        if name == "resnet18" and not args.skip_profile:
            with torch.inference_mode():
                agg, total, prof = profile_resnet18(model, inp)
            results["resnet18"]["layer_breakdown_ms"] = agg
            results["resnet18"]["layer_breakdown_total_ms"] = total
            print(f"\n## resnet18 per-layer breakdown (self GPU time / iter, eager): total={total:.3f} ms")
            for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
                print(f"  {k:6s}: {v:.3f} ms  ({100*v/total:.1f}%)")
            print("\n### top-15 aten ops (by self CUDA time):")
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

    out = "experiments/ast_codegen/bench_torch_baseline_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n# results saved to {out}")


if __name__ == "__main__":
    main()
