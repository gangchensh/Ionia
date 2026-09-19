#!/usr/bin/env python
"""Supporting experiment 2 (2026-08-28):
A) cuDNN kernel names of a single conv (ResNet18 stage2 representative
   shape 128->128@28^2 k3) with TF32 on/off, confirming the TF32 kernel is
   indeed a tensor-op implementation;
B) matmul micro-benchmark quantifying this machine's TF32 vs FP32 peak gap
   (quantifying the hardware generation gap).
"""
import torch
import torch.backends.cudnn as cudnn
from torch.profiler import profile, ProfilerActivity

DEV = "cuda"


def conv_kernel_names(tf32):
    cudnn.benchmark = True
    cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    conv = torch.nn.Conv2d(128, 128, 3, 1, 1, bias=False).eval().to(DEV)
    x = torch.randn(1, 128, 28, 28, device=DEV)
    with torch.inference_mode():
        for _ in range(30):
            conv(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                conv(x)
            torch.cuda.synchronize()
    print(f"--- conv 128->128@28^2 k3 tf32={tf32} kernels ---")
    for ev in prof.key_averages():
        if ev.self_device_time_total > 0:
            print(f"  {ev.key[:110]}  cnt={ev.count} "
                  f"self_gpu={ev.self_device_time_total/20:.1f}us/iter")


def matmul_peak(tf32, n=4096):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    cudnn.allow_tf32 = tf32
    a = torch.randn(n, n, device=DEV)
    b = torch.randn(n, n, device=DEV)
    with torch.inference_mode():
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        it = 20
        for _ in range(it):
            a @ b
        e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / it
    tf = 2.0 * n ** 3 / (ms * 1e-3) / 1e12
    print(f"matmul {n}^3 tf32={tf32}: {ms:.3f} ms  {tf:.2f} TFLOP/s")
    return tf


for tf32 in (True, False):
    conv_kernel_names(tf32)
for tf32 in (True, False):
    matmul_peak(tf32)
