#!/usr/bin/env python3
"""cuDNN comparison for the TF32 pilot: same shapes AND same data bins as
tf32_conv_pilot.cu / validate_tf32_pilot.py.

For each shape: cuDNN TF32=on conv timing (warmup 20 + min-of-50 CUDA
events) and output error vs the float64 reference (same reference math as
validate_tf32_pilot.py), so our kernel and cuDNN errors are comparable.
"""
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

ROOT = Path(__file__).parent
DIR = ROOT / "tf32_pilot"
SHAPES = [
    ("l1_64ch_56x56", 64, 64, 56),
    ("l2_128ch_28x28", 128, 128, 28),
    ("l4_512ch_7x7", 512, 512, 7),
]

cudnn.benchmark = True
cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

for name, ci, co, hw in SHAPES:
    x = torch.from_numpy(
        np.fromfile(DIR / f"{name}_input.bin", np.float32).reshape(1, ci, hw, hw)
    ).cuda()
    w = torch.from_numpy(
        np.fromfile(DIR / f"{name}_weight.bin", np.float32).reshape(co, ci, 3, 3)
    ).cuda()
    with torch.inference_mode():
        y = torch.nn.functional.conv2d(x, w, padding=1)
        for _ in range(20):
            torch.nn.functional.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        best = 1e30
        for _ in range(50):
            s.record()
            y = torch.nn.functional.conv2d(x, w, padding=1)
            e.record()
            torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e))
    # float64 reference
    xd = x.double().cpu()
    wd = w.double().cpu()
    ref = torch.nn.functional.conv2d(xd, wd, padding=1).numpy()
    err = np.abs(y.double().cpu().numpy() - ref)
    mflop = 2.0 * co * ci * 9 * hw * hw / 1e6
    tf = mflop / (best * 1e3)
    print(
        f"{name}: cudnn_tf32 {best*1e3:8.1f} us  {tf:6.2f} TFLOP/s  "
        f"maxabs={err.max():.3e} "
        f"normrel={np.linalg.norm(err.ravel())/np.linalg.norm(ref.ravel()):.3e}"
    )
