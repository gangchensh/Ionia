#!/usr/bin/env python3
"""Per-kernel profile of the generated Triton ResNet-18 chain.

Groups torch.profiler rows by generated kernel name prefix (k<idx>_) and
maps them back to the operator kind via the generation log.  Used for the
Triton-vs-CUDA gap analysis.
"""

import numpy as np
import torch

import resnet18_triton as G
from run_triton_resnet18 import g, fwd_params  # reuse staged tensors


def triton_forward():
    return G.forward(**{p: g[p] for p in fwd_params})


# warmup (JIT + cudnn)
for _ in range(10):
    triton_forward()
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        triton_forward()
    torch.cuda.synchronize()

rows = {}
for e in prof.key_averages():
    if e.self_device_time_total <= 0:
        continue
    name = e.key
    rows[name] = e.self_device_time_total / 10.0  # us per iter

total = sum(rows.values())
print(f"GPU busy total: {total:.1f} us/iter")
for name, us in sorted(rows.items(), key=lambda kv: -kv[1]):
    print(f"{us:9.1f} us  {name[:90]}")
