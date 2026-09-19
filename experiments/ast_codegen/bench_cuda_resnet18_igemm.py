#!/usr/bin/env python3
"""Benchmark the round-4 implicit-GEMM ResNet-18 CUDA executable (and, for
same-session baseline numbers, the E4 fused binary).

Runs each binary N times (fresh process per run, CUDA context setup
excluded from TIME), parses the wrapper instrumentation line

  TIME <network> <ms>   -- whole kernel chain, CUDA events, no H2D/D2H

and reports min/median/max.  Per-kernel decomposition comes from the
_kt() variant's KMIN lines (printed once per run after TIME; min-of-30
per kernel inside the process).  Usage: bench_cuda_resnet18_igemm.py [N]
"""

import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

BINARIES = [
    ("resnet18_fused", "E4 fully fused"),  # round-3 baseline
    ("resnet18_igemm", "R4 implicit GEMM"),  # this work
]


def bench(binary, n):
    times = []
    medians = []
    gtimes = []
    gmedians = []
    kmin = {}
    for i in range(n):
        r = subprocess.run([f"./{binary}"], cwd=ROOT, check=True,
                           capture_output=True, text=True)
        out = r.stdout
        times.append(float(re.search(rf"^TIME {binary} (\S+)$", out, re.M).group(1)))
        m = re.search(rf"^TIMEMED {binary} (\S+)$", out, re.M)
        if m:
            medians.append(float(m.group(1)))
        g = re.search(rf"^TIME_GRAPH {binary} (\S+)$", out, re.M)
        if g and g.group(1) != "unsupported":
            gtimes.append(float(g.group(1)))
        gm = re.search(rf"^TIMEGRAPH_MED {binary} (\S+)$", out, re.M)
        if gm:
            gmedians.append(float(gm.group(1)))
        if i == 0:
            for m in re.finditer(r"^K(?:MIN|TIME)\s+\d+\s+(\S+)\s+(\S+)$", out, re.M):
                kmin[m.group(1)] = float(m.group(2))
    return times, medians, gtimes, gmedians, kmin


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    for binary, label in BINARIES:
        if not (ROOT / binary).exists():
            print(f"{label:16s} ({binary}): binary missing, skipped")
            continue
        times, medians, gtimes, gmedians, kmin = bench(binary, n)
        inner = (f"  inner_med={statistics.median(medians):.3f}"
                 if medians else "")
        print(f"{label:16s} ({binary}): min={min(times):.3f} ms  "
              f"median={statistics.median(times):.3f} ms  "
              f"max={max(times):.3f} ms  n={n}{inner}")
        if gtimes:
            print(f"    graph replay: min={min(gtimes):.3f} ms  "
                  f"median={statistics.median(gtimes):.3f} ms  "
                  f"inner_med={statistics.median(gmedians):.3f} ms")
        if kmin:
            print(f"    KMIN sum={sum(kmin.values()):.3f} ms  "
                  + "  ".join(f"{k}={v:.4f}" for k, v in kmin.items()))


if __name__ == "__main__":
    main()
