#!/usr/bin/env python3
"""Benchmark the generated ResNet-18 CUDA executable.

Runs ./resnet18_test N times (each run is a fresh process, including CUDA
context setup), parses the wrapper's instrumentation lines:

  MEMTIME_H2D resnet18_test <ms>   -- input staging (fread) + H2D copies
  TIME       resnet18_test <ms>    -- kernel chain (33 launches + syncs)
  MEMTIME_D2H resnet18_test <ms>   -- output D2H copy

and reports min/median/max for each.  Usage: bench_cuda_resnet18.py [N]
"""

import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    times, h2d, d2h = [], [], []
    for i in range(n):
        r = subprocess.run(["./resnet18_test"], cwd=ROOT, check=True,
                           capture_output=True, text=True)
        out = r.stdout
        times.append(float(re.search(r"^TIME resnet18_test (\S+)$", out, re.M).group(1)))
        h2d.append(float(re.search(r"^MEMTIME_H2D resnet18_test (\S+)$", out, re.M).group(1)))
        d2h.append(float(re.search(r"^MEMTIME_D2H resnet18_test (\S+)$", out, re.M).group(1)))
    for label, xs in [("TIME (kernel chain)", times),
                      ("MEMTIME_H2D (staging+H2D)", h2d),
                      ("MEMTIME_D2H", d2h)]:
        print(f"{label:28s} min={min(xs):.3f} ms  median={statistics.median(xs):.3f} ms  "
              f"max={max(xs):.3f} ms  n={len(xs)}")
    print("raw TIME:    ", " ".join(f"{x:.3f}" for x in times))
    print("raw H2D:     ", " ".join(f"{x:.3f}" for x in h2d))
    print("raw D2H:     ", " ".join(f"{x:.3f}" for x in d2h))


if __name__ == "__main__":
    main()
