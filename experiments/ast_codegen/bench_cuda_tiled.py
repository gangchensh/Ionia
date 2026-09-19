#!/usr/bin/env python3
"""Benchmark a generated CUDA executable (G1 tiled and/or M10 baseline).

Usage: bench_cuda_tiled.py <exe_name> [N]

Runs ./<exe_name> N times (fresh process each), parses:
  TIME <exe> <ms>        -- kernel chain (CUDA events, wrapper-timed)
  MEMTIME_H2D / MEMTIME_D2H
  KTIME <i> <kernel> <ms> -- per-kernel CUDA-event timing from the
                             instrumented <net>_kt() pass (if present)
Reports min/median/max of TIME and the KTIME breakdown of the run with
minimal TIME (per-kernel min across runs is also reported).
"""

import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    exe = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    times, h2d, d2h = [], [], []
    kt_runs = []  # list of {idx: (name, ms)}
    for _ in range(n):
        r = subprocess.run([f"./{exe}"], cwd=ROOT, check=True,
                           capture_output=True, text=True)
        out = r.stdout
        times.append(float(re.search(rf"^TIME {exe} (\S+)$", out, re.M).group(1)))
        h2d.append(float(re.search(rf"^MEMTIME_H2D {exe} (\S+)$", out, re.M).group(1)))
        d2h.append(float(re.search(rf"^MEMTIME_D2H {exe} (\S+)$", out, re.M).group(1)))
        kt = {}
        for m in re.finditer(r"^KTIME (\d+) (\S+) ([0-9.]+)$", out, re.M):
            kt[int(m.group(1))] = (m.group(2), float(m.group(3)))
        if kt:
            kt_runs.append(kt)
    for label, xs in [(f"TIME {exe}", times),
                      ("MEMTIME_H2D", h2d),
                      ("MEMTIME_D2H", d2h)]:
        print(f"{label:24s} min={min(xs):.3f} ms  median={statistics.median(xs):.3f} ms  "
              f"max={max(xs):.3f} ms  n={len(xs)}")
    print("raw TIME:", " ".join(f"{x:.3f}" for x in times))
    if kt_runs:
        best = kt_runs[times.index(min(times))][:0] if False else kt_runs[times.index(min(times))]
        total = sum(v[1] for v in best.values())
        print(f"KTIME breakdown (run with min TIME; sum={total:.3f} ms):")
        for i in sorted(best):
            print(f"  KTIME {i:3d} {best[i][0]:40s} {best[i][1]:.4f}")
        # per-kernel min across runs
        kmin = {}
        for kt in kt_runs:
            for i, (nm, ms) in kt.items():
                if i not in kmin or ms < kmin[i][1]:
                    kmin[i] = (nm, ms)
        tmin = sum(v[1] for v in kmin.values())
        print(f"KTIME per-kernel min across {len(kt_runs)} runs (sum={tmin:.3f} ms):")
        for i in sorted(kmin):
            print(f"  KMIN  {i:3d} {kmin[i][0]:40s} {kmin[i][1]:.4f}")


if __name__ == "__main__":
    main()
