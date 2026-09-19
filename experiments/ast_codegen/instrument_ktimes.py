#!/usr/bin/env python3
"""Instrument a generated (naive) CUDA body with a per-kernel timed variant.

Usage: instrument_ktimes.py <body.cu> <out_body.cu> <host_fn_name>

Copies the body and appends a `<host_fn_name>_kt()` function that replays
the same launch chain with CUDA events around each kernel, printing
`KTIME <i> <kernel> <ms>` lines after the chain completes.  Feed the output
to wrap_generated_cuda.py with the SAME network name; the wrapper detects
the _kt function and calls it after its own TIME measurement.

Used to get a per-kernel breakdown of the M10 naive baselines without
touching the M10 printers.
"""

import re
import sys
from pathlib import Path


def main():
    body_path, out_path, host = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    body = body_path.read_text()
    m = re.search(rf"void {host}\(\)\n{{\n(.*?)\n}}", body, re.S)
    if not m:
        print(f"host function {host}() not found in {body_path}")
        sys.exit(1)
    fn_body = m.group(1)
    launches = re.findall(r"^  (\w+)<<<(.*?)>>>\(\);$", fn_body, re.M)
    if not launches:
        print("no launches found")
        sys.exit(1)
    lines = []
    lines.append(f"void {host}_kt()")
    lines.append("{")
    lines.append("  cudaEvent_t eva, evb;")
    lines.append("  cudaEventCreate(&eva);")
    lines.append("  cudaEventCreate(&evb);")
    lines.append(f"  float kt[{len(launches)}];")
    for i, (kname, cfg) in enumerate(launches):
        lines.append("  cudaEventRecord(eva);")
        lines.append(f"  {kname}<<<{cfg}>>>();")
        lines.append("  cudaEventRecord(evb);")
        lines.append("  cudaEventSynchronize(evb);")
        lines.append(f"  cudaEventElapsedTime(&kt[{i}], eva, evb);")
    for i, (kname, _) in enumerate(launches):
        lines.append(f'  printf("KTIME {i} {kname} %.6f\\n", kt[{i}]);')
    lines.append("  cudaEventDestroy(eva);")
    lines.append("  cudaEventDestroy(evb);")
    lines.append("}")
    out_path.write_text(body + "\n" + "\n".join(lines) + "\n")
    print(f"Instrumented body written to {out_path} ({len(launches)} kernels)")


if __name__ == "__main__":
    main()
