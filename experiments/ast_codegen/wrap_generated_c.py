#!/usr/bin/env python3
"""Wrap a generated network body with #include of extern declarations
and a void function definition, producing a compilable C translation unit."""

import sys
import os

if len(sys.argv) != 4:
    print("Usage: wrap_generated_c.py <network> <body.c> <out.c>")
    print("  network: any name with a matching externs/<network>_externs.h")
    print("           (shipped: alexnet, lenet5, resnet18, minitransformer variants, ...)")
    sys.exit(1)

network, body_path, out_path = sys.argv[1:4]

header = f"externs/{network}_externs.h"
if not os.path.exists(header):
    print(f"Unknown network '{network}' or missing {header}")
    sys.exit(1)

with open(body_path, 'r') as f:
    body = f.read()

with open(out_path, 'w') as f:
    f.write(f'#include <math.h>\n')
    f.write(f'#include "{header}"\n\n')
    f.write(f'void {network}()\n{{\n')
    f.write(body)
    f.write('}\n')

print(f"Wrote {out_path}")
