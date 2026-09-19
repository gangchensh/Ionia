#!/usr/bin/env python3
"""Generate externs/vgg16_externs.h declaring all parameters and buffers."""

from pathlib import Path

convs = [
    (1, 1, 3, 64, 224),
    (1, 2, 64, 64, 224),
    (2, 1, 64, 128, 112),
    (2, 2, 128, 128, 112),
    (3, 1, 128, 256, 56),
    (3, 2, 256, 256, 56),
    (3, 3, 256, 256, 56),
    (4, 1, 256, 512, 28),
    (4, 2, 512, 512, 28),
    (4, 3, 512, 512, 28),
    (5, 1, 512, 512, 14),
    (5, 2, 512, 512, 14),
    (5, 3, 512, 512, 14),
]

pools = [
    (1, 64, 112),
    (2, 128, 56),
    (3, 256, 28),
    (4, 512, 14),
    (5, 512, 7),
]

lines = []
lines.append("#ifndef VGG16_EXTERNS_H")
lines.append("#define VGG16_EXTERNS_H")
lines.append("")
lines.append("/* Input / output */")
lines.append("extern float input[1][3][224][224];")
lines.append("extern float output[1][1000];")
lines.append("")
lines.append("/* Convolution weights and biases */")
for block, layer, inch, outch, m in convs:
    lines.append(f"extern float conv{block}_{layer}_w[{outch}][{inch}][3][3];")
    lines.append(f"extern float conv{block}_{layer}_b[{outch}];")
lines.append("")
lines.append("/* Fully-connected weights and biases */")
lines.append("extern float fc1_w[4096][25088];")
lines.append("extern float fc1_b[4096];")
lines.append("extern float fc2_w[4096][4096];")
lines.append("extern float fc2_b[4096];")
lines.append("extern float fc3_w[1000][4096];")
lines.append("extern float fc3_b[1000];")
lines.append("")
lines.append("/* Convolution feature maps */")
for block, layer, inch, outch, m in convs:
    lines.append(f"extern float conv{block}_{layer}_out[1][{outch}][{m}][{m}];")
    lines.append(f"extern float relu{block}_{layer}_out[1][{outch}][{m}][{m}];")
lines.append("")
lines.append("/* Max-pooling feature maps */")
for block, ch, m in pools:
    lines.append(f"extern float pool{block}_out[1][{ch}][{m}][{m}];")
lines.append("")
lines.append("/* Classifier feature maps */")
lines.append("extern float flatten_out[1][25088];")
lines.append("extern float fc1_out[1][4096];")
lines.append("extern float fc1_relu_out[1][4096];")
lines.append("extern float fc2_out[1][4096];")
lines.append("extern float fc2_relu_out[1][4096];")
lines.append("")
lines.append("#endif /* VGG16_EXTERNS_H */")

out_path = Path(__file__).parent / "externs" / "vgg16_externs.h"
out_path.write_text("\n".join(lines))
print(f"Wrote {out_path}")
