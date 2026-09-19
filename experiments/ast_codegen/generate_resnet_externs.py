#!/usr/bin/env python3
"""Generate externs/resnet{50,101,152}_externs.h."""

from pathlib import Path

configs = {
    "50": [
        (1, 256, 64, 56, 3, 1),
        (2, 512, 128, 28, 4, 2),
        (3, 1024, 256, 14, 6, 2),
        (4, 2048, 512, 7, 3, 2),
    ],
    "101": [
        (1, 256, 64, 56, 3, 1),
        (2, 512, 128, 28, 4, 2),
        (3, 1024, 256, 14, 23, 2),
        (4, 2048, 512, 7, 3, 2),
    ],
    "152": [
        (1, 256, 64, 56, 3, 1),
        (2, 512, 128, 28, 8, 2),
        (3, 1024, 256, 14, 36, 2),
        (4, 2048, 512, 7, 3, 2),
    ],
}


def generate(net):
    layers = configs[net]
    out = Path(f"externs/resnet{net}_externs.h")
    lines = []
    lines.append(f"#ifndef RESNET{net}_EXTERNS_H")
    lines.append(f"#define RESNET{net}_EXTERNS_H")
    lines.append("")
    lines.append("/* Input / output */")
    lines.append("extern float input[1][3][224][224];")
    lines.append("extern float output[1][1000];")
    lines.append("")
    lines.append("/* Stem */")
    lines.append("extern float conv1_w[64][3][7][7];")
    lines.append("extern float conv1_b[64];")
    for s in ["mean", "var", "gamma", "beta"]:
        lines.append(f"extern float bn1_{s}[64];")
    lines.append("extern float bn_eps;")
    lines.append("extern float conv1_out[1][64][112][112];")
    lines.append("extern float bn1_out[1][64][112][112];")
    lines.append("extern float relu1_out[1][64][112][112];")
    lines.append("extern float pool_out[1][64][56][56];")
    lines.append("")

    prev_outch = 64
    for layer, outch, mid, m, blocks, first_stride in layers:
        lines.append(f"/* Layer {layer}: {outch} channels, {m}x{m} */")
        for b in range(1, blocks + 1):
            prefix = f"l{layer}_b{b}"
            proj = (b == 1)
            inch = prev_outch if proj else outch
            stride = first_stride if proj else 1
            lines.append(f"extern float {prefix}_w1[{mid}][{inch}][1][1];")
            lines.append(f"extern float {prefix}_b1[{mid}];")
            lines.append(f"extern float {prefix}_w2[{mid}][{mid}][3][3];")
            lines.append(f"extern float {prefix}_b2[{mid}];")
            lines.append(f"extern float {prefix}_w3[{outch}][{mid}][1][1];")
            lines.append(f"extern float {prefix}_b3[{outch}];")
            for s in ["mean", "var", "gamma", "beta"]:
                lines.append(f"extern float {prefix}_bn1_{s}[{mid}];")
                lines.append(f"extern float {prefix}_bn2_{s}[{mid}];")
                lines.append(f"extern float {prefix}_bn3_{s}[{outch}];")
            if stride == 2:
                conv1_m = m * 2
                lines.append(f"extern float {prefix}_conv1_out[1][{mid}][{conv1_m}][{conv1_m}];")
                lines.append(f"extern float {prefix}_bn1_out[1][{mid}][{conv1_m}][{conv1_m}];")
                lines.append(f"extern float {prefix}_relu1_out[1][{mid}][{conv1_m}][{conv1_m}];")
                lines.append(f"extern float {prefix}_conv2_out[1][{mid}][{m}][{m}];")
                lines.append(f"extern float {prefix}_bn2_out[1][{mid}][{m}][{m}];")
                lines.append(f"extern float {prefix}_relu2_out[1][{mid}][{m}][{m}];")
            else:
                lines.append(f"extern float {prefix}_conv1_out[1][{mid}][{m}][{m}];")
                lines.append(f"extern float {prefix}_bn1_out[1][{mid}][{m}][{m}];")
                lines.append(f"extern float {prefix}_relu1_out[1][{mid}][{m}][{m}];")
                lines.append(f"extern float {prefix}_conv2_out[1][{mid}][{m}][{m}];")
                lines.append(f"extern float {prefix}_bn2_out[1][{mid}][{m}][{m}];")
                lines.append(f"extern float {prefix}_relu2_out[1][{mid}][{m}][{m}];")
            lines.append(f"extern float {prefix}_conv3_out[1][{outch}][{m}][{m}];")
            lines.append(f"extern float {prefix}_bn3_out[1][{outch}][{m}][{m}];")
            lines.append(f"extern float {prefix}_add_out[1][{outch}][{m}][{m}];")
            if proj:
                lines.append(f"extern float {prefix}_shortcut_w[{outch}][{inch}][1][1];")
                lines.append(f"extern float {prefix}_shortcut_b[{outch}];")
                for s in ["mean", "var", "gamma", "beta"]:
                    lines.append(f"extern float {prefix}_shortcut_{s}[{outch}];")
                lines.append(f"extern float {prefix}_shortcut_out[1][{outch}][{m}][{m}];")
            lines.append("")
        prev_outch = outch

    lines.append("/* Classifier */")
    lines.append("extern float avgpool_out[1][2048];")
    lines.append("extern float fc_w[1000][2048];")
    lines.append("extern float fc_b[1000];")
    lines.append("")
    lines.append("#endif")
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    for net in ["50", "101", "152"]:
        generate(net)
