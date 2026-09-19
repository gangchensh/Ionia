#!/usr/bin/env python3
"""Generate Coq ResNet-101/152 network definitions."""

from pathlib import Path

configs = {
    "101": {
        "layers": [
            (1, 256, 64, 56, 3, 1),
            (2, 512, 128, 28, 4, 2),
            (3, 1024, 256, 14, 23, 2),
            (4, 2048, 512, 7, 3, 2),
        ],
    },
    "152": {
        "layers": [
            (1, 256, 64, 56, 3, 1),
            (2, 512, 128, 28, 8, 2),
            (3, 1024, 256, 14, 36, 2),
            (4, 2048, 512, 7, 3, 2),
        ],
    },
}


def t4(batch, ch, h, w):
    return f"(ary '{{{batch}}} (ary '{{{ch}}} (ary '{{{h}}} (ary '{{{w}}} num))))"


def t1(ch):
    return f"(ary '{{{ch}}} num)"


def pdef(lines, prefix, name, typ):
    lines.append(f'Definition {prefix}_{name} : exp {typ} := vrd "{prefix}_{name}".')


def adef(lines, prefix, name, typ):
    lines.append(f'Definition {prefix}_{name} : acc {typ} := vwt "{prefix}_{name}".')


def generate(net):
    cfg = configs[net]
    name = f"resnet{net}"
    lines = []
    lines.append(f"(* ImageNet ResNet-{net} imperative AST. *)")
    lines.append("")
    lines.append("Require Import Ionia.CNN_Examples_Base.")
    lines.append("Require Import Logic.FunctionalExtensionality.")
    lines.append("")
    lines.append("From Stdlib Require Import List String.")
    lines.append("Import ListNotations.")
    lines.append("Open Scope string_scope.")
    lines.append("Set Implicit Arguments.")
    lines.append("Opaque CG_acc CG_exp.")
    lines.append("")
    lines.append(f'Definition {name}_input : exp {t4(1,3,224,224)} := vrd "input".')
    lines.append("")
    lines.append("(* -------------------- Stem -------------------- *)")
    pdef(lines, name, "conv1_w", t4(64,3,7,7))
    pdef(lines, name, "conv1_b", t1(64))
    for s in ["mean", "var", "gamma", "beta"]:
        pdef(lines, name, f"bn1_{s}", t1(64))
    pdef(lines, name, "eps", "num")
    adef(lines, name, "conv1_out", t4(1,64,112,112))
    adef(lines, name, "bn1_out", t4(1,64,112,112))
    adef(lines, name, "relu1_out", t4(1,64,112,112))
    adef(lines, name, "pool_out", t4(1,64,56,56))
    lines.append("")

    prev_outch = 64
    for layer, outch, mid, m, blocks, first_stride in cfg["layers"]:
        lines.append(f"(* -------------------- Layer {layer} : {outch} channels, {m}x{m} -------------------- *)")
        for b in range(1, blocks + 1):
            prefix = f"{name}_l{layer}_b{b}"
            proj = (b == 1)
            inch = prev_outch if proj else outch
            stride = first_stride if proj else 1
            conv1_m = m * 2 if stride == 2 else m
            pdef(lines, prefix, "w1", t4(mid, inch, 1, 1))
            pdef(lines, prefix, "b1", t1(mid))
            pdef(lines, prefix, "w2", t4(mid, mid, 3, 3))
            pdef(lines, prefix, "b2", t1(mid))
            pdef(lines, prefix, "w3", t4(outch, mid, 1, 1))
            pdef(lines, prefix, "b3", t1(outch))
            for s in ["mean", "var", "gamma", "beta"]:
                pdef(lines, prefix, f"bn1_{s}", t1(mid))
                pdef(lines, prefix, f"bn2_{s}", t1(mid))
                pdef(lines, prefix, f"bn3_{s}", t1(outch))
            adef(lines, prefix, "conv1_out", t4(1, mid, conv1_m, conv1_m))
            adef(lines, prefix, "bn1_out", t4(1, mid, conv1_m, conv1_m))
            adef(lines, prefix, "relu1_out", t4(1, mid, conv1_m, conv1_m))
            adef(lines, prefix, "conv2_out", t4(1, mid, m, m))
            adef(lines, prefix, "bn2_out", t4(1, mid, m, m))
            adef(lines, prefix, "relu2_out", t4(1, mid, m, m))
            adef(lines, prefix, "conv3_out", t4(1, outch, m, m))
            adef(lines, prefix, "bn3_out", t4(1, outch, m, m))
            adef(lines, prefix, "add_out", t4(1, outch, m, m))
            if proj:
                pdef(lines, prefix, "shortcut_w", t4(outch, inch, 1, 1))
                pdef(lines, prefix, "shortcut_b", t1(outch))
                for s in ["mean", "var", "gamma", "beta"]:
                    pdef(lines, prefix, f"shortcut_{s}", t1(outch))
                adef(lines, prefix, "shortcut_out", t4(1, outch, m, m))
            lines.append("")
        prev_outch = outch

    lines.append("(* -------------------- Classifier -------------------- *)")
    adef(lines, name, "avgpool_out", "(ary '{1} (ary '{2048} num))")
    pdef(lines, name, "fc_w", "(ary '{1000} (ary '{2048} num))")
    pdef(lines, name, "fc_b", t1(1000))
    adef(lines, name, "output", "(ary '{1} (ary '{1000} num))")
    lines.append("")

    lines.append(f"Definition {name}_comm : comm :=")
    lines.append(f"  @Conv4d_p3_k7_s2_comm 1 3 64 224 {name}_input {name}_conv1_w {name}_conv1_b {name}_conv1_out ';")
    lines.append(f"  @BatchNorm4d_comm 1 64 112 112")
    lines.append(f"    (vrd \"conv1_out\") {name}_bn1_mean {name}_bn1_var {name}_bn1_gamma {name}_bn1_beta {name}_eps {name}_bn1_out ';")
    lines.append(f"  @ReLU4d_comm 1 64 112 112")
    lines.append(f"    (vrd \"bn1_out\") {name}_relu1_out ';")
    lines.append(f"  @MaxPool4d_pad_comm 1 64 112 3 2 1")
    lines.append(f"    (vrd \"relu1_out\") {name}_pool_out ';")

    prev_outch = 64
    prev_add = "pool_out"
    for layer, outch, mid, m, blocks, first_stride in cfg["layers"]:
        for b in range(1, blocks + 1):
            prefix = f"{name}_l{layer}_b{b}"
            short_prefix = f"l{layer}_b{b}"
            proj = (b == 1)
            inch = prev_outch if proj else outch
            stride = first_stride if proj else 1
            conv1_m = m * 2 if stride == 2 else m
            if proj:
                comb = "ResNet50_Bottleneck_proj_s2_comm" if stride == 2 else "ResNet50_Bottleneck_proj_s1_comm"
                lines.append(f"  @{comb} 1 {inch} {outch} {mid} {conv1_m}")
            else:
                lines.append(f"  @ResNet50_Bottleneck_id_comm 1 {outch} {mid} {m}")
            lines.append(f"    (vrd \"{prev_add}\")")
            lines.append(f"    {prefix}_w1 {prefix}_b1")
            lines.append(f"    {prefix}_w2 {prefix}_b2")
            lines.append(f"    {prefix}_w3 {prefix}_b3")
            lines.append(f"    {prefix}_bn1_mean {prefix}_bn1_var {prefix}_bn1_gamma {prefix}_bn1_beta")
            lines.append(f"    {prefix}_bn2_mean {prefix}_bn2_var {prefix}_bn2_gamma {prefix}_bn2_beta")
            lines.append(f"    {prefix}_bn3_mean {prefix}_bn3_var {prefix}_bn3_gamma {prefix}_bn3_beta")
            lines.append(f"    {name}_eps")
            lines.append(f"    {prefix}_conv1_out {prefix}_bn1_out {prefix}_relu1_out")
            lines.append(f"    {prefix}_conv2_out {prefix}_bn2_out {prefix}_relu2_out")
            lines.append(f"    {prefix}_conv3_out {prefix}_bn3_out {prefix}_add_out")
            if proj:
                lines.append(f"    {prefix}_shortcut_out")
                lines.append(f"    {prefix}_shortcut_w {prefix}_shortcut_b")
                lines.append(f"    {prefix}_shortcut_mean {prefix}_shortcut_var {prefix}_shortcut_gamma {prefix}_shortcut_beta")
                lines.append(f"    \"{short_prefix}_shortcut_out\"")
            strings = [f"{short_prefix}_conv1_out", f"{short_prefix}_bn1_out", f"{short_prefix}_relu1_out",
                       f"{short_prefix}_conv2_out", f"{short_prefix}_bn2_out", f"{short_prefix}_relu2_out",
                       f"{short_prefix}_conv3_out", f"{short_prefix}_bn3_out", f"{short_prefix}_add_out"]
            quoted = ' '.join(f'"{s}"' for s in strings)
            lines.append(f"    {quoted} ';")
            prev_add = f"{short_prefix}_add_out"
        prev_outch = outch

    lines.append(f"  @GlobalAvgPool4d_comm 1 2048 7 7")
    lines.append(f"    (vrd \"{prev_add}\") {name}_avgpool_out ';")
    lines.append(f"  @Linear2d_comm 1 2048 1000")
    lines.append(f"    {name}_fc_w {name}_fc_b (vrd \"avgpool_out\") {name}_output.")
    lines.append("")

    # In the development repository this file lands in src/Ionia/; when run
    # standalone it is written to the current directory.
    out_path = Path(f"CNN_Examples_ResNet{net}.v")
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    for net in ["101", "152"]:
        generate(net)
