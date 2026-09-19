#!/usr/bin/env python3
"""Generate Coq VGG-16 network definition."""

from pathlib import Path


def t4(batch, ch, h, w):
    return f"(ary '{{{batch}}} (ary '{{{ch}}} (ary '{{{h}}} (ary '{{{w}}} num))))"


def t1(ch):
    return f"(ary '{{{ch}}} num)"


def t2(batch, m):
    return f"(ary '{{{batch}}} (ary '{{{m}}} num))"


def pdef(lines, defname, stringname, typ):
    lines.append(f'Definition {defname} : exp {typ} := vrd "{stringname}".')


def adef(lines, defname, stringname, typ):
    lines.append(f'Definition {defname} : acc {typ} := vwt "{stringname}".')


def generate():
    name = "vgg16"
    lines = []
    lines.append("(* ImageNet VGG-16 imperative AST. *)")
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

    # VGG-16 conv configuration: (block, layer_in_block, inch, outch, spatial_m)
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

    for block, layer, inch, outch, m in convs:
        cname = f"conv{block}_{layer}"
        dname = f"{name}_{cname}"
        pdef(lines, f"{dname}_w", f"{cname}_w", t4(outch, inch, 3, 3))
        pdef(lines, f"{dname}_b", f"{cname}_b", t1(outch))
        adef(lines, f"{dname}_out", f"{cname}_out", t4(1, outch, m, m))
        adef(lines, f"{name}_relu{block}_{layer}_out", f"relu{block}_{layer}_out", t4(1, outch, m, m))

    # MaxPool outputs
    pools = [
        (1, 64, 112),
        (2, 128, 56),
        (3, 256, 28),
        (4, 512, 14),
        (5, 512, 7),
    ]
    for block, ch, m in pools:
        adef(lines, f"{name}_pool{block}_out", f"pool{block}_out", t4(1, ch, m, m))

    # Classifier
    adef(lines, f"{name}_flatten_out", "flatten_out", t2(1, 25088))
    pdef(lines, f"{name}_fc1_w", "fc1_w", t2(4096, 25088))
    pdef(lines, f"{name}_fc1_b", "fc1_b", t1(4096))
    adef(lines, f"{name}_fc1_out", "fc1_out", t2(1, 4096))
    adef(lines, f"{name}_fc1_relu_out", "fc1_relu_out", t2(1, 4096))
    pdef(lines, f"{name}_fc2_w", "fc2_w", t2(4096, 4096))
    pdef(lines, f"{name}_fc2_b", "fc2_b", t1(4096))
    adef(lines, f"{name}_fc2_out", "fc2_out", t2(1, 4096))
    adef(lines, f"{name}_fc2_relu_out", "fc2_relu_out", t2(1, 4096))
    pdef(lines, f"{name}_fc3_w", "fc3_w", t2(1000, 4096))
    pdef(lines, f"{name}_fc3_b", "fc3_b", t1(1000))
    adef(lines, f"{name}_output", "output", t2(1, 1000))
    lines.append("")

    # Assemble network
    lines.append(f"Definition {name}_comm : comm :=" )
    first = True
    for block, layer, inch, outch, m in convs:
        cname = f"conv{block}_{layer}"
        dname = f"{name}_{cname}"
        if first:
            input_src = f"{name}_input"
            first = False
        else:
            # input is relu output of previous conv; special-case first of block > 1
            prev_block, prev_layer, _, prev_outch, _ = convs[convs.index((block, layer, inch, outch, m)) - 1]
            input_src = f'(vrd "relu{prev_block}_{prev_layer}_out")'
        lines.append(f"  @Conv4d_p1_k3_comm 1 {inch} {outch} {m}")
        lines.append(f"    {input_src} {dname}_w {dname}_b {dname}_out ';")
        lines.append(f"  @ReLU4d_comm 1 {outch} {m} {m}")
        lines.append(f"    (vrd \"{cname}_out\") {name}_relu{block}_{layer}_out ';")
        # Pool after last conv of each block
        if layer == len([c for c in convs if c[0] == block]):
            lines.append(f"  @MaxPool4d_k2_comm 1 {outch} {m}")
            lines.append(f"    (vrd \"relu{block}_{layer}_out\") {name}_pool{block}_out ';")

    lines.append("  @Flatten4d_comm 1 512 7 7")
    lines.append('    (vrd "pool5_out") vgg16_flatten_out \';')
    lines.append("  @Linear2d_comm 1 25088 4096")
    lines.append("    vgg16_fc1_w vgg16_fc1_b (vrd \"flatten_out\") vgg16_fc1_out ';")
    lines.append("  @ReLU2d_comm 1 4096")
    lines.append("    (vrd \"fc1_out\") vgg16_fc1_relu_out ';")
    lines.append("  @Linear2d_comm 1 4096 4096")
    lines.append("    vgg16_fc2_w vgg16_fc2_b (vrd \"fc1_relu_out\") vgg16_fc2_out ';")
    lines.append("  @ReLU2d_comm 1 4096")
    lines.append("    (vrd \"fc2_out\") vgg16_fc2_relu_out ';")
    lines.append("  @Linear2d_comm 1 4096 1000")
    lines.append("    vgg16_fc3_w vgg16_fc3_b (vrd \"fc2_relu_out\") vgg16_output.")
    lines.append("")

    # In the development repository this file lands in src/Ionia/; when run
    # standalone it is written to the current directory.
    out_path = Path("CNN_Examples_VGG16.v")
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    generate()
