#!/usr/bin/env python
"""TF32 factor experiments + per-layer breakdown (2026-08-28 performance
gap analysis).

1) ResNet18 / VGG16 end-to-end: cudnn.allow_tf32 True vs False (everything
   else follows the bench_torch_baseline.py protocol: batch=1, 224^2, eval,
   inference_mode, cudnn.benchmark=True, warmup 20, min/median of 50 runs).
2) Single-conv microbench: representative shapes of each ResNet18 stage +
   the large VGG convs, TF32 on/off, reporting measured TFLOP/s.
3) ResNet18 per-stage breakdown (profiler, aten op self GPU time grouped
   by name), one run each with TF32 on/off.
"""
import json
import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models

torch.manual_seed(20260828)
DEV = "cuda"
OUT = {}


def bench_model(name, net, x, tf32, iters=50, warmup=20):
    cudnn.benchmark = True
    cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    net = net.eval().to(DEV)
    x = x.to(DEV)
    with torch.inference_mode():
        for _ in range(warmup):
            net(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record(); net(x); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    times.sort()
    return {"min": times[0], "median": times[len(times) // 2]}


def bench_conv(cin, cout, hw, k=3, s=1, p=1, tf32=True, iters=200, warmup=50):
    cudnn.benchmark = True
    cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    conv = torch.nn.Conv2d(cin, cout, k, s, p, bias=False).eval().to(DEV)
    x = torch.randn(1, cin, hw, hw, device=DEV)
    flops = 2.0 * cout * (hw // s) ** 2 * cin * k * k
    with torch.inference_mode():
        for _ in range(warmup):
            conv(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            s_ = torch.cuda.Event(True); e_ = torch.cuda.Event(True)
            s_.record(); conv(x); e_.record()
            torch.cuda.synchronize()
            times.append(s_.elapsed_time(e_))
    times.sort()
    tmin = times[0]
    return {"min_ms": tmin, "median_ms": times[len(times) // 2],
            "tflops": flops / (tmin * 1e-3) / 1e12}


# ---------- 1) end-to-end ----------
nets = {
    "resnet18": (models.resnet18(weights=None), (1, 3, 224, 224)),
    "vgg16": (models.vgg16(weights=None), (1, 3, 224, 224)),
}
for name, (net, shape) in nets.items():
    x = torch.randn(*shape)
    OUT[f"e2e_{name}"] = {}
    for tf32 in (True, False):
        OUT[f"e2e_{name}"][f"tf32_{tf32}"] = bench_model(name, net, x, tf32)
        print(f"e2e {name} tf32={tf32}: {OUT[f'e2e_{name}'][f'tf32_{tf32}']}",
              flush=True)
    del net
    torch.cuda.empty_cache()

# ---------- 2) single-conv microbench ----------
shapes = {
    # name: (cin, cout, hw, k, s, p)
    "res18_stem_7x7_s2_112":   (3,   64, 224, 7, 2, 3),
    "res18_l1_64_56":          (64,  64, 56, 3, 1, 1),
    "res18_l2_128_28":         (128, 128, 28, 3, 1, 1),   # stage2 representative conv
    "res18_l2down_64_128_s2":  (64, 128, 56, 3, 2, 1),
    "res18_l3_256_14":         (256, 256, 14, 3, 1, 1),
    "res18_l4_512_7":          (512, 512, 7, 3, 1, 1),
    "vgg_conv3_3_256_56":      (256, 256, 56, 3, 1, 1),
    "vgg_conv5_512_14":        (512, 512, 14, 3, 1, 1),
}
OUT["conv_micro"] = {}
for name, (cin, cout, hw, k, s, p) in shapes.items():
    OUT["conv_micro"][name] = {}
    for tf32 in (True, False):
        r = bench_conv(cin, cout, hw, k, s, p, tf32)
        OUT["conv_micro"][name][f"tf32_{tf32}"] = r
        print(f"conv {name} tf32={tf32}: min={r['min_ms']*1000:.1f}us "
              f"tf={r['tflops']:.2f}", flush=True)

# ---------- 3) ResNet18 per-stage profiler breakdown ----------
def profile_resnet18_stages(tf32):
    """Per-module timing by layer name: hooks are not fine-grained enough, so use the profiler's aten rows + per-module events."""
    cudnn.benchmark = True
    cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    net = models.resnet18(weights=None).eval().to(DEV)
    x = torch.randn(1, 3, 224, 224, device=DEV)
    # per-module event timing (layer1..layer4 / stem / fc), warmed
    mods = {"stem": torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool),
            "layer1": net.layer1, "layer2": net.layer2,
            "layer3": net.layer3, "layer4": net.layer4,
            "gap_fc": torch.nn.Sequential(net.avgpool, torch.nn.Flatten(1), net.fc)}
    res = {}
    with torch.inference_mode():
        h = net.conv1(net.bn1.running_mean.new_zeros(0) if False else x)  # warmup path build
        # manual stage-by-stage forward
        def run_stages(inp):
            a = mods["stem"](inp)
            b = mods["layer1"](a)
            c = mods["layer2"](b)
            d = mods["layer3"](c)
            e = mods["layer4"](d)
            f = mods["gap_fc"](e)
            return f
        for _ in range(30):
            run_stages(x)
        torch.cuda.synchronize()
        # per-stage timing needs the intermediate tensors; use hooks to record each stage's event interval
        segs = {k: [] for k in mods}
        acts = {}
        for _ in range(50):
            s_ev = {k: torch.cuda.Event(True) for k in list(mods) + ["end"]}
            e_ev = {k: torch.cuda.Event(True) for k in list(mods) + ["end"]}
            a = x
            order = ["stem", "layer1", "layer2", "layer3", "layer4", "gap_fc"]
            for k in order:
                s_ev[k].record()
                a = mods[k](a)
                e_ev[k].record()
            torch.cuda.synchronize()
            for k in order:
                segs[k].append(s_ev[k].elapsed_time(e_ev[k]))
    for k, v in segs.items():
        v.sort()
        res[k] = {"min_ms": v[0], "median_ms": v[len(v) // 2]}
    return res


for tf32 in (True, False):
    OUT[f"resnet18_stages_tf32_{tf32}"] = profile_resnet18_stages(tf32)
    print(f"stages tf32={tf32}: {OUT[f'resnet18_stages_tf32_{tf32}']}", flush=True)

with open("experiments/ast_codegen/tf32_gap_results.json", "w") as f:
    json.dump(OUT, f, indent=2)
print("WROTE experiments/ast_codegen/tf32_gap_results.json")
