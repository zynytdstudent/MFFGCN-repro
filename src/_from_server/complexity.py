"""Complexity analysis: parameters, MACs and latency.

No training involved. Reproduces the "complexity" part of the paper: how big is
each variant, how much compute does one 256x256 map cost, and how fast is a
forward pass.

    python complexity.py                      # params + MACs only (CPU is fine)
    python complexity.py --bench --gpu 0      # also time the forward pass

MACs are counted with hooks on Conv2d / Linear only. The kNN graph built inside
GraphRefineNet uses cdist + topk, which has no hook, so it is reported as a
caveat instead of being silently folded into the total.
"""
import argparse
import time

import torch
import torch.nn as nn

from models import MFFGCN, SingleEncoderUNet


def n_params(mod):
    return sum(p.numel() for p in mod.parameters() if p.requires_grad)


def macs_of(model, run):
    """Count multiply-accumulates for Conv2d and Linear layers."""
    total = [0]

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] * m.kernel_size[1]
            total[0] += out.numel() * (m.in_channels // m.groups) * k
        elif isinstance(m, nn.Linear):
            total[0] += out.numel() * m.in_features

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        run(model)
    for h in hs:
        h.remove()
    return total[0]


def build(kind):
    if kind == "mffgcn":
        return MFFGCN(), "MFFGCN (proposed)"
    if kind == "mffgcn_w_o_dynamic":
        return MFFGCN(gcn_mode="static"), "MFFGCN w/o Dynamic"
    if kind == "mffgcn_w_o_gcn":
        return MFFGCN(gcn_mode="none"), "MFFGCN w/o GCN"
    if kind == "mffgcn_w_o_residual":
        return MFFGCN(residual=False), "MFFGCN w/o Residual"
    if kind == "single":
        return SingleEncoderUNet(), "Single Encoder"
    raise ValueError(kind)


VARIANTS = ["mffgcn", "mffgcn_w_o_dynamic", "mffgcn_w_o_gcn",
            "mffgcn_w_o_residual", "single"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="time the forward pass")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()

    dev = torch.device(f"cuda:{a.gpu}" if a.bench and torch.cuda.is_available()
                       else "cpu")
    obs = torch.randn(1, 1, 256, 256, device=dev)
    env = torch.randn(1, 2, 256, 256, device=dev)
    msk = (torch.rand(1, 1, 256, 256, device=dev) > 0.9).float()
    stacked = torch.cat([obs, env], dim=1)

    def runner(model):
        if isinstance(model, SingleEncoderUNet):
            return lambda m: m(stacked)
        return lambda m: m(obs, env, msk)

    print(f"{'variant':22s} {'params(M)':>10s} {'fusion(M)':>10s} "
          f"{'refine(M)':>10s} {'GMACs':>8s} {'ms/fig':>8s}")
    for k in VARIANTS:
        model, name = build(k)
        model = model.to(dev).eval()
        run = runner(model)
        p_all = n_params(model)
        p_f = n_params(model.fusion) if hasattr(model, "fusion") else float("nan")
        p_r = n_params(model.refine) if hasattr(model, "refine") else float("nan")
        g = macs_of(model, run) / 1e9

        ms = float("nan")
        if a.bench:
            with torch.no_grad():
                for _ in range(10):
                    run(model)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(a.iters):
                    run(model)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                ms = (time.time() - t0) / a.iters * 1000

        print(f"{name:22s} {p_all/1e6:10.2f} {p_f/1e6:10.2f} {p_r/1e6:10.2f} "
              f"{g:8.1f} {ms:8.2f}")

    print("\nMACs cover Conv2d and Linear only; the kNN graph (cdist + topk) "
          "inside GraphRefineNet is not counted. 256x256 map, batch 1.")


if __name__ == "__main__":
    main()
