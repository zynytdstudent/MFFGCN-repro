"""Paper Fig.15 replication: impact of map resolution (downsampling ratio).

The building map is downsampled to `ratio` of its original resolution and
upsampled back to 256x256 before the forward pass; observations and target
are untouched.  Ratios follow the paper: 25 / 33 / 50 / 75 percent.

Inference only.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import RadioMapSeer
from evaluate import build_from_ckpt
from baselines import build_baseline

MODELS = [
    ("RadioUNet", "bl_radiounet_s1", "radiounet", "#1f77b4", "o"),
    ("RME-GAN", "bl_rmegan_s1", "pix2pix", "#ff7f0e", "+"),
    ("MFFGCN", "txf_mffgcn_s1", "mffgcn", "#2ca02c", "x"),
]

RATIOS = [25, 33, 50, 75]


def load_model(tag, kind, device):
    ck = torch.load("runs/%s/best.pt" % tag, map_location=device, weights_only=False)
    if kind == "mffgcn":
        m, _ = build_from_ckpt(ck, device)
        return m, "mffgcn"
    m = build_baseline(kind)
    m.load_state_dict(ck["model"])
    if kind == "radiounet":
        m.stage = 2
    return m.to(device), kind


def degrade(env, ratio):
    """Keep `ratio` percent of the map resolution, then scale back to 256."""
    s = max(8, int(round(256 * ratio / 100.0)))
    small = F.interpolate(env, size=(s, s), mode="bilinear",
                          align_corners=False, antialias=True)
    return F.interpolate(small, size=(256, 256), mode="bilinear",
                         align_corners=False)


@torch.no_grad()
def score(model, kind, ds, n, ratio, device):
    se = sq = 0.0
    cnt = 0
    for i in range(min(n, len(ds))):
        b = ds[i]
        obs = b["obs"][None].to(device)
        env = b["env"][None].to(device)
        mask = b["mask"][None].to(device)
        tgt = b["target"][None].to(device)
        env = degrade(env, ratio)
        if kind == "mffgcn":
            _, y, _ = model(obs, env, mask)
        else:
            # baselines take (obs, env); RME-GAN returns (fake, ...)
            out = model(obs, env)
            y = out[-1] if isinstance(out, (tuple, list)) else out
        y = y.clamp(0, 1)
        se += float(((y - tgt) ** 2).sum())
        sq += float((tgt ** 2).sum())
        cnt += tgt.numel()
    return se / max(sq, 1e-12), (se / max(cnt, 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--out", default="/tmp/pfig15")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for name, tag, kind, color, marker in MODELS:
        m, kind2 = load_model(tag, kind, device)
        m.eval()
        models.append((name, m, kind2, color, marker))
        print("[load] %-10s %s (%s)" % (name, tag, kind2), flush=True)

    ds = RadioMapSeer(args.cache, "test", "s1", seed=2024)
    res = {n: {} for n, _, _, _, _ in models}

    for ratio in RATIOS:
        for name, m, kind, color, marker in models:
            nm, rm = score(m, kind, ds, args.n, ratio, device)
            res[name][ratio] = (nm, rm)
            print("[%.0f%%] %-10s NMSE %.5f  RMSE %.5f" % (ratio, name, nm, rm), flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=200)
    xs = RATIOS
    for ax, k, label in ((axes[0], 0, "NMSE"), (axes[1], 1, "RMSE")):
        for name, _, _, color, marker in models:
            ys = [res[name][r][k] for r in xs]
            ax.plot(range(len(xs)), ys, marker=marker, ms=5, lw=1.5,
                    color=color, label=name)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(["%d%%" % r for r in xs])
        ax.set_xlabel("Downsampling ratio", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(args.out, "fig15.png")
    fig.savefig(out)
    plt.close(fig)
    print("[save]", out)

    with open(os.path.join(args.out, "fig15.json"), "w") as f:
        json.dump({k: {str(r): v for r, v in d.items()} for k, d in res.items()},
                  f, indent=1)


if __name__ == "__main__":
    main()
