"""Paper Fig.16 replication: impact of missing buildings (1-4).

The building layer of the ENVIRONMENT input has its N largest connected
components erased before the forward pass; observations and target stay
untouched, i.e. the map is incomplete but the measurement is not.

Inference only.
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

from dataset import RadioMapSeer
from evaluate import build_from_ckpt
from baselines import build_baseline

MODELS = [
    ("RadioUNet", "bl_radiounet_s1", "radiounet", "#1f77b4"),
    ("RME-GAN", "bl_rmegan_s1", "pix2pix", "#ff7f0e"),
    ("MFFGCN", "txf_mffgcn_s1", "mffgcn", "#2ca02c"),
]


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


def drop_buildings(bld, n):
    """Zero the n largest connected building components of a (1,256,256) map."""
    if n <= 0:
        return bld
    b = (bld[0].cpu().numpy() > 0.5).astype(np.uint8)
    lab, num = ndimage.label(b)
    if num == 0:
        return bld
    areas = np.asarray(ndimage.sum(b, lab, range(1, num + 1)))
    # The biggest component of a 256x256 map can cover a third of the canvas;
    # erasing it destroys far more than "one missing building".  Restrict the
    # pool to normal-sized buildings, biggest first.
    pool = [i for i in range(num) if 200 <= areas[i] <= 6000]
    if len(pool) < n:
        pool = [i for i in range(num) if areas[i] >= 100]
    pool.sort(key=lambda i: -areas[i])
    out = b.copy()
    for i in pool[:n]:
        out[lab == (i + 1)] = 0
    t = torch.from_numpy(out.astype(np.float32))[None]
    return t


@torch.no_grad()
def score(model, kind, ds, n, miss, device):
    se = sq = 0.0
    cnt = 0
    for i in range(min(n, len(ds))):
        b = ds[i]
        obs = b["obs"][None].to(device)
        env = b["env"][None].to(device)
        mask = b["mask"][None].to(device)
        tgt = b["target"][None].to(device)
        if miss:
            env = env.clone()
            env[:, 0:1] = drop_buildings(env[:, 0:1], miss).to(device)
        if kind == "mffgcn":
            _, y, _ = model(obs, env, mask)
        else:
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
    ap.add_argument("--out", default="/tmp/pfig")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")

    models = []
    for name, tag, kind, color in MODELS:
        m, kind = load_model(tag, kind, device)
        m.eval()
        models.append((name, m, kind))
        print("[load] %-10s %s" % (name, tag), flush=True)

    ds = RadioMapSeer(args.cache, "test", "s1", seed=2024)
    miss_list = [1, 2, 3, 4]
    res = {}
    for miss in miss_list:
        for name, m, kind in models:
            nm, rm = score(m, kind, ds, args.n, miss, device)
            res.setdefault(name, {})[miss] = (nm, rm)
            print("[fig16] miss=%d %-10s NMSE %.5f RMSE %.5f" % (miss, name, nm, rm), flush=True)

    with open(os.path.join(args.out, "fig16.json"), "w") as fh:
        json.dump({k: {str(a): list(b) for a, b in v.items()} for k, v in res.items()},
                  fh, indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=200)
    width = 0.26
    for ax, idx, ylab in ((axes[0], 0, "NMSE"), (axes[1], 1, "RMSE")):
        top = 0.0
        for j, (name, _, _, color) in enumerate(MODELS):
            vals = [res[name][mi][idx] for mi in miss_list]
            top = max(top, max(vals))
            ax.bar(np.arange(len(miss_list)) + (j - 1) * width, vals, width,
                   label=name, color=color)
        ax.set_xticks(np.arange(len(miss_list)))
        ax.set_xticklabels([str(mi) for mi in miss_list])
        ax.set_xlabel("Number of missing buildings")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, top * 1.18)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig16.png"))
    print("[done] ->", args.out, flush=True)


if __name__ == "__main__":
    main()
