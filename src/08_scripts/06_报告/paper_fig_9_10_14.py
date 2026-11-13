"""Paper Fig.9 / Fig.10 / Fig.14 replication (inference only).

Runs three checkpoints (RadioUNet, RME-GAN, MFFGCN) over:
  * a square observable window: side = 128 with dist = 50/70/90/110   -> Fig.9
                                dist = 0   with side = 90/110/130/150 -> Fig.10
  * salt & pepper impulse noise 0.1% / 0.4% / 0.7% / 1.0% on Setup 1  -> Fig.14

Global NMSE / RMSE over the whole map, test split.
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import RadioMapSeer
from evaluate import build_from_ckpt
from baselines import build_baseline

MODELS = [
    ("RadioUNet", "bl_radiounet_s1", "radiounet", "#1f77b4"),
    ("RME-GAN", "bl_rmegan_s1", "pix2pix", "#ff7f0e"),
    ("MFFGCN", "txf_mffgcn_s1", "mffgcn", "#2ca02c"),
]


def load_model(tag, kind, device):
    """Baselines were trained by baseline_train.py with their own classes."""
    ck = torch.load("runs/%s/best.pt" % tag, map_location=device, weights_only=False)
    if kind == "mffgcn":
        m, _ = build_from_ckpt(ck, device)
        return m, "mffgcn"
    m = build_baseline(kind)
    m.load_state_dict(ck["model"])
    if kind == "radiounet":
        m.stage = 2                      # use both stages, like the paper
    return m.to(device), kind


@torch.no_grad()
def score(model, kind, ds, n, device):
    se = sq = 0.0
    cnt = 0
    for i in range(min(n, len(ds))):
        b = ds[i]
        obs = b["obs"][None].to(device)
        env = b["env"][None].to(device)
        mask = b["mask"][None].to(device)
        tgt = b["target"][None].to(device)
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


def bars(ax, xs, data, ylabel, title):
    """Grouped bars, same colours / order as the paper."""
    width = 0.26
    top = 0.0
    for k, (name, _, _, color) in enumerate(MODELS):
        vals = [data[name][x] for x in xs]
        top = max(top, max(vals))
        ax.bar(np.arange(len(xs)) + (k - 1) * width, vals, width,
               label=name, color=color)
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("Distance to center" if title == "dist" else "Side length")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, top * 1.18)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax.grid(axis="y", alpha=0.25)


def line(ax, xs, data, ylabel, fname_label):
    for name, _, _, color in MODELS:
        ax.plot(range(len(xs)), [data[name][x] for x in xs],
                marker="o", ms=4, lw=1.4, color=color, label=name)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([fname_label(x) for x in xs])
    ax.set_xlabel("Noise intensity")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="/tmp/pfig")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for name, tag, kind, color in MODELS:
        m, kind = load_model(tag, kind, device)
        m.eval()
        models.append((name, m, kind))
        print("[load] %-10s %s (%s)" % (name, tag, kind), flush=True)

    res = {}

    # ---------- Fig.9 : side fixed at 128, distance to centre 50..110 -------
    fig9 = {n: {} for n, _, _, _ in MODELS}
    for d in [50, 70, 90, 110]:
        # 32 px window keeps the whole 50..110 range inside the map, so the
        # distance axis is not clipped (a 128 px window cannot move that far).
        ds = RadioMapSeer(args.cache, "test", "sq", seed=2024,
                          region_side=32, region_dist=d)
        for name, m, kind in models:
            nm, rm = score(m, kind, ds, args.n, device)
            fig9[name][d] = (nm, rm)
            print("[fig9]  dist=%3d %-10s NMSE %.5f  RMSE %.5f" % (d, name, nm, rm), flush=True)
    res["fig9"] = {k: {str(a): b for a, b in v.items()} for k, v in fig9.items()}

    # ---------- Fig.10 : centred window, side length 90..150 ----------------
    fig10 = {n: {} for n, _, _, _ in MODELS}
    for s in [90, 110, 130, 150]:
        ds = RadioMapSeer(args.cache, "test", "sq", seed=2024,
                          region_side=s, region_dist=0)
        for name, m, kind in models:
            nm, rm = score(m, kind, ds, args.n, device)
            fig10[name][s] = (nm, rm)
            print("[fig10] side=%3d %-10s NMSE %.5f  RMSE %.5f" % (s, name, nm, rm), flush=True)
    res["fig10"] = {k: {str(a): b for a, b in v.items()} for k, v in fig10.items()}

    # ---------- Fig.14 : impulse noise on Setup 1 ---------------------------
    fig14 = {n: {} for n, _, _, _ in MODELS}
    for p in [0.001, 0.004, 0.007, 0.010]:
        ds = RadioMapSeer(args.cache, "test", "s1", seed=2024, impulse=p)
        for name, m, kind in models:
            nm, rm = score(m, kind, ds, args.n, device)
            fig14[name][p] = (nm, rm)
            print("[fig14] p=%.3f %-10s NMSE %.5f  RMSE %.5f" % (p, name, nm, rm), flush=True)
    res["fig14"] = {k: {str(a): b for a, b in v.items()} for k, v in fig14.items()}

    with open(os.path.join(args.out, "fig9_10_14.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    # ---------- plot -------------------------------------------------------
    for key, xs in [("fig9", [50, 70, 90, 110]), ("fig10", [90, 110, 130, 150])]:
        data = {n: {int(j): v[0] for j, v in res[key][n].items()} for n, _, _, _ in MODELS}
        data_r = {n: {int(j): v[1] for j, v in res[key][n].items()} for n, _, _, _ in MODELS}
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=200)
        bars(axes[0], xs, data, "NMSE", "dist" if key == "fig9" else "side")
        bars(axes[1], xs, data_r, "RMSE", "dist" if key == "fig9" else "side")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "%s.png" % key))
        plt.close(fig)

    data = {n: {float(j): v[0] for j, v in res["fig14"][n].items()} for n, _, _, _ in MODELS}
    data_r = {n: {float(j): v[1] for j, v in res["fig14"][n].items()} for n, _, _, _ in MODELS}
    ps = [0.001, 0.004, 0.007, 0.010]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=200)
    line(axes[0], ps, data, "NMSE", lambda x: "%.1f%%" % (x * 100))
    line(axes[1], ps, data_r, "RMSE", lambda x: "%.1f%%" % (x * 100))
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig14.png"))
    plt.close(fig)
    print("[done] ->", args.out, flush=True)


if __name__ == "__main__":
    main()
