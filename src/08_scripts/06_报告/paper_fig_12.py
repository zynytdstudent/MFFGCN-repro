"""Fig.12 replica: what the dynamic graph looks like and what it fixes.

Four panels, same order as the paper:
  (a) full ground truth with a red zoom box
  (b) coarse estimate inside the box
  (c) the 16x16 patch grid with kNN edges drawn
  (d) GCN-refined estimate inside the box
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dataset import RadioMapSeer
from evaluate import build_from_ckpt


def zoom_box():
    """Region used for (b) and (d); sits over the building cluster."""
    return 118, 74, 86, 86          # x0, y0, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--ckpt", default="runs/txf_mffgcn_s1/best.pt")
    ap.add_argument("--index", type=int, default=7)
    ap.add_argument("--out", default="/tmp/pfig12")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model, _ = build_from_ckpt(ck, dev)
    model.eval()

    ds = RadioMapSeer(args.cache, "test", "s1", seed=2024)
    b = ds[args.index]
    obs = b["obs"][None].to(dev)
    env = b["env"][None].to(dev)
    mask = b["mask"][None].to(dev)
    gt = b["target"][None].to(dev)

    with torch.no_grad():
        coarse, fine, _ = model(obs, env, mask)
        coarse = coarse.clamp(0, 1)
        fine = fine.clamp(0, 1)

        # rebuild the exact node features the graph is built from, so the
        # edges drawn here are the ones the GCN actually used
        r = model.refine
        x_in = obs * mask + coarse * (1.0 - mask)
        f_map = r.map_branch(torch.cat([x_in, mask], dim=1))
        f_gui = r.guide_branch(coarse)
        feat = r.merge(torch.cat([f_map, f_gui], dim=1))
        z = r.patch_embed(feat)
        B, C, H, W = z.shape
        z = z.flatten(2).transpose(1, 2) + r.pos_embed
        knn = r.build_knn(z, r.k)               # (1, 256, k)

    gt_np = gt[0, 0].cpu().numpy()
    co_np = coarse[0, 0].cpu().numpy()
    fi_np = fine[0, 0].cpu().numpy()
    z_np = z[0].cpu().numpy()                   # (256, dim)
    knn_np = knn[0].cpu().numpy()               # (256, k)

    x0, y0, w, h = zoom_box()

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.4), dpi=200)

    ax = axes[0]
    ax.imshow(gt_np, cmap="viridis")
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False,
                           edgecolor="#d01010", lw=1.2))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(a)", y=-0.14, fontsize=12)

    ax = axes[1]
    ax.imshow(co_np[y0:y0 + h, x0:x0 + w], cmap="viridis")
    ax.add_patch(Rectangle((0, 0), w - 1, h - 1, fill=False,
                           edgecolor="#d01010", lw=1.2))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(b)", y=-0.14, fontsize=12)

    # ---- (c) patch grid + kNN edges -------------------------------------
    ax = axes[2]
    g = 16
    colour = z_np.mean(axis=1).reshape(g, g)
    colour = (colour - colour.min()) / max(float(np.ptp(colour)), 1e-8)
    # draw each patch as a square with a small gap, paper style
    for i in range(g):
        for j in range(g):
            ax.add_patch(Rectangle((j + 0.06, i + 0.06), 0.88, 0.88,
                                   facecolor=plt.cm.viridis(colour[i, j]),
                                   edgecolor="none"))
    # only the edges of the zoomed block are drawn, otherwise the plot turns
    # into a hairball
    cols = slice(x0 * g // 256, (x0 + w) * g // 256)
    rows = slice(y0 * g // 256, (y0 + h) * g // 256)
    keep = set()
    for i in range(rows.start, rows.stop):
        for j in range(cols.start, cols.stop):
            keep.add(i * g + j)
    for n in keep:
        i, j = divmod(n, g)
        for m in knn_np[n]:
            if m not in keep:
                continue
            i2, j2 = divmod(int(m), g)
            ax.plot([j + 0.5, j2 + 0.5], [i + 0.5, i2 + 0.5],
                    color="#e02020", lw=0.5, alpha=0.55, zorder=5)
    for n in keep:
        i, j = divmod(n, g)
        ax.plot(j + 0.5, i + 0.5, "o", ms=1.8, color="#b01010", zorder=6)
    ax.set_xlim(0, g); ax.set_ylim(g, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title("(c)", y=-0.14, fontsize=12)

    ax = axes[3]
    ax.imshow(fi_np[y0:y0 + h, x0:x0 + w], cmap="viridis")
    ax.add_patch(Rectangle((0, 0), w - 1, h - 1, fill=False,
                           edgecolor="#d01010", lw=1.2))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(d)", y=-0.14, fontsize=12)

    fig.tight_layout()
    out = os.path.join(args.out, "fig12.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("[save]", out)

    # diagnostics that go into the report text
    def nmse(a):
        return float(((a - gt_np) ** 2).sum() / (gt_np ** 2).sum())
    print("[num] coarse NMSE %.5f  refined NMSE %.5f  gain %.1f%%"
          % (nmse(co_np), nmse(fi_np), 100 * (1 - nmse(fi_np) / nmse(co_np))))
    print("[num] nodes %d  k %d  edges in block %d"
          % (z_np.shape[0], knn_np.shape[1], len(keep) * knn_np.shape[1]))


if __name__ == "__main__":
    main()
