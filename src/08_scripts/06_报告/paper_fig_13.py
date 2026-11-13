"""Paper Fig.13 replication: NMSE / RMSE vs propagation steps x neighbours.

Our GraphRefineNet is trained with three MRConv layers, so the propagation-step
axis is 1..3 instead of the paper's 1..5 (documented in the report).  Neighbours
k is a runtime parameter of the kNN graph, so 4 / 8 / 16 are all valid.

Inference only, no retraining.
"""
import argparse
import json
import os
import types

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import RadioMapSeer
from evaluate import build_from_ckpt


def make_forward(steps):
    """Same as GraphRefineNet.forward but runs only the first `steps` layers."""

    def fwd(self, obs, coarse, mask):
        x_in = obs * mask + coarse * (1.0 - mask)
        f_map = self.map_branch(torch.cat([x_in, mask], dim=1))
        f_gui = self.guide_branch(coarse)
        feat = self.merge(torch.cat([f_map, f_gui], dim=1))

        z = self.patch_embed(feat)
        B, C, H, W = z.shape
        z = z.flatten(2).transpose(1, 2) + self.pos_embed

        if self.gcn_mode == "static":
            knn_idx = self.build_knn(self.pos_embed.expand(B, -1, -1), self.k)
        else:
            knn_idx = self.build_knn(z, self.k)
        for layer in self.gcn[:steps]:
            z = z + layer(z, knn_idx)

        return self.g2i(z.transpose(1, 2).reshape(B, C, H, W))

    return fwd


@torch.no_grad()
def score(model, ds, n):
    se = sq = 0.0
    cnt = 0
    for i in range(min(n, len(ds))):
        b = ds[i]
        obs = b["obs"][None].cuda()
        env = b["env"][None].cuda()
        m = b["mask"][None].cuda()
        tgt = b["target"][None].cuda()
        _, y, _ = model(obs, env, m)
        y = y.clamp(0, 1)
        se += float(((y - tgt) ** 2).sum())
        sq += float((tgt ** 2).sum())
        cnt += tgt.numel()
    return se / max(sq, 1e-12), (se / max(cnt, 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--ckpt", default="runs/txf_mffgcn_s1/best.pt")
    ap.add_argument("--out", default="/tmp/pfig")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model, _ = build_from_ckpt(ck, torch.device("cuda"))
    model.eval()
    n_layers = len(model.refine.gcn)
    print("[info] gcn layers in ckpt:", n_layers, flush=True)

    ds = RadioMapSeer(args.cache, "test", "s1", seed=2024)
    steps_list = list(range(1, n_layers + 1))
    ks = [4, 8, 16]
    res = {}

    for k in ks:
        for st in steps_list:
            model.refine.k = k
            model.refine.forward = types.MethodType(make_forward(st), model.refine)
            nm, rm = score(model, ds, args.n)
            res["k%d_step%d" % (k, st)] = (nm, rm)
            print("[fig13] k=%2d steps=%d  NMSE %.6f  RMSE %.6f" % (k, st, nm, rm), flush=True)

    with open(os.path.join(args.out, "fig13.json"), "w") as fh:
        json.dump({a: list(b) for a, b in res.items()}, fh, indent=1)

    style = {4: ("o", "#1f77b4"), 8: ("s", "#ff7f0e"), 16: ("*", "#2ca02c")}
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 8.2), dpi=200)
    for ax, idx, ylab, tag in ((axes[0], 0, "NMSE", "a"), (axes[1], 1, "RMSE", "b")):
        for k in ks:
            ys = [res["k%d_step%d" % (k, st)][idx] for st in steps_list]
            mk, col = style[k]
            ax.plot(steps_list, ys, marker=mk, ms=6, lw=1.4, color=col,
                    label="%d neighbors" % k)
        ax.set_xlabel("Propagation Steps")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.35)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title("(%s) %s" % (tag, ylab), y=-0.22, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig13.png"))
    print("[done] ->", args.out, flush=True)


if __name__ == "__main__":
    main()
