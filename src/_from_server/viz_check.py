"""Visual sanity check: what does the model actually put on the map?"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import RadioMapSeer
from models import MFFGCN, SingleEncoderUNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--split", default="val")
    ap.add_argument("--setup", default="s1")
    ap.add_argument("--arch", default="single")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="../eval_viz")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda:0"
    model = (SingleEncoderUNet() if a.arch == "single" else MFFGCN()).to(dev)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print("ckpt epoch", ck["epoch"], "val", {k: round(v, 4) for k, v in ck["val"].items()})

    ds = RadioMapSeer(a.cache, a.split, a.setup, seed=2024)
    for i in range(a.n):
        b = ds[i]
        obs = b["obs"][None].to(dev)
        env = b["env"][None].to(dev)
        mask = b["mask"][None].to(dev)
        tgt = b["target"][None].to(dev)
        with torch.no_grad():
            if a.arch == "single":
                pred = model(torch.cat([obs, env], 1)).clamp(0, 1)
            else:
                _, pred, _ = model(obs, env, mask)
                pred = pred.clamp(0, 1)
        t = tgt[0, 0].cpu().numpy()
        p = pred[0, 0].cpu().numpy()
        o = obs[0, 0].cpu().numpy()
        e = p - t

        fig, ax = plt.subplots(1, 5, figsize=(21, 4.4))
        panels = [
            (t, "Ground truth", "viridis", 0, 1),
            (o, "Sparse observations", "viridis", 0, 1),
            (env[0, 0].cpu().numpy(), "Building map", "gray", 0, 1),
            (p, "Prediction", "viridis", 0, 1),
            (e, "Error (pred - GT)", "RdBu_r", -0.5, 0.5),
        ]
        for k, (img, title, cmap, vmin, vmax) in enumerate(panels):
            im = ax[k].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax[k].set_title(title, fontsize=12)
            ax[k].axis("off")
            plt.colorbar(im, ax=ax[k], fraction=0.046)
        fig.tight_layout()
        fn = os.path.join(a.out, f"viz_{a.arch}_{i}.png")
        fig.savefig(fn, dpi=110)
        plt.close(fig)
        mse = float(((p - t) ** 2).mean())
        corr = float(np.corrcoef(p.ravel(), t.ravel())[0, 1])
        print(f"  sample {i}: mse={mse:.5f} corr={corr:.3f} pred_mean={p.mean():.3f} "
              f"gt_mean={t.mean():.3f} -> {fn}")


if __name__ == "__main__":
    main()
