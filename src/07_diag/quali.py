"""quali.py - 生成定性对比图：三行分别是 Setup 1/2/3，各选不同的测试图。

列：真值 / 稀疏观测 / 可观测区域 / 粗估计 / 精化结果 / 误差图
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import RadioMapSeer
from evaluate import build_from_ckpt

CACHE = "../data/cache"
PICK = [("s1", 7), ("s2", 23), ("s3", 41)]
CKPT = "runs/FINAL_{}/last.pt"
OUT = "figs/fig5_quali.png"

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
plt.rcParams.update({"font.size": 9})

fig, axes = plt.subplots(3, 6, figsize=(19, 10))
for r, (setup, idx) in enumerate(PICK):
    ck = torch.load(CKPT.format(setup), map_location=dev, weights_only=False)
    model, arch = build_from_ckpt(ck, dev)
    model.eval()
    ds = RadioMapSeer(CACHE, "test", setup, seed=2024, obs_on_signal=True)
    b = ds[idx]
    obs, env = b["obs"][None].to(dev), b["env"][None].to(dev)
    mask, tgt = b["mask"][None].to(dev), b["target"][None].to(dev).float()
    allowed = b["allowed"][None].to(dev)
    with torch.no_grad():
        if arch == "single":
            fine = coarse = model(torch.cat([obs, env], dim=1)).clamp(0, 1)
        else:
            coarse, fine, _ = model(obs, env, mask)
            coarse, fine = coarse.clamp(0, 1), fine.clamp(0, 1)

    t = tgt[0, 0].cpu().numpy()
    o = obs[0, 0].cpu().numpy()
    c = coarse[0, 0].cpu().numpy()
    f = fine[0, 0].cpu().numpy()
    e = (fine - tgt)[0, 0].cpu().numpy()
    a = allowed[0, 0].cpu().numpy().astype(float)

    nm_c = ((c - t) ** 2).sum() / (t ** 2).sum()
    nm_f = ((f - t) ** 2).sum() / (t ** 2).sum()

    bld = env[0, 0].cpu().numpy()
    txm = env[0, 1].cpu().numpy()
    panels = [
        (t, "Ground truth", "viridis", 0, 1, None),
        (o, "Sparse observations", "viridis", 0, 1, None),
        (np.where(a < 0.5, bld * 0.35, bld), "Input: buildings + Tx\n(masked area dimmed, red x = Tx)",
         "gray", 0, 1, a),
        (c, "Coarse estimate\nNMSE %.4f" % nm_c, "viridis", 0, 1, None),
        (f, "MFFGCN refined\nNMSE %.4f" % nm_f, "viridis", 0, 1, None),
        (e, "Error map\nMAE %.3f" % np.abs(e).mean(), "RdBu_r", -0.3, 0.3, a),
    ]
    for k, (img, ttl, cmap, vmin, vmax, region) in enumerate(panels):
        ax = axes[r, k]
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(ttl, fontsize=9.5)
        ax.axis("off")
        if k == 2 and txm.max() > 0:
            ys, xs = np.nonzero(txm > 0.5)
            ax.scatter(xs.mean(), ys.mean(), c="red", s=28, marker="x", lw=1.6)
        if region is not None and (region < 0.5).any():
            ax.contour(region, levels=[0.5], colors="k", linewidths=1.1)
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        if k == 0:
            ax.text(-0.10, 0.5, "Setup %d" % (r + 1), transform=ax.transAxes,
                    rotation=90, va="center", ha="center", fontsize=13)

fig.suptitle("Qualitative results on three uneven-sampling setups "
             "(same model family, different test maps)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
os.makedirs("figs", exist_ok=True)
fig.savefig(OUT, dpi=170)
print("[ok]", OUT)
