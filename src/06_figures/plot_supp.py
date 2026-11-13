# -*- coding: utf-8 -*-
"""把四组补充实验合成一张 2x2 总图（figs/supp_all.png）。

读 ../results/supp_{missing,degrade,noise,shape}.json（由 04_eval/supp_eval.py 产出）。
在服务器上跑（本地没有 matplotlib）。

用法：python plot_supp.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "../results"
C = {"MFFGCN": "#2E7D32", "RadioUNet": "#1f77b4", "RME-GAN": "#ff7f0e"}
M = {"MFFGCN": "MFFGCN (ours)", "RadioUNet": "RadioUNet", "RME-GAN": "RME-GAN"}


def load(mode):
    with open(os.path.join(R, "supp_%s.json" % mode), encoding="utf-8") as fh:
        return json.load(fh)


def line_panel(ax, d, xkey, xlab, title, xticklabels=None, logx=False):
    levels = sorted(d["models"]["MFFGCN"], key=lambda k: float(k))
    for name in ("RadioUNet", "RME-GAN", "MFFGCN"):
        v = d["models"].get(name)
        if not v:
            continue
        ys = [v[k]["nmse"] for k in levels]
        xs = [float(k) for k in levels]
        ax.plot(xs, ys, "o-", lw=1.9, ms=5.5, label=M[name], color=C[name])
    if logx:
        ax.set_xscale("log", base=2)
        ax.invert_xaxis()          # 左=完好 右=退化，方向与"退化"直觉一致
    ax.set_xticks([float(k) for k in levels])
    lbl = xticklabels or levels
    ax.set_xticklabels(lbl, fontsize=8.5,
                       rotation=0 if len(str(lbl[0])) > 2 else 0)
    if xticklabels and len(str(xticklabels[0])) <= 2:
        ax.tick_params(axis="x", pad=2)
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel("NMSE", fontsize=9)
    ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=0.25, lw=0.7)
    ax.legend(fontsize=8, frameon=False)


fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0), dpi=170)

dm = load("missing")
line_panel(axes[0, 0], dm, "n", "Number of missing buildings",
           "(a) Missing buildings (Fig.16, Setup 3)\nerased area %.2f%%-%.2f%%"
           % (min(v for k, v in dm["erase_frac"].items() if k != "0") * 100,
              max(dm["erase_frac"].values()) * 100))

dd = load("degrade")
line_panel(axes[0, 1], dd, "res", "Environment map resolution (px, re-binarized)",
           "(b) Environment resolution (Fig.15, Setup 3)", logx=True)

dn = load("noise")
line_panel(axes[1, 0], dn, "p", "Impulse noise ratio on building map",
           "(c) Impulse noise (Fig.14, Setup 3)",
           xticklabels=["0", "0.1%", "0.5%", "1%", "2%"])

ds = load("shape")
shapes = ["strip", "circle", "triangle", "irregular"]
lab = ["strip\n(seen)", "circle", "triangle", "irregular"]
w = 0.26
x = np.arange(len(shapes))
for j, name in enumerate(("RadioUNet", "RME-GAN", "MFFGCN")):
    v = ds["models"][name]
    axes[1, 1].bar(x + (j - 1) * w, [v[s]["nmse"] for s in shapes], w,
                   label=M[name], color=C[name])
    for xi, s in zip(x + (j - 1) * w, shapes):
        axes[1, 1].text(xi, v[s]["nmse"] * 1.02, "%.4f" % v[s]["nmse"],
                        ha="center", va="bottom", fontsize=6.8, rotation=90)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(lab, fontsize=9)
axes[1, 1].set_ylabel("NMSE", fontsize=9)
axes[1, 1].set_title("(d) Zero-shot blind-zone shapes (Table VI, s2 weights)", fontsize=10.5)
axes[1, 1].grid(axis="y", alpha=0.25, lw=0.7)
axes[1, 1].set_ylim(0, max(max(v[s]["nmse"] for s in shapes)
                           for _, v in ds["models"].items()) * 1.42)
axes[1, 1].legend(fontsize=8, frameon=False, loc="upper left", ncol=3)

fig.suptitle("Supplementary experiments: sensitivity to environment map quality and blind-zone shape", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "../figs/supp_all.png"
fig.savefig(out, bbox_inches="tight")
print("saved:", out)
