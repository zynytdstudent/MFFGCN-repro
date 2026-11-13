# -*- coding: utf-8 -*-
"""Fig.14 第三轮诊断图：噪声注入方向拆分 + 中值滤波对照。

数据来自 diag_noise_fix.py 的三个 json（同一份 s3 权重，n=60），
只画"同一批权重、不同注入口径"的对比，不掺论文数字化值（除两条参考线）。

左 (a)：绝对 NMSE（log 轴），看量级；
右 (b)：相对干净基线的劣化倍数，虚线是论文 Fig.14 给出的 MFFGCN 区间
        （数字化 ×1.9 ~ 正文 +145% = ×2.45），用来判断"回到同量级没有"。

输出 ../figs/fig14_diag.png（2400x640 @200dpi），供对照页 Fig.14 卡片配图。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
for _rel in ("..", os.path.join("..", "..")):
    c = os.path.abspath(os.path.join(HERE, _rel))
    if os.path.isdir(os.path.join(c, "results")):
        ROOT = c
        break

CASES = ["clean", "flip_1pct", "flip_med3", "erase_1pct", "add_1pct"]
LABEL = ["clean", "1% flip", "1% flip\n+3x3 med", "1% erase", "1% add"]
MODELS = [
    ("MFFGCN", "diag_noise_fix_txf_mffgcn.json", "#1c1ca8"),
    ("RadioUNet", "diag_noise_fix_bl_radiounet.json", "#d62728"),
    ("SingleEncoder", "diag_noise_fix_txf_single.json", "#2ca02c"),
]


def main():
    data = {}
    for name, fn, color in MODELS:
        p = os.path.join(ROOT, "results", fn)
        d = json.load(open(p, encoding="utf-8"))
        data[name] = d["nmse"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 3.2), dpi=200)
    x = np.arange(len(CASES))
    w = 0.26
    for j, (name, _, color) in enumerate(MODELS):
        nm = data[name]
        b = nm["clean"]
        vals = [nm[c] for c in CASES]
        ax1.bar(x + (j - 1) * w, vals, w, label=name, color=color, alpha=0.9)
        ax2.bar(x + (j - 1) * w, [v / b for v in vals], w, label=name,
                color=color, alpha=0.9)

    ax1.set_yscale("log")
    ax1.set_ylim(0.004, 0.13)
    ax1.set_yticks([0.005, 0.01, 0.02, 0.05, 0.1])
    ax1.set_yticklabels(["0.005", "0.01", "0.02", "0.05", "0.1"], fontsize=8.5)
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.set_ylabel("NMSE", fontsize=10)
    ax1.set_title("(c) absolute NMSE by injection mode", fontsize=10)

    ax2.set_ylabel("degradation  vs clean  (x)", fontsize=10)
    ax2.set_title("(d) degradation factor vs clean", fontsize=10)
    ax2.axhspan(1.9, 2.45, color="#b26a00", alpha=0.20, lw=0)
    ax2.axhline(1.9, color="#b26a00", ls=":", lw=1.1)
    ax2.axhline(2.45, color="#b26a00", ls=":", lw=1.1)


    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(LABEL, fontsize=8.5)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax2.set_ylim(0, 18)
    ax2.set_xlim(-0.55, 5.5)
    handles, labels = ax2.get_legend_handles_labels()
    handles.append(Patch(facecolor="#b26a00", alpha=0.20, edgecolor="none"))
    labels.append("paper MFFGCN ref. 1.9-2.45x")
    ax2.legend(handles, labels, fontsize=7.8, frameon=False, loc="upper left",
               ncol=1, handlelength=1.6)
    fig.tight_layout(w_pad=2.0)

    out = os.path.join(ROOT, "figs", "fig14_diag.png")
    fig.savefig(out, dpi=200, facecolor="white")
    print("saved", out)
    for name, _, _ in MODELS:
        nm = data[name]
        b = nm["clean"]
        print("%-14s" % name, " ".join("%s=%.5f(%.2fx)" % (c, nm[c], nm[c] / b)
                                       for c in CASES))


if __name__ == "__main__":
    main()
