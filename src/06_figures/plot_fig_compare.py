# -*- coding: utf-8 -*-
"""plot_fig_compare.py - 把"我们的复现"画在**论文自己的坐标范围**里。

为什么需要它：之前 reports/09 里 Fig.9/10/14/15/16 的右图直接用了旧脚本的
默认坐标（纵轴自动缩放、横轴换成绝对像素），同一条曲线放进两个坐标系，
形状当然完全不同。这里强制：
  1. 纵轴范围 = 论文图上的刻度范围（逐图写死在 AXES 里）；
  2. 横轴刻度 = 论文图上的档位（0.1/0.4/0.7/1.0%、25/33/50/75%、1-4 栋、50-110、90-150）；
  3. 如果能从论文位图反解出曲线（results/paper_digitized.json），就叠加同轴对比。

输出 ../figs/cmp_<mode>.png
用法：python plot_fig_compare.py [mode ...]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOR = {"RadioUNet": "#1f77b4", "RME-GAN": "#ff7f0e", "MFFGCN": "#2ca02c"}
ORDER = ["RadioUNet", "RME-GAN", "MFFGCN"]

# 每张图：模式 -> (横轴档位, 横轴标签, 纵轴范围 NMSE, RMSE, 图型, 论文图号)
AXES = {
    "dist":    ([50, 70, 90, 110], "Distance from map centre (px)",
                (0.0040, 0.0072), (0.0140, 0.0200), "bar", "9"),
    "side":    ([90, 110, 130, 150], "Side length of restricted region (px)",
                (0.0040, 0.0072), (0.0140, 0.0200), "bar", "10"),
    "noise":   ([0.001, 0.004, 0.007, 0.010], "Noise intensity",
                (0.006, 0.070), (0.020, 0.070), "line", "14"),
    "degrade": ([64, 85, 128, 192], "Downsampling rate",
                (0.020, 0.070), (0.035, 0.065), "line", "15"),
    "missing": ([1, 2, 3, 4], "Number of missing buildings",
                (0.020, 0.090), (0.030, 0.080), "bar", "16"),
}

# 论文横轴不是数字而是百分比时，刻度文字这样写
XTICK = {
    "noise": ["0.1%", "0.4%", "0.7%", "1.0%"],
    "degrade": ["25%", "33%", "50%", "75%"],
}


def load(mode):
    with open("../results/supp_%s_paper.json" % mode, encoding="utf-8") as fh:
        return json.load(fh)


def load_paper():
    p = "../results/paper_digitized.json"
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def fill_none(v):
    v = np.array([np.nan if x is None else x for x in v], float)
    i = np.arange(len(v))
    good = ~np.isnan(v)
    if good.sum() >= 2 and good.sum() < len(v):
        v[~good] = np.interp(i[~good], i[good], v[good])
    return v


def draw(mode, res, paper):
    levels, xlab, yl_n, yl_r, kind, fignum = AXES[mode]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), dpi=200)
    xs = np.arange(len(levels))
    model_names = [m for m in ORDER if m in res["models"]]

    for ax, key, ylim in ((axes[0], "nmse", yl_n), (axes[1], "rmse", yl_r)):
        if kind == "bar":
            w = 0.26
            for j, m in enumerate(model_names):
                vals = [res["models"][m][str(l)][key] for l in levels]
                ax.bar(xs + (j - 1) * w, vals, w, color=COLOR[m], label=m,
                       edgecolor="none")
        else:
            for m in model_names:
                vals = [res["models"][m][str(l)][key] for l in levels]
                ax.plot(xs, vals, "o-", lw=1.9, ms=5, color=COLOR[m], label=m)

        # 论文同轴曲线（有数字化结果时才叠）
        blob = paper.get(fignum)
        if blob and kind == "line":
            pk = "a" if key == "nmse" else "b"
            for m in model_names:
                ser = blob["panels"].get(pk, {}).get("series", {}).get(m)
                if not ser:
                    continue
                pv = fill_none(ser)[:len(levels)]
                ax.plot(xs, pv, "s--", lw=1.1, ms=4.5, mfc="none",
                        color=COLOR[m], alpha=0.85,
                        label="%s (paper)" % m)

        # 论文的纵轴范围是"画的时候用的"，我们的值超出去就自动扩一点，
        # 否则柱子会被截断 —— 截断比范围不一致更容易让人误判。
        hi = max(v for m in model_names
                 for v in [res["models"][m][str(l)][key] for l in levels])
        lo = min(v for m in model_names
                 for v in [res["models"][m][str(l)][key] for l in levels])
        ax.set_ylim(min(ylim[0], lo * 0.92), max(ylim[1], hi * 1.06))
        ax.set_xticks(xs)
        ax.set_xticklabels(XTICK.get(mode, [str(l) for l in levels]), fontsize=9)
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_ylabel(key.upper(), fontsize=9)
        ax.grid(axis="y", alpha=0.25, lw=0.7)
        ax.tick_params(labelsize=8.5)

    axes[0].legend(fontsize=7.2, frameon=False, loc="upper left", ncol=2)
    setup = res.get("setup", "")
    fig.suptitle("Our reproduction - Fig.%s protocol (%s weights, n=%s)"
                 % (fignum, setup, res.get("n", "?")), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = "../figs/cmp_%s.png" % mode
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def main():
    modes = sys.argv[1:] or list(AXES)
    paper = load_paper()
    for mode in modes:
        draw(mode, load(mode), paper)


if __name__ == "__main__":
    main()
