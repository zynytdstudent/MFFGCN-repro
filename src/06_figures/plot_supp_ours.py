# -*- coding: utf-8 -*-
"""Fig.9/10/14/15/16 的复现图（各自独立成图，格式照论文，不与论文叠画）。

每张图只画我们的三个模型（RadioUNet / RME-GAN / MFFGCN），
配色、线型、轴名、图例位置都按论文原图排，方便左右并排看。

数据来源：results/supp_{dist,side,noise,degrade,missing}_paper.json
（supp_eval.py --span --paper-grid 产出，档位与论文横轴一致）。

用法： python plot_supp_ours.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
for _rel in ("..", os.path.join("..", "..")):
    _c = os.path.abspath(os.path.join(HERE, _rel))
    if os.path.isdir(os.path.join(_c, "results")):
        ROOT = _c
        break
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figs")

# 论文三色：RadioUNet 蓝、RME-GAN 橙、MFFGCN 绿（论文里 MFFGCN 标注为 proposed）
STYLE = {
    "RadioUNet": ("#1f77b4", "o", "RadioUNet"),
    "RME-GAN":   ("#ff7f0e", "+", "RME-GAN"),
    "MFFGCN":    ("#2ca02c", "x", "MFFGCN(proposed)"),
}
ORDER = ["RadioUNet", "RME-GAN", "MFFGCN"]


def load(mode):
    p = os.path.join(RES, "supp_%s_paper.json" % mode)
    return json.load(open(p, encoding="utf-8"))


def series(d, model, key):
    return [d["models"][model][lv][key] for lv in d["models"][model]]


def levels(d):
    return list(d["models"]["MFFGCN"].keys())


def bars_out(d, ticks, xlabel, fname, ylim_top=None, legend_loc="upper right"):
    """柱状图版（Fig.9 / Fig.10 / Fig.16 在论文里都是分组柱）。"""
    xs = np.arange(len(ticks))
    w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for ax, key, ylab, tag in ((axes[0], "nmse", "NMSE", "(a)"),
                               (axes[1], "rmse", "RMSE", "(b)")):
        for j, m in enumerate(ORDER):
            color, _, lab = STYLE[m]
            ax.bar(xs + (j - 1) * w, series(d, m, key), w, color=color,
                   label=lab)
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks, fontsize=9.5)
        ax.set_xlim(-0.55, len(ticks) - 0.45)
        top = ylim_top or max(max(series(d, m, key)) for m in ORDER) * 1.30
        ax.set_ylim(0, top)
        ax.tick_params(labelsize=9.5)
        ax.set_xlabel(xlabel, fontsize=10.5)
        ax.set_ylabel(ylab, fontsize=10.5)
        ax.legend(fontsize=8.8, loc=legend_loc, frameon=True,
                  handlelength=1.5, borderpad=0.5)
        ax.grid(axis="y", alpha=0.20, lw=0.6)
        ax.text(0.5, -0.19, tag, transform=ax.transAxes, ha="center",
                fontsize=13.5)
    fig.tight_layout(w_pad=2.6)
    out = os.path.join(FIG, fname)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def lines_out(d, xticks, xlabel, fname, logy=False, legend_loc="upper left"):
    """折线图版（Fig.14 / Fig.15 在论文里是折线）。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for ax, key, ylab, tag in ((axes[0], "nmse", "NMSE", "(a)"),
                               (axes[1], "rmse", "RMSE", "(b)")):
        for m in ORDER:
            color, mk, lab = STYLE[m]
            ax.plot(xticks, series(d, m, key), "-", color=color, lw=1.5,
                    marker=mk, ms=5.0, mew=1.2, label=lab)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(xticks)
        ax.tick_params(labelsize=9.5)
        ax.set_xlabel(xlabel, fontsize=10.5)
        ax.set_ylabel(ylab, fontsize=10.5)
        ax.legend(fontsize=8.8, loc=legend_loc, frameon=True,
                  handlelength=2.0, borderpad=0.5)
        ax.grid(alpha=0.20, lw=0.6)
        ax.text(0.5, -0.19, tag, transform=ax.transAxes, ha="center",
                fontsize=13.5)
    fig.tight_layout(w_pad=2.6)
    out = os.path.join(FIG, fname)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def check(d, name):
    """自检：三角色共用同一个 E[t^2]，说明三组结果同分母、可横向比较。"""
    vals = [d["models"][m][lv]["rmse"] ** 2 / d["models"][m][lv]["nmse"]
            for m in ORDER for lv in levels(d)]
    print("  %-8s E[t^2] = %.5f ± %.1e" % (name, np.mean(vals),
                                           max(vals) - min(vals)))


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    # ---- Fig.9 盲区离图中心的距离（柱） ----
    d = load("dist")
    check(d, "fig9")
    bars_out(d, ["50", "70", "90", "110"], "Distance to center",
             "fig9_ours.png")

    # ---- Fig.10 盲区边长（柱） ----
    d = load("side")
    check(d, "fig10")
    bars_out(d, ["90", "110", "130", "150"], "Side length", "fig10_ours.png")

    # ---- Fig.14 冲激噪声（折线，横轴 %，论文从 0.1% 起） ----
    d = load("noise")
    check(d, "fig14")
    lv = levels(d)                      # ['0.0','0.001','0.004','0.007','0.01']
    keep = [i for i, v in enumerate(lv) if float(v) > 0]
    d14 = {"models": {m: {lv[i]: d["models"][m][lv[i]] for i in keep}
                      for m in ORDER}}
    lines_out(d14, [0.1, 0.4, 0.7, 1.0], "Noise intensity (%)",
              "fig14_ours.png", legend_loc="upper left")

    # ---- Fig.15 环境图降分辨率（折线，横轴 25/33/50/75%） ----
    d = load("degrade")
    check(d, "fig15")
    lv = levels(d)                      # ['256','192','128','85','64']
    pick = ["64", "85", "128", "192"]   # 25% / 33% / 50% / 75%
    d15 = {"models": {m: {k: d["models"][m][k] for k in pick}
                      for m in ORDER}}
    lines_out(d15, [25, 33, 50, 75], "Downsampling ratio (%)",
              "fig15_ours.png", legend_loc="upper right")

    # ---- Fig.16 缺失建筑（柱，1–4 栋） ----
    d = load("missing")
    check(d, "fig16")
    d16 = {"models": {m: {k: d["models"][m][k] for k in ("1", "2", "3", "4")}
                      for m in ORDER}}
    bars_out(d16, ["1", "2", "3", "4"], "Number of missing buildings",
             "fig16_ours.png", legend_loc="upper left")


if __name__ == "__main__":
    main()
