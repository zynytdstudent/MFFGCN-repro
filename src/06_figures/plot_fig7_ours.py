# -*- coding: utf-8 -*-
"""Fig.7 复现图（独立成图，形式与论文一致，不与论文叠画）。

论文 Fig.7：Setup 1 下不同观测样本数对 7 个方法的影响，
左 (a) NMSE、右 (b) RMSE，x 轴 50–300（步长 50），图例 7 条放右上角，
y 轴对数刻度。

本图只画**我们的复现**。论文原图截图在对照页左半边，两者不叠在一张图上。

数据来源：results/fig7_obs_sweep.json（fig7_obs_sweep.py 产出，test 全集、
每图观测点数固定为 n，权重不动、只改推理时的观测点数量）。

用法： python plot_fig7_ours.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
# 脚本在本地是 src/06_figures/，在服务器上被平铺到 src/，两种都要能找到根目录
ROOT = HERE
for _rel in ("..", os.path.join("..", "..")):
    _c = os.path.abspath(os.path.join(HERE, _rel))
    if os.path.isdir(os.path.join(_c, "results")):
        ROOT = _c
        break
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figs")

# 顺序、颜色、标记按论文 Fig.7 的观感排（AE 蓝、DeepAE 橙、UNet 绿、
# RadioUNet 红、Pix2pix+ 紫、RME-GAN 棕、MFFGCN 深蓝）
STYLE = [
    ("AE",        "#1f77b4", "o", 1.3),
    ("DeepAE",    "#ff7f0e", "s", 1.3),
    ("UNet",      "#2ca02c", "+", 1.3),
    ("RadioUNet", "#d62728", "x", 1.3),
    ("Pix2pix+",  "#9467bd", "*", 1.3),
    ("RME-GAN",   "#8c564b", "d", 1.2),
    ("MFFGCN",    "#1c1ca8", ">", 2.4),
]


def main():
    d = json.load(open(os.path.join(RES, "fig7_obs_sweep.json"),
                       encoding="utf-8"))
    counts = d["counts"]
    curves = {k.replace("U-Net", "UNet").replace("U-Net", "UNet"): v
              for k, v in d["curves"].items()}
    got = [n for n, _, _, _ in STYLE if n in curves]
    miss = [n for n, _, _, _ in STYLE if n not in curves]
    print("有数据：", got)
    if miss:
        print("缺数据：", miss)

    plt.rcParams.update({"font.family": "DejaVu Sans",
                         "mathtext.fontset": "dejavusans"})
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.1))

    for ax, key, tag, ylab in ((axes[0], "nmse", "(a)", "NMSE"),
                               (axes[1], "rmse", "(b)", "RMSE")):
        for name, color, mk, lw in STYLE:
            if name not in curves:
                continue
            y = curves[name][key]
            ax.plot(counts, y, "-", color=color, lw=lw, marker=mk,
                    ms=4.2 if lw < 2 else 5.2, mfc="none", mew=1.1,
                    label=name, zorder=3 if lw > 2 else 2)
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(LogLocator(subs="auto", numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(35, 315)
        # 给右上角图例留出空间：论文的曲线陡降，图例压不到线；我们的
        # DeepAE 几乎水平且在最上面，必须抬高上界，让最高曲线落在
        # 坐标区的 66% 处（对数轴上算），图例才不遮线。
        _lo = min(float(np.min(curves[m][key])) for m in got)
        _hi = max(float(np.max(curves[m][key])) for m in got)
        _bot = _lo * 0.72
        # 图例高度随条目数增长，最高曲线允许占到的坐标区高度也要跟着降：
        # 7 条时压到 58%，6 条时 63%，图例才不会压线。
        _frac = max(0.52, 0.95 - 0.055 * len(got))
        _top = 10 ** (np.log10(_bot) + (np.log10(_hi) - np.log10(_bot)) / _frac)
        ax.set_ylim(_bot, _top)
        ax.set_xticks(counts)
        ax.set_xlabel("Number of observation samples", fontsize=10.5)
        # 口径：x 轴是每个测试图给的观测点数（不是训练样本量），
        # 与论文 paper_text.txt:854 的 "observation samples" 对齐（2026-11-12 纠正）
        ax.set_ylabel(ylab, fontsize=10.5)
        ax.tick_params(which="both", labelsize=9)
        ax.grid(alpha=0.20, lw=0.6, which="major")
        ax.legend(fontsize=8.0, loc="upper right", frameon=True,
                  handlelength=1.9, borderpad=0.42, labelspacing=0.26)
        ax.text(0.5, -0.20, tag, transform=ax.transAxes, ha="center",
                fontsize=14)

    fig.tight_layout(w_pad=2.6)
    out = os.path.join(FIG, "fig7_ours.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)

    # 把"图上实际用的这组数"同步写成 figs/fig7.json。
    # 2026-11-12 补：这个 json 之前一直停在 11-11 的**训练样本量**口径
    # （由已废弃的 paper_fig_7.py 写入，AE 起点 0.175），而 png 早就换成
    # 观测点数口径了，两者对不上——谁去读 json 谁拿到错数。现在由绘图
    # 脚本自己写，杜绝 png 与 json 再次分叉。
    dump = {name: {"counts": list(counts),
                   "nmse": curves[name]["nmse"],
                   "rmse": curves[name]["rmse"]} for name in got}
    jout = os.path.join(FIG, "fig7.json")
    with open(jout, "w", encoding="utf-8") as fh:
        json.dump(dump, fh, ensure_ascii=False, indent=1)
    print("saved", jout)

    # 单调性自检：论文说所有曲线都随样本数下降
    for name in got:
        for key in ("nmse", "rmse"):
            y = np.asarray(curves[name][key], float)
            if not np.all(np.diff(y) < 0):
                print("  [!] %s %s 非单调：%s" % (name, key,
                                                  np.round(y, 5).tolist()))
    print("单调性检查完成")


if __name__ == "__main__":
    main()
