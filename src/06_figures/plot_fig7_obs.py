# -*- coding: utf-8 -*-
"""plot_fig7_obs.py - 论文 Fig.7（观测点数扫描）与我们复现的同轴对照。

输入
----
results/fig7_obs_sweep.json      我们的 7 模型 × n∈{50..300} 实测
results/paper_fig7_digitized.json 从 paper_fig07.png 反解出的论文曲线

输出
----
figs/fig7_obs_compare.png   两联图：(a) 同轴叠加，(b) 相对偏差 vs n
results/fig7_obs_vs_paper.csv 逐点比值

用法：python plot_fig7_obs.py [--ours ...] [--paper ...] [--out ...]
"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 与论文 Fig.7 图例一致的取色（tab10 + 纯蓝）
COLOR = {
    "AE": "#1f77b4",
    "DeepAE": "#ff7f0e",
    "U-Net": "#2ca02c",
    "RadioUNet": "#d62728",
    "Pix2pix+": "#9467bd",
    "RME-GAN": "#8c564b",
    "MFFGCN": "#0000ff",
}
ORDER = ["AE", "DeepAE", "U-Net", "RadioUNet", "Pix2pix+", "RME-GAN", "MFFGCN"]
# 数字化脚本里 U-Net 存成 "UNet"，这里统一
ALIAS = {"UNet": "U-Net"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="../results/fig7_obs_sweep.json")
    ap.add_argument("--paper", default="../results/paper_fig7_digitized.json")
    ap.add_argument("--out", default="../figs/fig7_obs_compare.png")
    ap.add_argument("--csv", default="../results/fig7_obs_vs_paper.csv")
    a = ap.parse_args()

    ours = json.load(open(a.ours, encoding="utf-8"))["curves"]
    pap = json.load(open(a.paper, encoding="utf-8"))
    ns = np.array(pap["counts"], float)
    pser = {ALIAS.get(k, k): np.array(v["nmse"], float)
            for k, v in pap["series"].items()}
    oser = {k: np.array(v["nmse"], float) for k, v in ours.items()}
    common = [m for m in ORDER if m in pser and m in oser
              and not np.isnan(oser[m]).any()]

    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.6))

    # ---- (a) 同轴叠加：论文细虚线 + 我们粗实线 ----
    for m in ORDER:
        c = COLOR[m]
        if m in pser:
            ax[0].plot(ns, pser[m], "--", color=c, lw=1.3, alpha=.8)
        if m in oser and not np.isnan(oser[m]).any():
            ax[0].plot(ns, oser[m], "-o", color=c, lw=1.8, ms=4, label=m)
    ax[0].set_yscale("log")
    ax[0].set_ylim(0.0028, 0.11)
    ax[0].set_xlabel("Number of observation samples")
    ax[0].set_ylabel("NMSE")
    ax[0].set_title("(a) NMSE vs #observations   dashed = paper, solid = ours")
    ax[0].grid(alpha=.25, which="both")
    # AE 没有权重（bl_ae_s1/best.pt 已丢），只有论文曲线，补一个图例代理
    from matplotlib.lines import Line2D
    h, l = ax[0].get_legend_handles_labels()
    h.append(Line2D([0], [0], ls="--", color=COLOR["AE"], lw=1.3))
    l.append("AE (paper)")
    ax[0].legend(h, l, fontsize=7.5, ncol=4, loc="upper center",
                 bbox_to_anchor=(0.5, -0.22), frameon=False)

    # ---- (b) 相对偏差（DeepAE 出界，单独标注）----
    for m in common:
        if m == "DeepAE":
            continue
        r = oser[m] / pser[m]
        ax[1].plot(ns, (r - 1) * 100, "-o", color=COLOR[m], lw=1.6, ms=4,
                   label=m)
    ax[1].axhline(0, color="k", lw=1)
    ax[1].axhspan(-5, 5, color="gray", alpha=.15, lw=0)
    if "DeepAE" in common:
        d = oser["DeepAE"] / pser["DeepAE"]
        ax[1].text(ns[0], 175, f"DeepAE: {100*(d[0]-1):+.0f}% → "
                              f"{100*(d[-1]-1):+.0f}%  (off scale)",
                   fontsize=8, color=COLOR["DeepAE"], va="top")
    ax[1].set_ylim(-50, 200)
    ax[1].set_xlabel("Number of observation samples")
    ax[1].set_ylabel("(ours / paper − 1) × 100%")
    ax[1].set_title("(b) Relative deviation of NMSE")
    ax[1].grid(alpha=.25)
    ax[1].legend(fontsize=7.5, ncol=3, loc="upper center",
                 bbox_to_anchor=(0.5, -0.22), frameon=False)

    fig.tight_layout()
    fig.savefig(a.out, dpi=160)
    print("saved", a.out)

    with open(a.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n", "paper_nmse", "ours_nmse", "rel_dev"])
        for m in common:
            for i, n in enumerate(ns):
                w.writerow([m, int(n), f"{pser[m][i]:.5f}",
                            f"{oser[m][i]:.5f}",
                            f"{oser[m][i] / pser[m][i] - 1:+.4f}"])
    print("saved", a.csv)

    # 终端摘要
    print(f"\n{'model':10s}" + "".join(f"{int(n):>9d}" for n in ns))
    for m in common:
        print(f"{m:10s}" + "".join(f"{(oser[m][i]/pser[m][i]-1)*100:+8.1f}%"
                                   for i in range(len(ns))))
    print(f"\n{'paper':10s}" + "".join(f"{v:9.4f}" for v in
                                       pser["MFFGCN"]))
    print(f"{'ours':10s}" + "".join(f"{v:9.4f}" for v in oser["MFFGCN"]))


if __name__ == "__main__":
    main()
