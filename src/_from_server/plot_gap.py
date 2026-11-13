"""画盲区专项分析图：误差随"离盲区边界距离"的变化。

对应论文图 8-11 想说明的问题：误差到底堆在盲区边缘，还是均匀摊在盲区内部。
如果曲线在 0-2 像素处明显更高、往里迅速走平，说明模型只是"估不准边界"，
而不是"盲区里完全没有信息可用"。

数据来自 reeval.py 写出的 bands 字段。

用法：
    python plot_gap.py --in reeval_all.json --out ../reports/fig_gap.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["1-2", "2-4", "4-8", "8-16", "16-32", "32+"]
PRETTY = {
    "FINAL_s2": "MFFGCN (full)",
    "ab_channel_s2": "fusion: sigmoid gate",
    "ab_nogcn_s2": "w/o GCN",
    "ab_nores_s2": "w/o residual",
    "ab_static_s2": "static graph",
    "ab_single_s2": "single encoder",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+",
                    default=["reeval_all.json", "reeval_extra.json"])
    ap.add_argument("--out", default="../reports/fig_gap.png")
    ap.add_argument("--setup", default="s2")
    a = ap.parse_args()

    data = {}
    for path in a.inp:
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for r in json.load(fh):
                data[r["tag"]] = r

    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    plotted = 0
    for tag, label in PRETTY.items():
        r = data.get(tag)
        if not r or not r.get("bands"):
            continue
        xs, ys = [], []
        for i, k in enumerate(ORDER):
            b = r["bands"].get(k)
            if b and b.get("px", 0) > 50:
                xs.append(i + 0.5)
                ys.append(b["rmse"])
        if len(xs) >= 3:
            ax.plot(xs, ys, marker="o", ms=4, lw=1.6, label=label)
            plotted += 1

    if plotted == 0:
        print("[warn] 没有可用的分带数据，请先跑 reeval.py")
        return

    ax.set_xticks([i + 0.5 for i in range(len(ORDER))])
    ax.set_xticklabels(ORDER)
    ax.set_xlabel("distance from nearest visible pixel (px)")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.3, ls="--")
    ax.set_title("Error vs depth inside the blind region (Setup 2)")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=160)
    print("[ok] wrote", a.out, f"({plotted} 条曲线)")


if __name__ == "__main__":
    main()
