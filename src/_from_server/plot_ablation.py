"""Ablation figure: paper Table V vs our reproduction, one panel per setup.

Reads ../eval_ALL/reports/comparison_all.json produced by compare_all.py.

Usage:
    python plot_ablation.py --json ../eval_ALL/reports/comparison_all.json \
                            --out  ../eval_ALL/reports
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_PAPER = "#2E7D59"
C_OURS = "#C0392B"

ORDER = ["single", "add", "concat", "channel", "cross", "nogcn", "static", "nores"]
SETUPS = ["s1", "s2", "s3"]
SETUP_TITLE = ["Setup 1 (uniform)", "Setup 2 (25% restricted)", "Setup 3 (10-50% restricted)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="../eval_ALL/reports/comparison_all.json")
    ap.add_argument("--out", default="../eval_ALL/reports")
    a = ap.parse_args()

    with open(a.json, encoding="utf-8") as fh:
        d = json.load(fh)
    abl, base, t5 = d["ablation"], d["main"], d["paper_table5"]

    rows = [k for k in ORDER if k in abl]
    labels = [t5[k][0] for k in rows] + ["Proposed (full model)"]
    y = np.arange(len(labels))[::-1]
    w = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), sharey=True)
    for j, (ax, s) in enumerate(zip(axes, SETUPS)):
        ours, paper = [], []
        for k in rows:
            m = abl[k].get(s)
            ours.append(m["fine"]["all_nmse"] if m else np.nan)
            paper.append(t5[k][1][j][0])
        m = base.get(s)
        ours.append(m["fine"]["all_nmse"] if m else np.nan)
        paper.append(d["paper_table3"]["MFFGCN"][j][0])

        ax.barh(y + w / 2, paper, w, color=C_PAPER, label="Paper (Table V)", zorder=3)
        ax.barh(y - w / 2, ours, w, color=C_OURS, label="Ours", zorder=3)
        for yi, v in zip(y - w / 2, ours):
            if v == v:
                ax.text(v * 1.06, yi, f"{v:.4f}", va="center", fontsize=7.6, color="#333333")
        ax.set_xscale("log")
        ax.set_xlim(0.003, 0.9)
        ax.set_title(SETUP_TITLE[j], fontsize=11)
        ax.set_xlabel("NMSE (log scale)", fontsize=9)
        ax.grid(axis="x", color="#E8E8E8", lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9.4)
    for lab in axes[0].get_yticklabels()[-1:]:
        lab.set_fontweight("bold")
        lab.set_color(C_OURS)
    axes[0].legend(fontsize=8.8, frameon=False, loc="lower right")

    fig.suptitle("Table V ablation: paper vs reproduction (same test split)",
                 fontsize=12.5, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "ablation_compare.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
