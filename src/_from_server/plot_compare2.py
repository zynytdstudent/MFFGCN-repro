"""Redraw the paper-vs-reproduction figures in a readable form.

Outputs
-------
nmse_compare.png       10 methods x 3 setups, horizontal bars, log scale
generalization_gap.png two panels: seen vs unseen maps, and split sensitivity

Usage (run on the server, matplotlib is installed there):
    python plot_compare2.py --json ../eval_FINAL/reports/comparison.json \
                            --out  ../eval_FINAL/reports
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["KNN", "Kriging", "AE", "DeepAE", "U-Net",
           "RadioUNet", "Pix2pix+", "RME-GAN", "MFFGCN"]
SETUPS = [("s1", "Setup 1\nuniform sampling"),
          ("s2", "Setup 2\n25% restricted"),
          ("s3", "Setup 3\n10-50% restricted")]

C_BASE = "#5B7DB1"
C_PAPER = "#2E7D59"
C_OURS = "#C0392B"


def fig_nmse(data, out_path):
    t3, ours = data["paper_table3"], data["ours"]
    n = len(METHODS) + 1
    y = np.arange(n)[::-1]
    labels = METHODS + ["Ours (reproduced)"]
    colors = [C_BASE] * len(METHODS)
    colors[METHODS.index("MFFGCN")] = C_PAPER
    colors.append(C_OURS)

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.8), sharey=True)
    for ax, (key, title) in zip(axes, SETUPS):
        idx = SETUPS.index((key, title))
        vals = [t3[m][idx][0] for m in METHODS] + [ours[key]["fine"]["all_nmse"]]
        ax.barh(y, vals, color=colors, height=0.72, zorder=3)

        for yi, v in zip(y, vals):
            ax.text(v * 1.09, yi, f"{v:.4f}", va="center", ha="left",
                    fontsize=8.4, color="#333333", zorder=4)

        ax.axvline(t3["MFFGCN"][idx][0], color=C_PAPER, ls="--", lw=1.0,
                   zorder=2, alpha=0.75)
        ax.set_xscale("log")
        ax.set_xlim(0.003, 3.2)
        ax.set_title(title, fontsize=11, color="#222222", pad=10)
        ax.set_xlabel("NMSE (log scale, lower is better)", fontsize=9.5)
        ax.grid(axis="x", which="major", color="#DDDDDD", lw=0.7, zorder=0)
        ax.grid(axis="x", which="minor", color="#F0F0F0", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9.6)
    for lab, m in zip(axes[0].get_yticklabels(), labels):
        if m.startswith("Ours"):
            lab.set_color(C_OURS)
            lab.set_fontweight("bold")

    handles = [plt.Rectangle((0, 0), 1, 1, color=C_BASE),
               plt.Rectangle((0, 0), 1, 1, color=C_PAPER),
               plt.Rectangle((0, 0), 1, 1, color=C_OURS)]
    fig.legend(handles, ["Paper (other methods)", "Paper (MFFGCN)",
                         "Ours (reproduced)"],
               fontsize=9.2, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.015))

    fig.suptitle("RadioMapSeer test split (last 100 city maps, unseen in training)",
                 fontsize=12.5, y=0.975)
    fig.tight_layout(rect=[0, 0.035, 1, 0.945])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_gap(train_nmse, test_nmse, paper_nmse, pair_nmse, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))

    ax = axes[0]
    vals = [train_nmse, test_nmse, paper_nmse]
    names = ["Train maps\n(seen)", "Test maps\n(unseen)", "Paper\nMFFGCN"]
    cols = ["#1D9E75", C_OURS, C_PAPER]
    bars = ax.bar(names, vals, color=cols, width=0.56, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.13, f"{v:.4f}",
                ha="center", fontsize=9.5, color="#333333")
    ax.set_yscale("log")
    ax.set_ylim(0.002, 0.4)
    ax.set_ylabel("NMSE (log scale)", fontsize=9.5)
    ax.set_title("Same weights, two splits (Setup 1)", fontsize=11)
    ax.text(0.5, 0.06, f"{test_nmse / train_nmse:.1f}x gap", transform=ax.transAxes,
            ha="center", fontsize=10, color=C_OURS)
    ax.grid(axis="y", color="#E5E5E5", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    vals = [test_nmse, pair_nmse]
    names = ["Split by map\n(paper protocol)", "Split by sample\n(diagnostic)"]
    cols = [C_OURS, "#8E7CC3"]
    bars = ax.bar(names, vals, color=cols, width=0.5, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.13, f"{v:.4f}",
                ha="center", fontsize=9.5, color="#333333")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 0.4)
    ax.set_ylabel("NMSE (log scale)", fontsize=9.5)
    ax.set_title("Sensitivity to the data split", fontsize=11)
    ax.text(0.5, 0.94, f"{test_nmse / pair_nmse:.1f}x lower, 1/6 of the budget",
            transform=ax.transAxes, ha="center", fontsize=9.6, color="#444441")
    ax.grid(axis="y", color="#E5E5E5", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="../eval_FINAL/reports/comparison.json")
    ap.add_argument("--out", default="../eval_FINAL/reports")
    ap.add_argument("--train-nmse", type=float, default=0.0068222570280554755)
    ap.add_argument("--pair-nmse", type=float, default=0.017207285982028257)
    a = ap.parse_args()

    with open(a.json, encoding="utf-8") as fh:
        data = json.load(fh)
    os.makedirs(a.out, exist_ok=True)

    p1 = fig_nmse(data, os.path.join(a.out, "nmse_compare.png"))
    p2 = fig_gap(a.train_nmse, data["ours"]["s1"]["fine"]["all_nmse"],
                 data["paper_table3"]["MFFGCN"][0][0], a.pair_nmse,
                 os.path.join(a.out, "generalization_gap.png"))
    print("saved:", p1)
    print("saved:", p2)


if __name__ == "__main__":
    main()
