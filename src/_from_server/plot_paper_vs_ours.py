"""Paper vs reproduction, side by side.

Three panels:
  A  grouped bars, NMSE per setup (log scale)      -> absolute gap
  B  line chart, NMSE vs setup                     -> degradation slope
  C  grouped bars, RMSE per setup (linear)         -> gap in physical units

Usage (server has matplotlib):
    python plot_paper_vs_ours.py --json ../eval_FINAL/reports/comparison.json \
                                 --out  ../eval_FINAL/reports
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
C_BASE = "#5B7DB1"
C_BASE2 = "#8FA8C8"
C_SEEN = "#1D9E75"

SETUPS = ["Setup 1\nuniform", "Setup 2\n25% restricted", "Setup 3\n10-50% restricted"]
SHORT = ["Setup 1", "Setup 2", "Setup 3"]


def panel_bars(ax, paper, ours, ylab, title, logy, fmt, note=None):
    x = np.arange(len(paper))
    w = 0.36
    b1 = ax.bar(x - w / 2, paper, w, color=C_PAPER, label="Paper MFFGCN", zorder=3)
    b2 = ax.bar(x + w / 2, ours, w, color=C_OURS, label="Ours (reproduced)", zorder=3)
    for b, v in list(zip(b1, paper)) + list(zip(b2, ours)):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8.6, color="#333333")
    for xi, p, o in zip(x, paper, ours):
        ax.annotate(f"{o/p:.1f}x", (xi, max(p, o) * (1.9 if logy else 1.12)),
                    ha="center", fontsize=9, color=C_OURS, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(SETUPS, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9.5)
    ax.set_title(title, fontsize=11)
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(min(paper) * 0.55, max(ours) * 4.2)
    else:
        ax.set_ylim(0, max(ours) * 1.35)
    ax.grid(axis="y", color="#E5E5E5", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8.8, frameon=False, loc="upper left")
    if note:
        ax.text(0.5, 0.02, note, transform=ax.transAxes, ha="center",
                fontsize=8.6, color="#555555")


def panel_curve(ax, t3, ours_nmse):
    x = np.arange(3)
    series = [("Paper MFFGCN", [t3["MFFGCN"][i][0] for i in range(3)], C_PAPER, "o", 2.4),
              ("Paper U-Net", [t3["U-Net"][i][0] for i in range(3)], C_BASE, "s", 1.6),
              ("Paper AE", [t3["AE"][i][0] for i in range(3)], C_BASE2, "^", 1.6),
              ("Ours (reproduced)", ours_nmse, C_OURS, "D", 2.4)]
    for name, vals, col, mk, lw in series:
        ax.plot(x, vals, marker=mk, lw=lw, ms=6.5, color=col, label=name,
                zorder=4 if name.startswith("Ours") else 3)
        growth = (vals[2] / vals[0] - 1) * 100
        ax.annotate(f"{growth:+.1f}%", (x[2], vals[2]),
                    textcoords="offset points", xytext=(8, -2),
                    fontsize=8.8, color=col, va="center")
    ax.set_yscale("log")
    ax.set_ylim(0.0033, 0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, fontsize=9)
    ax.set_xlim(-0.25, 2.55)
    ax.set_ylabel("NMSE (log scale)", fontsize=9.5)
    ax.set_title("Degradation from Setup 1 to Setup 3", fontsize=11)
    ax.grid(axis="y", color="#E5E5E5", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8.6, frameon=False, loc="upper left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="../eval_FINAL/reports/comparison.json")
    ap.add_argument("--out", default="../eval_FINAL/reports")
    ap.add_argument("--train-nmse", type=float, default=0.0068222570280554755)
    a = ap.parse_args()

    with open(a.json, encoding="utf-8") as fh:
        d = json.load(fh)
    t3 = d["paper_table3"]
    ours_nmse = [d["ours"][s]["fine"]["all_nmse"] for s in ("s1", "s2", "s3")]
    ours_rmse = [d["ours"][s]["fine"]["all_rmse"] for s in ("s1", "s2", "s3")]
    paper_nmse = [t3["MFFGCN"][i][0] for i in range(3)]
    paper_rmse = [t3["MFFGCN"][i][1] for i in range(3)]

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.0))

    panel_bars(axes[0], paper_nmse, ours_nmse, "NMSE (log scale, lower is better)",
               "A. NMSE per sampling setup", True, "{:.4f}")
    h, l = axes[0].get_legend_handles_labels()
    axes[0].axhline(a.train_nmse, color=C_SEEN, ls=":", lw=1.6, zorder=2,
                    label=f"ours on seen maps ({a.train_nmse:.4f})")
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h, l, fontsize=8.4, frameon=False, loc="upper left")

    panel_curve(axes[1], t3, ours_nmse)

    panel_bars(axes[2], paper_rmse, ours_rmse, "RMSE (lower is better)",
               "C. RMSE per sampling setup", False, "{:.4f}")

    fig.suptitle("MFFGCN reproduction vs paper: same test split (last 100 city maps, unseen)",
                 fontsize=12.5, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(a.out, "paper_vs_ours.png")
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
