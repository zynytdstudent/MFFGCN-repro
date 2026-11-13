"""Robustness curves: NMSE against measurement noise and sensor budget.

Reads metrics_rob_*.json written by robustness.sh and renders two panels.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_MAIN = "#c0392b"
C_GRID = "#d9d9d9"


def load(d, name):
    p = os.path.join(d, f"metrics_{name}.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def nm(d):
    return None if d is None else d["fine"]["all_nmse"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../eval_ALL")
    ap.add_argument("--out", default="../eval_ALL/reports")
    a = ap.parse_args()

    clean = nm(load(a.dir, "rob_clean"))
    if clean is None:
        raise SystemExit("rob_clean.json missing - run robustness.sh first")

    snrs = [30, 20, 10]
    snr_y = [nm(load(a.dir, f"rob_snr{s}")) for s in snrs]
    bud_k = [("rob_bud0p5", 0.5), ("rob_clean", 1.0),
             ("rob_bud2", 2.0), ("rob_bud4", 4.0)]
    bud_x = [k for _, k in bud_k]
    bud_y = [nm(load(a.dir, n)) for n, _ in bud_k]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    xs = [40] + snrs                       # 40 dB stands in for "clean"
    ys = [clean] + [v if v else float("nan") for v in snr_y]
    ax.plot(xs, ys, "-o", color=C_MAIN, lw=1.8, ms=6, zorder=3)
    for x, y in zip(xs, ys):
        if y == y:
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8.4)
    ax.set_xlabel("measurement SNR (dB)   -   40 dB = clean")
    ax.set_ylabel("test NMSE (Setup 1)")
    ax.set_title("A. Sensitivity to measurement noise", fontsize=11, loc="left")
    ax.invert_xaxis()
    ax.margins(x=0.08, y=0.20)
    ax.grid(alpha=0.35, color=C_GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    ys = [v if v else float("nan") for v in bud_y]
    ax.plot(bud_x, ys, "-o", color=C_MAIN, lw=1.8, ms=6, zorder=3)
    for x, y in zip(bud_x, ys):
        if y == y:
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8.4)
    ax.axvline(1.0, color="#7f8c8d", ls=":", lw=1.2)
    ax.annotate("paper budget\n(10-300 sensors)", (1.0, ax.get_ylim()[1]),
                xytext=(6, -14), textcoords="offset points",
                fontsize=8.2, color="#7f8c8d", va="top")
    ax.set_xlabel("sensor budget multiplier (x paper setting)")
    ax.set_ylabel("test NMSE (Setup 1)")
    ax.margins(x=0.08, y=0.20)
    ax.set_title("B. Sensitivity to the number of sensors", fontsize=11, loc="left")
    ax.grid(alpha=0.35, color=C_GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("Robustness of the reproduced MFFGCN on unseen cities",
                 fontsize=12.5, y=1.00)
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "robustness.png")
    fig.savefig(p, dpi=180, bbox_inches="tight")
    print(f"[ok] {p}")

    print(f"{'condition':16s} {'NMSE':>8s}")
    print(f"{'clean':16s} {clean:8.4f}")
    for s, v in zip(snrs, snr_y):
        if v:
            print(f"{'SNR %d dB' % s:16s} {v:8.4f}  ({v/clean:.2f}x)")
    for (n, k), v in zip(bud_k, bud_y):
        if v:
            print(f"{'budget x%.1f' % k:16s} {v:8.4f}  ({v/clean:.2f}x)")


if __name__ == "__main__":
    main()
