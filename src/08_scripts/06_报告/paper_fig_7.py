"""Fig.7 replica: NMSE / RMSE versus number of training samples.

Reads runs/fs_<model>_c<N>/test_metrics.json produced by autopilot4.sh and
draws the two panels with the same axes as the paper (x 50-300, NMSE on a
log scale, RMSE linear).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


# Only these two were retrained over the subset sweep; the other five
# baselines of the paper figure need 20 more runs and are left as a gap.
CURVES = [
    ("MFFGCN", "fs_mffgcn_c%d", "#0000cc", "s"),
    ("RadioUNet", "fs_radiounet_c%d", "#e0245e", "x"),
]

COUNTS = [50, 100, 200, 300]


def load(runs, pat, c):
    p = os.path.join(runs, pat % c, "test_metrics.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d.get("nmse"), d.get("rmse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="/tmp/pfig7")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = {}
    for name, pat, color, marker in CURVES:
        xs, nm, rm = [], [], []
        for c in COUNTS:
            v = load(args.runs, pat, c)
            if v is None or v[0] is None:
                print("[skip] %s c%d" % (name, c))
                continue
            xs.append(c)
            nm.append(v[0])
            rm.append(v[1])
        if xs:
            data[name] = (xs, nm, rm, color, marker)
            print("[load] %-10s %s" % (name, list(zip(xs, nm, rm))))

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.2), dpi=200)

    ax = axes[0]
    for name, (xs, nm, rm, color, marker) in data.items():
        ax.plot(xs, nm, marker=marker, ms=5, lw=1.4, color=color, label=name)
    ax.set_yscale("log")
    ax.set_xlim(30, 320)
    ax.set_ylim(4e-3, 1.5e-1)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Number of samples", fontsize=11)
    ax.set_ylabel("NMSE", fontsize=11)
    ax.set_xticks(COUNTS)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    for name, (xs, nm, rm, color, marker) in data.items():
        ax.plot(xs, rm, marker=marker, ms=5, lw=1.4, color=color, label=name)
    ax.set_xlim(30, 320)
    ax.set_ylim(1.6e-2, 4.2e-2)
    ax.set_xlabel("Number of samples", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_xticks(COUNTS)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = os.path.join(args.out, "fig7.png")
    fig.savefig(out)
    plt.close(fig)
    print("[save]", out)

    with open(os.path.join(args.out, "fig7.json"), "w") as f:
        json.dump({k: {"counts": v[0], "nmse": v[1], "rmse": v[2]}
                   for k, v in data.items()}, f, indent=1)


if __name__ == "__main__":
    main()
