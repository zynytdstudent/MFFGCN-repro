"""Bar chart: paper TABLE III versus our reproduction (NMSE, log scale)."""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare import PAPER_TABLE3, SETUP_NAME

OURS = "Ours (reproduced)"
METHODS = ["Kriging", "AE", "DeepAE", "U-Net", "RadioUNet", "RME-GAN", "MFFGCN"]
SETUPS = ["s1", "s2", "s3"]


def load_ours(run_dir):
    out = {}
    for p in glob.glob(os.path.join(run_dir, "metrics_*.json")):
        m = json.load(open(p))
        s = m.get("setup") or os.path.basename(p)[8:-5]
        fine = m["fine"] if "fine" in m else {"all_nmse": m["nmse"],
                                              "all_rmse": m["rmse"]}
        out[s] = fine
    return out


def main(run_dir, out_png):
    ours = load_ours(run_dir)
    names = METHODS + [OURS]
    x = np.arange(len(SETUPS))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, name in enumerate(names):
        vals = []
        for s in SETUPS:
            if name == OURS:
                v = ours.get(s, {}).get("all_nmse")
            else:
                v = PAPER_TABLE3[name][SETUPS.index(s)][0]
            vals.append(v if v else np.nan)
        col = "#c0392b" if name == OURS else "#5b7fbd"
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=name, color=col,
                      edgecolor="none")
        if name == OURS:
            for b, v in zip(bars, vals):
                if v and not np.isnan(v):
                    ax.annotate(f"{v:.4f}", (b.get_x() + b.get_width() / 2, v),
                                textcoords="offset points", xytext=(0, 3),
                                ha="center", fontsize=7, color="#c0392b")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([SETUP_NAME[s] for s in SETUPS], fontsize=9)
    ax.set_ylabel("NMSE (log scale)")
    ax.set_title("TABLE III reproduction: NMSE per method and sampling setup")
    ax.grid(axis="y", ls=":", alpha=0.45)
    ax.legend(fontsize=7, ncol=4, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("saved", out_png)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="../eval_FINAL")
    ap.add_argument("--out", default="../eval_FINAL/reports/nmse_compare.png")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    main(a.runs, a.out)
