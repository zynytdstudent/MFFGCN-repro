"""Depth profile inside the blind region, one figure per sampling setup.

This is the per-setup version of plot_gap.py. The old script hardcoded the
Setup-2 tags of a single round of runs, so only Setup 1 ever got a figure;
here the run is picked by `txf_mffgcn_<setup>` out of reeval_gap.json, which
is produced by:

    python reeval.py --tags txf_mffgcn_s1 txf_mffgcn_s2 txf_mffgcn_s3 \
                     --out reeval_gap.json

The x axis is the distance from the nearest observed pixel (in pixels, EDT),
so the leftmost bin is "right at the edge of the blind region" and the right
side is "deep inside".  A curve that is high on the left and flat on the
right means the model only struggles at the boundary; the opposite means the
blind interior really is information-free.

Usage:
    python plot_gap_setup.py --setup s2 --out ../figs/fig_gap_s2.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["0-1", "1-2", "2-4", "4-8", "8-16", "16-32", "32+"]
TITLE = {
    "s1": "Setup 1 (uniform sampling)",
    "s2": "Setup 2 (25% of the map restricted)",
    "s3": "Setup 3 (10-50% restricted, uneven)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="reeval_gap.json")
    ap.add_argument("--setup", default="s2")
    ap.add_argument("--tag", default=None,
                    help="默认 txf_mffgcn_<setup>")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-px", type=int, default=200,
                    help="像素数少于此值的分带不画（统计噪声太大）")
    a = ap.parse_args()

    tag = a.tag or ("txf_mffgcn_" + a.setup)
    out = a.out or ("../figs/fig_gap_%s.png" % a.setup)

    with open(a.inp) as fh:
        runs = {r["tag"]: r for r in json.load(fh)}
    r = runs.get(tag)
    if not r:
        raise SystemExit("[err] %s not in %s (have: %s)"
                         % (tag, a.inp, ", ".join(sorted(runs))))

    bands = r.get("bands") or {}
    xs, ys, ns = [], [], []
    for i, k in enumerate(ORDER):
        b = bands.get(k)
        if b and b.get("px", 0) >= a.min_px:
            xs.append(i + 0.5)
            ys.append(b["rmse"])
            ns.append(b["px"])

    if len(xs) < 2:
        raise SystemExit("[err] not enough bands for %s (%d)" % (tag, len(xs)))

    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(xs, ys, marker="o", ms=5, lw=1.8, color="#C0392B",
            label=tag.replace("txf_", ""))
    for x, y, n in zip(xs, ys, ns):
        ax.annotate("%.4f" % y, (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.6, color="#333333")
    ax.set_xticks([i + 0.5 for i in range(len(ORDER))])
    ax.set_xticklabels(ORDER)
    ax.set_xlabel("distance from nearest observed pixel (px)")
    ax.set_ylabel("RMSE inside the blind region")
    ax.set_title(TITLE.get(a.setup, a.setup)
                 + "  |  error vs depth in the blind region", fontsize=11.5)
    ax.grid(alpha=0.3, ls="--")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out, dpi=170)
    print("[ok] wrote %s (%d bands, constr_nmse=%.5f)"
          % (out, len(xs), r["fine"].get("constr_nmse", float("nan"))))


if __name__ == "__main__":
    main()
