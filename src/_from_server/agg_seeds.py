"""Aggregate test metrics over seeds -> mean +- std, ready for the paper table.

Groups every runs/<tag>/test_metrics.json by the tag with the trailing
`_seed<N>` stripped, so txf_single_s1 and txf_single_s1_seed1..4 collapse into
one row.  Paper reference values (Table III / V, Setup 1, NMSE / RMSE):

    U-Net (single encoder)   0.0182 RMSE
    MFFGCN                   0.0043 NMSE / 0.0150 RMSE
    MFFGCN w/o Residual      0.0231 NMSE / 0.0347 RMSE
"""
import glob
import json
import os
import re
import statistics as st

PAPER = {
    "txf_single_s1": "RMSE 0.0182",
    "txf_mffgcn_s1": "NMSE 0.0043 / RMSE 0.0150",
}

runs = {}
for p in glob.glob("runs/*/test_metrics.json"):
    tag = os.path.basename(os.path.dirname(p))
    try:
        m = json.load(open(p))
    except Exception:
        continue
    m["tag"] = tag
    m["seed_suffix"] = bool(re.search(r"_seed\d+$", tag))
    base = re.sub(r"_seed\d+$", "", tag)
    runs.setdefault(base, []).append(m)


def fmt(vals, nd=4):
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    return f"{st.mean(vals):.{nd}f} ± {st.stdev(vals):.{nd}f}"


print(f"{'group':22s} {'n':>2s}  {'NMSE':>18s}  {'RMSE':>18s}  {'coarse RMSE':>18s}  best_ep")
print("-" * 104)
for base in sorted(runs):
    ms = sorted(runs[base], key=lambda m: m["tag"])
    nm = [m["nmse"] for m in ms]
    rm = [m["rmse"] for m in ms]
    cr = [m.get("coarse_rmse", float("nan")) for m in ms]
    be = [m.get("best_epoch", "-") for m in ms]
    print(f"{base:22s} {len(ms):2d}  {fmt(nm):>18s}  {fmt(rm):>18s}  {fmt(cr):>18s}  {be}")

print()
for base, ref in PAPER.items():
    if base in runs:
        rm = [m["rmse"] for m in runs[base]]
        nm = [m["nmse"] for m in runs[base]]
        print(f"{base}: ours RMSE {st.mean(rm):.4f}  vs paper {ref}")
