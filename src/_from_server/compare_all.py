"""Full reproduction report: main comparison (Setup 1-5) + Table V ablation.

Reads every metrics_*.json produced by evaluate.py in one flat directory.

Naming convention
-----------------
    metrics_s1.json .. metrics_s5.json          -> main model, per setup
    metrics_<variant>_s1.json ...               -> ablation variant

Variant keys: single, add, concat, channel, cross, nogcn, static, nores

Usage:
    python compare_all.py --dir ../eval_ALL --out ../eval_ALL/reports
"""
import argparse
import glob
import json
import os

PAPER_T3 = {
    "KNN":       [(0.9973, 0.2312), (0.9980, 0.2313), (0.9981, 0.2313)],
    "Kriging":   [(0.4545, 0.1389), (0.4545, 0.1389), (0.4575, 0.1401)],
    "AE":        [(0.0387, 0.0456), (0.0753, 0.0627), (0.0829, 0.0657)],
    "DeepAE":    [(0.0156, 0.0288), (0.0276, 0.0319), (0.0395, 0.0422)],
    "U-Net":     [(0.0063, 0.0182), (0.0075, 0.0199), (0.0081, 0.0206)],
    "RadioUNet": [(0.0049, 0.0160), (0.0058, 0.0175), (0.0063, 0.0181)],
    "Pix2pix+":  [(0.0046, 0.0156), (0.0053, 0.0169), (0.0054, 0.0172)],
    "RME-GAN":   [(0.0043, 0.0151), (0.0051, 0.0165), (0.0055, 0.0172)],
    "MFFGCN":    [(0.0043, 0.0150), (0.0045, 0.0154), (0.0046, 0.0155)],
}

# Setup 4 / 5 only report MFFGCN against RME-GAN in the paper
PAPER_S45 = {"s4": (0.0023, 0.0110), "s5": (0.0024, 0.0113)}
PAPER_S45_RMEGAN_NMSE = {"s4": 0.0036, "s5": 0.0038}

PAPER_T5 = {
    "single":  ("Single Encoder",    [(0.0048, 0.0158), (0.0053, 0.0168), (0.0053, 0.0167)]),
    "add":     ("Additive",          [(0.0050, 0.0162), (0.0056, 0.0171), (0.0059, 0.0175)]),
    "concat":  ("Concatenation",     [(0.0047, 0.0158), (0.0053, 0.0166), (0.0053, 0.0167)]),
    "channel": ("Channel Attention", [(0.0045, 0.0155), (0.0051, 0.0164), (0.0053, 0.0165)]),
    "cross":   ("Cross-attention",   [(0.0046, 0.0156), (0.0052, 0.0164), (0.0053, 0.0164)]),
    "nogcn":   ("w/o GCN",           [(0.0049, 0.0161), (0.0054, 0.0166), (0.0055, 0.0170)]),
    "static":  ("w/o Dynamic",       [(0.0046, 0.0154), (0.0049, 0.0161), (0.0049, 0.0162)]),
    "nores":   ("w/o Residual",      [(0.0231, 0.0347), (0.0234, 0.0350), (0.0237, 0.0352)]),
}

SETUPS = ["s1", "s2", "s3", "s4", "s5"]
SETUP_NAME = {
    "s1": "Setup 1 (uniform)", "s2": "Setup 2 (25% restricted)",
    "s3": "Setup 3 (10-50% restricted)", "s4": "Setup 4 (1-10% restricted)",
    "s5": "Setup 5 (two sides, 1% vs 10%)",
}
COL = {"s1": 0, "s2": 1, "s3": 2}


def pct(v, ref):
    return float("nan") if not ref else (v - ref) / ref * 100.0


def load(path):
    out = {}
    for p in sorted(glob.glob(os.path.join(path, "metrics_*.json"))):
        with open(p) as fh:
            m = json.load(fh)
        out[m.get("name") or os.path.basename(p)[8:-5]] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../eval_ALL")
    ap.add_argument("--out", default="../eval_ALL/reports")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    data = load(a.dir)
    if not data:
        print(f"[warn] no metrics_*.json in {a.dir}")
        return
    base = {s: data[s] for s in SETUPS if s in data}
    abl = {k: {s: data.get(f"{k}_{s}") for s in SETUPS[:3]}
           for k in PAPER_T5 if any(f"{k}_{s}" in data for s in SETUPS[:3])}

    L = []
    L.append("# MFFGCN full reproduction - all experiments\n")
    L.append("RadioMapSeer DPM test split, last 100 city maps, unseen during "
             "training. NMSE / RMSE, lower is better.\n")

    # ---- T1 main across all five setups ---------------------------------
    L.append("## Table 1 - Main results over all five sampling setups\n")
    L.append("| Setup | Paper NMSE | Ours NMSE | Δ NMSE | Paper RMSE | Ours RMSE | Δ RMSE |")
    L.append("|---|---|---|---|---|---|---|")
    for s, m in base.items():
        f = m["fine"]
        if s in COL:
            pn, pr = PAPER_T3["MFFGCN"][COL[s]]
        else:
            pn, pr = PAPER_S45[s]
        L.append(f"| {SETUP_NAME[s]} | {pn:.4f} | {f['all_nmse']:.4f} | "
                 f"{pct(f['all_nmse'], pn):+.1f}% | {pr:.4f} | "
                 f"{f['all_rmse']:.4f} | {pct(f['all_rmse'], pr):+.1f}% |")

    # ---- T2 setup 4/5 head to head with RME-GAN -------------------------
    if any(s in base for s in ("s4", "s5")):
        L.append("\n## Table 2 - Extreme sparsity (Setup 4/5) vs the paper\n")
        L.append("| Setup | Paper MFFGCN | Paper RME-GAN | Ours | Ours vs paper MFFGCN |")
        L.append("|---|---|---|---|---|")
        for s in ("s4", "s5"):
            if s not in base:
                continue
            pn = PAPER_S45[s][0]
            rn = PAPER_S45_RMEGAN_NMSE[s]
            on = base[s]["fine"]["all_nmse"]
            L.append(f"| {SETUP_NAME[s]} | {pn:.4f} | {rn:.4f} | {on:.4f} | "
                     f"{on / pn:.1f}x |")

    # ---- T3 coarse vs refined ------------------------------------------
    L.append("\n## Table 3 - Graph refinement contribution (ours)\n")
    L.append("| Setup | Coarse NMSE | Refined NMSE | Gain | Coarse RMSE | Refined RMSE | Gain |")
    L.append("|---|---|---|---|---|---|---|")
    for s, m in base.items():
        c, f = m["coarse"], m["fine"]
        L.append(f"| {SETUP_NAME[s]} | {c['all_nmse']:.4f} | {f['all_nmse']:.4f} | "
                 f"{pct(f['all_nmse'], c['all_nmse']):+.1f}% | "
                 f"{c['all_rmse']:.4f} | {f['all_rmse']:.4f} | "
                 f"{pct(f['all_rmse'], c['all_rmse']):+.1f}% |")

    # ---- T4 ablation vs paper Table V -----------------------------------
    if abl:
        L.append("\n## Table 4 - Ablation (NMSE), ours vs paper Table V\n")
        hdr = "| Variant | " + " | ".join(f"{SETUP_NAME[s]}" for s in SETUPS[:3]) + " |"
        L.append(hdr)
        L.append("|---|" + "---|" * 3)
        for k, (name, pv) in PAPER_T5.items():
            if k not in abl:
                continue
            cells = []
            for i, s in enumerate(SETUPS[:3]):
                m = abl[k][s]
                if not m:
                    cells.append("—")
                    continue
                on = m["fine"]["all_nmse"]
                # Two deltas side by side: the first is our own effect size
                # (variant vs our Proposed), the second is the paper's
                # (variant vs its Proposed). Absolute levels differ by ~20x,
                # so only these two numbers are comparable to each other.
                ob = base[s]["fine"]["all_nmse"] if s in base else float("nan")
                pp = PAPER_T3["MFFGCN"][i][0]
                cells.append(f"{on:.4f} ({pct(on, ob):+.1f}% / {pct(pv[i][0], pp):+.1f}%)")
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        cells = []
        for s in SETUPS[:3]:
            if s in base:
                on = base[s]["fine"]["all_nmse"]
                cells.append(f"**{on:.4f}**")
            else:
                cells.append("—")
        L.append("| **Proposed (ours)** | " + " | ".join(cells) + " |")
        L.append("\nEach cell reads `NMSE (ours vs our Proposed / paper vs its "
                 "Proposed)`. The left number is the ablation effect measured "
                 "inside our reproduction, the right one is the same effect in "
                 "the paper. Comparing our absolute NMSE to the paper's "
                 "absolute NMSE is meaningless because the two baselines are "
                 "~20x apart; only the two relative effects are comparable.\n")

        # ---- T5 ablation ranking inside our own reproduction -------------
        L.append("## Table 5 - Ablation ranking inside our reproduction\n")
        L.append("| Variant | S1 NMSE | Δ vs ours Proposed (S1) | Paper Δ vs its Proposed (S1) |")
        L.append("|---|---|---|---|")
        ref_o = base.get("s1", {}).get("fine", {}).get("all_nmse")
        for k, (name, pv) in PAPER_T5.items():
            if k not in abl or not abl[k]["s1"] or ref_o is None:
                continue
            on = abl[k]["s1"]["fine"]["all_nmse"]
            L.append(f"| {name} | {on:.4f} | {pct(on, ref_o):+.1f}% | "
                     f"{pct(pv[0][0], PAPER_T3['MFFGCN'][0][0]):+.1f}% |")

    txt = "\n".join(L)
    with open(os.path.join(a.out, "comparison_all.md"), "w") as fh:
        fh.write(txt + "\n")
    print(txt)
    with open(os.path.join(a.out, "comparison_all.json"), "w") as fh:
        json.dump({"main": base, "ablation": abl,
                   "paper_table3": PAPER_T3, "paper_s45": PAPER_S45,
                   "paper_table5": PAPER_T5}, fh, indent=2)
    print(f"\n[ok] written to {a.out}")


if __name__ == "__main__":
    main()
