"""Compare our reproduced metrics against the values reported in the paper."""

import argparse
import glob
import json
import os

# --------------------------------------------------------------------------
# TABLE III  - comparison with state-of-the-art  (NMSE / RMSE)
# transcribed from page 9 of the PDF (values are the paper's own numbers)
# --------------------------------------------------------------------------
PAPER_TABLE3 = {
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

# TABLE V - ablation  (NMSE / RMSE)
PAPER_TABLE5 = {
    "Single Encoder":    [(0.0048, 0.0158), (0.0053, 0.0168), (0.0053, 0.0167)],
    "Additive":          [(0.0050, 0.0162), (0.0056, 0.0171), (0.0059, 0.0175)],
    "Concatenation":     [(0.0047, 0.0158), (0.0053, 0.0166), (0.0053, 0.0167)],
    "Channel Attention": [(0.0045, 0.0155), (0.0051, 0.0164), (0.0053, 0.0165)],
    "Cross-attention":   [(0.0046, 0.0156), (0.0052, 0.0164), (0.0053, 0.0164)],
    "w/o GCN":           [(0.0049, 0.0161), (0.0054, 0.0166), (0.0055, 0.0170)],
    "w/o Dynamic":       [(0.0046, 0.0154), (0.0049, 0.0161), (0.0049, 0.0162)],
    "w/o Residual":      [(0.0231, 0.0347), (0.0234, 0.0350), (0.0237, 0.0352)],
    "Proposed":          [(0.0043, 0.0150), (0.0045, 0.0154), (0.0046, 0.0155)],
}

SETUP_IDX = {"s1": 0, "s2": 1, "s3": 2}
SETUP_NAME = {"s1": "Setup 1 (uniform)", "s2": "Setup 2 (25% restricted)",
              "s3": "Setup 3 (10-50% restricted)"}


def _normalize(m):
    """Accept both evaluate.py output and train.py test_metrics.json."""
    if "fine" in m:
        return m
    return {
        "fine": {"all_nmse": m["nmse"], "all_rmse": m["rmse"]},
        "coarse": {"all_nmse": m.get("coarse_nmse", m.get("nmse_coarse")),
                   "all_rmse": m.get("coarse_rmse", m.get("rmse_coarse"))},
    }


def load_ours(path):
    """Scan a directory for per-setup metric files (either format)."""
    files = (glob.glob(os.path.join(path, "metrics_*.json"))
             + glob.glob(os.path.join(path, "*", "test_metrics.json")))
    out = {}
    for p in sorted(files):
        with open(p) as fh:
            m = json.load(fh)
        setup = m.get("setup")
        if not setup:
            tag = os.path.basename(os.path.dirname(p))
            setup = next((s for s in SETUP_IDX if tag.startswith(s)), tag)
        m = _normalize(m)
        m["setup"] = setup
        out.setdefault(setup, m)          # first hit wins
    return out


def pct(ours, ref):
    if ref == 0:
        return float("nan")
    return (ours - ref) / ref * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="../reports")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ours = load_ours(a.runs)
    if not ours:
        print(f"[warn] no test_metrics.json under {a.runs}")
        return

    lines = []
    lines.append("# MFFGCN Reproduction vs. Paper\n")
    lines.append("All numbers are reported on the RadioMapSeer DPM test split "
                 "(last 100 city maps, unseen during training).\n")

    # ---------- main comparison ------------------------------------------
    lines.append("## Table 1 - Reproduction vs. paper (MFFGCN)\n")
    lines.append("| Setup | Paper NMSE | Ours NMSE | Δ NMSE | Paper RMSE | "
                 "Ours RMSE | Δ RMSE |")
    lines.append("|---|---|---|---|---|---|---|")
    summary = {}
    for s in ("s1", "s2", "s3"):
        if s not in ours:
            continue
        m = ours[s]
        pn, pr = PAPER_TABLE3["MFFGCN"][SETUP_IDX[s]]
        on = m["fine"][f"all_nmse"]
        orr = m["fine"][f"all_rmse"]
        summary[s] = dict(paper_nmse=pn, ours_nmse=on,
                          paper_rmse=pr, ours_rmse=orr,
                          d_nmse=pct(on, pn), d_rmse=pct(orr, pr))
        lines.append(
            f"| {SETUP_NAME[s]} | {pn:.4f} | {on:.4f} | {pct(on,pn):+.1f}% | "
            f"{pr:.4f} | {orr:.4f} | {pct(orr,pr):+.1f}% |")

    # ---------- coarse vs refined ---------------------------------------
    lines.append("\n## Table 2 - Effect of the graph refinement stage (ours)\n")
    lines.append("| Setup | Coarse NMSE | Refined NMSE | Gain | Coarse RMSE | "
                 "Refined RMSE | Gain |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in ("s1", "s2", "s3"):
        if s not in ours:
            continue
        m = ours[s]
        cn, rn = m["coarse"]["all_nmse"], m["fine"]["all_nmse"]
        cr, rr = m["coarse"]["all_rmse"], m["fine"]["all_rmse"]
        lines.append(f"| {SETUP_NAME[s]} | {cn:.4f} | {rn:.4f} | "
                     f"{pct(rn,cn):+.1f}% | {cr:.4f} | {rr:.4f} | "
                     f"{pct(rr,cr):+.1f}% |")

    # ---------- per-region ----------------------------------------------
    lines.append("\n## Table 3 - Reconstruction quality by region (ours)\n")
    lines.append("| Setup | Whole map RMSE | Constrained region RMSE | "
                 "Observed pixels RMSE |")
    lines.append("|---|---|---|---|")
    for s in ("s1", "s2", "s3"):
        if s not in ours:
            continue
        f = ours[s]["fine"]
        cr = f.get("constr_rmse")
        cr = "—" if cr is None else f"{cr:.4f}"
        ob = f.get("obs_rmse")
        ob = "—" if ob is None else f"{ob:.4f}"
        lines.append(f"| {SETUP_NAME[s]} | {f['all_rmse']:.4f} | {cr} | {ob} |")

    # ---------- degradation with setup difficulty -----------------------
    lines.append("\n## Table 4 - Robustness to uneven sampling\n")
    lines.append("| Setup | Paper RMSE increase vs S1 | Ours RMSE increase vs S1 |")
    lines.append("|---|---|---|")
    base_p = PAPER_TABLE3["MFFGCN"][0][1]
    if "s1" in ours:
        base_o = ours["s1"]["fine"]["all_rmse"]
        for s in ("s2", "s3"):
            if s not in ours:
                continue
            pp = PAPER_TABLE3["MFFGCN"][SETUP_IDX[s]][1]
            oo = ours[s]["fine"]["all_rmse"]
            lines.append(f"| {SETUP_NAME[s]} | {pct(pp,base_p):+.1f}% | "
                         f"{pct(oo,base_o):+.1f}% |")

    # ---------- ladder against published baselines ----------------------
    lines.append("\n## Table 5 - Where our model sits among published baselines\n")
    hdr = "| Method | Setup 1 (NMSE / RMSE) | Setup 2 (NMSE / RMSE) | Setup 3 (NMSE / RMSE) |"
    lines.append(hdr)
    lines.append("|---|---|---|---|")
    for name, vals in PAPER_TABLE3.items():
        cells = " | ".join(f"{n:.4f} / {r:.4f}" for n, r in vals)
        lines.append(f"| {name} (paper) | {cells} |")
    if all(s in ours for s in ("s1", "s2", "s3")):
        cells = " | ".join(
            f"{ours[s]['fine']['all_nmse']:.4f} / {ours[s]['fine']['all_rmse']:.4f}"
            for s in ("s1", "s2", "s3"))
        lines.append(f"| **Ours (reproduced)** | {cells} |")

    txt = "\n".join(lines)
    path = os.path.join(a.out, "comparison.md")
    with open(path, "w") as fh:
        fh.write(txt + "\n")
    print(txt)
    print(f"\n[ok] written to {path}")

    with open(os.path.join(a.out, "comparison.json"), "w") as fh:
        json.dump({"ours": ours, "summary": summary,
                   "paper_table3": PAPER_TABLE3,
                   "paper_table5": PAPER_TABLE5}, fh, indent=2)


if __name__ == "__main__":
    main()
