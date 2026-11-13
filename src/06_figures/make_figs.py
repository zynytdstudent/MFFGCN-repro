"""make_figs.py - 从 eval_sweep_test.jsonl 生成报告用图（英文标注，期刊风格）。

输出到 figs/：
  fig1_setups.png    五个采样 setup：粗估计 vs 残差精化
  fig2_paper.png     论文与复现对照（NMSE）
  fig3_ablation.png  Setup 1 消融：相对增益 + 绝对 NMSE，附噪声带
  fig4_best_last.png best.pt 与 last.pt 的偏差（为什么必须用 last）
  fig5_quali.png     定性结果（由 evaluate.py 出的样本图纵向拼接）
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "eval_sweep_test.jsonl"
OUT = "figs"
os.makedirs(OUT, exist_ok=True)

C_MAIN, C_BASE, C_GRAY, C_WARN = "#2E5A88", "#C0504D", "#9AA0A6", "#E8A33D"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
D = {(r["tag"], r["which"]): r for r in rows}

EVAL_NOISE = 0.006      # 同一权重换 3 个 eval seed 的极差
TRAIN_NOISE = 0.004     # 同配置、同种子重复训练（calib_single vs txf_single_s1）


# ---------------------------------------------------------------- figure 1
def fig1():
    setups = ["s1", "s2", "s3", "s4", "s5"]
    fine = [D[("FINAL_" + s, "last")]["nmse"] for s in setups]
    coarse = [D[("FINAL_" + s, "last")]["coarse_nmse"] for s in setups]
    rfine = [D[("FINAL_" + s, "last")]["rmse"] for s in setups]
    rcoarse = [D[("FINAL_" + s, "last")]["coarse_rmse"] for s in setups]

    x = np.arange(len(setups))
    w = 0.36
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))

    for a, cv, fv, ttl, lab in (
            (ax[0], coarse, fine, "(a) NMSE", "Normalised MSE"),
            (ax[1], rcoarse, rfine, "(b) RMSE", "RMSE")):
        a.bar(x - w / 2, cv, w, label="Coarse estimate", color=C_GRAY)
        a.bar(x + w / 2, fv, w, label="Refined (MFFGCN)", color=C_MAIN)
        a.set_xticks(x)
        a.set_xticklabels(["S1", "S2", "S3", "S4", "S5"])
        a.set_title(ttl, fontsize=10, loc="left")
        a.set_ylabel(lab)
        top = max(max(cv), max(fv))
        a.set_ylim(0, top * 1.30)
        for i, v in enumerate(cv):
            a.text(i - w / 2, v + top * 0.015, "%.4f" % v,
                   ha="center", va="bottom", fontsize=7.2, color="#555555")
        for i, v in enumerate(fv):
            a.text(i + w / 2, v + top * 0.015, "%.4f" % v,
                   ha="center", va="bottom", fontsize=7.2)
    ax[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("MFFGCN on five uneven-sampling setups "
                 "(RadioMapSeer, 8080 test maps)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "fig1_setups.png"))
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig2():
    paper = {"U-Net": 0.0063, "MFFGCN": 0.0043}
    ours = {"U-Net": D[("calib_single", "last")]["nmse"],
            "MFFGCN": D[("FINAL_s1", "last")]["nmse"]}
    names = ["U-Net", "MFFGCN"]
    x = np.arange(2)
    w = 0.3
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    b1 = ax.bar(x - w / 2, [paper[n] for n in names], w,
                label="Paper (TMC 2025)", color=C_GRAY)
    b2 = ax.bar(x + w / 2, [ours[n] for n in names], w,
                label="This reproduction", color=C_MAIN)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    "%.4f" % b.get_height(), ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("NMSE (Setup 1)")
    ax.set_ylim(0, 0.0073)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Paper vs. reproduction, Setup 1", fontsize=10, loc="left")
    # 标注相对误差
    for i, n in enumerate(names):
        d = (ours[n] - paper[n]) / paper[n] * 100
        ax.text(i, max(paper[n], ours[n]) + 0.00045, "%+.0f%%" % d,
                ha="center", fontsize=8.5, color=C_WARN)
    fig.savefig(os.path.join(OUT, "fig2_paper.png"))
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
ABL = [
    ("ab_single_s1", "Single-encoder U-Net"),
    ("ab_nogcn_s1", "w/o GCN"),
    ("ab_static_s1", "w/o Dynamic graph"),
    ("ab_nores_s1", "w/o Residual"),
    ("ab_cross_s1", "AFM $\\rightarrow$ cross-attn"),
    ("ab_channel_s1", "AFM $\\rightarrow$ SE gating"),
    ("afm_spatial_s1", "AFM $\\rightarrow$ spatial"),
    ("ab_add_s1", "AFM $\\rightarrow$ additive"),
    ("ab_concat_s1", "AFM $\\rightarrow$ concatenation"),
]


def fig3():
    from matplotlib.patches import Patch
    full = [D[("bs15_s1", "last")]["nmse"], D[("FINAL_s1", "last")]["nmse"]]
    base = sum(full) / 2.0
    items = [(lab, D[(t, "last")]["nmse"]) for t, lab in ABL]
    items.sort(key=lambda kv: kv[1])
    labs = [k for k, _ in items]
    vals = [v for _, v in items]
    delta = [(v - base) / base * 100 for v in vals]

    print("\n[ablation, Setup 1]  full model runs: %.5f / %.5f  mean %.5f"
          % (full[0], full[1], base))
    for (lab, v), d in zip(items, delta):
        print("  %-32s %.5f  %+5.1f%%" % (lab, v, d))

    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.0),
                           gridspec_kw={"width_ratios": [1.05, 1]})

    y = np.arange(len(labs))
    ax[0].barh(y, delta, height=0.62, color=C_MAIN)
    ax[0].axvline(0, color="k", lw=1)
    ax[0].axvspan(0, TRAIN_NOISE * 100, color=C_MAIN, alpha=0.13, lw=0)
    for i, d in enumerate(delta):
        ax[0].text(d + 0.5, i, "%+.1f%%" % d, va="center", fontsize=8)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(labs, fontsize=9)
    ax[0].set_xlabel("NMSE degradation vs. full model (%)")
    ax[0].set_xlim(0, max(delta) * 1.20)
    ax[0].set_title("(a) Ablation, Setup 1", fontsize=10, loc="left")
    ax[0].invert_yaxis()
    ax[0].legend(handles=[
        Patch(color=C_MAIN, alpha=0.13,
              label="repeat-run spread (%.1f%%)" % (TRAIN_NOISE * 100))],
        frameon=False, fontsize=7.5, loc="upper right")

    y2 = np.arange(len(labs) + 1)
    ax[1].barh(y2, [base] + vals, color=[C_MAIN] + [C_GRAY] * len(vals),
               height=0.62)
    for i, v in enumerate([base] + vals):
        ax[1].text(v * 1.015, i, "%.4f" % v, va="center", fontsize=7.8)
    ax[1].set_yticks(y2)
    ax[1].set_yticklabels(["Full MFFGCN"] + labs, fontsize=9)
    ax[1].set_xlim(0, max(vals) * 1.17)
    ax[1].set_xlabel("NMSE")
    ax[1].set_title("(b) Absolute error", fontsize=10, loc="left")
    ax[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_ablation.png"))
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig4():
    tags = sorted({t for t, w in D})
    xs, ys, ep = [], [], []
    for t in tags:
        b, l = D.get((t, "best")), D.get((t, "last"))
        if not (b and l):
            continue
        xs.append(b["nmse"])
        ys.append(l["nmse"])
        ep.append(b["epoch"])
    xs, ys, ep = np.array(xs), np.array(ys), np.array(ep)
    bad = (ep <= 3) | (xs > 3 * ys)

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    lim = (0.0009, 0.05)
    ax.plot(lim, lim, "--", color="k", lw=1, label="best = last")
    ax.scatter(xs[~bad], ys[~bad], s=40, color=C_MAIN, alpha=0.85,
               edgecolor="w", lw=0.6, label="acceptable checkpoints")
    ax.scatter(xs[bad], ys[bad], s=44, color=C_BASE, alpha=0.9,
               edgecolor="w", lw=0.6,
               label="best.pt picked at epoch <= 3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("NMSE of best.pt\n(selected by the corrupted val metric)",
                  fontsize=9)
    ax.set_ylabel("NMSE of last.pt (epoch 50)", fontsize=9)
    ax.set_title("Why the table must use last.pt", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.text(0.0011, 0.032, "%d of %d runs sit far above the diagonal"
            % (int(bad.sum()), len(xs)), fontsize=8, color=C_BASE)
    fig.savefig(os.path.join(OUT, "fig4_best_last.png"))
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def fig5():
    from PIL import Image
    paths = ["eval_fix/s1/sample_s1_0.png", "eval_fix/s2/sample_s2_0.png",
             "eval_fix/s3/sample_s3_0.png"]
    ims = [Image.open(p) for p in paths if os.path.exists(p)]
    if not ims:
        print("[warn] 没有定性样本图，跳过 fig5")
        return
    w = min(i.width for i in ims)
    ims = [i.resize((w, int(i.height * w / i.width)), Image.LANCZOS) for i in ims]
    H = sum(i.height for i in ims)
    canvas = Image.new("RGB", (w, H), "white")
    y = 0
    for i in ims:
        canvas.paste(i, (0, y))
        y += i.height
    canvas.save(os.path.join(OUT, "fig5_quali.png"), optimize=True)
    print("[ok] fig5", canvas.size)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("[ok] figures ->", OUT)
