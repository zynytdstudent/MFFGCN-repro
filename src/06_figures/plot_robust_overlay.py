"""论文 Fig.14/15/16 vs 复现图 同轴叠合。

上一轮的错误在于画法：我把"绝对像素分辨率"和"干净基线"放进了横轴，而论文图
只画"相对退化档位"、且不含干净点。坐标轴不同，两条曲线当然看着完全不一样。
这里把横轴换成论文自己的档位，再把论文曲线（从 PDF 位图数字化）叠上来。

用法（项目根目录）：
    python src/06_figures/plot_robust_overlay.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break

COLORS = {"RadioUNet": "#1f77b4", "RME-GAN": "#ff7f0e", "MFFGCN": "#2ca02c"}
KS = ["RadioUNet", "RME-GAN", "MFFGCN"]

PANELS = [
    ("noise",   "14", [0.001, 0.004, 0.007, 0.010],
     ["0.1%", "0.4%", "0.7%", "1.0%"], "Impulse noise intensity"),
    ("degrade", "15", [64, 85, 128, 192],
     ["25%", "33%", "50%", "75%"], "Downsampling ratio"),
    ("missing", "16", [1, 2, 3, 4],
     ["1", "2", "3", "4"], "Number of missing buildings"),
]


def fill_none(vals, n=4):
    """补点：数字化时被图例挡住、被别的曲线压住、或整根柱子漏检的点，
    用同一条曲线线性插值补齐，并统一重采样到 n 个点。"""
    v = np.array([np.nan if x is None else x for x in vals], dtype=float)
    idx = np.arange(len(v))
    good = ~np.isnan(v)
    if good.sum() >= 2 and good.sum() < len(v):
        v[~good] = np.interp(idx[~good], idx[good], v[good])
    if len(v) != n:
        if good.sum() >= 2:
            v = np.interp(np.linspace(0, len(v) - 1, n), idx, v)
        else:
            v = np.full(n, np.nan)
    return v


def load_paper():
    """读数字化结果并补点。

    补点分两步，顺序不能反：
      1. 同一条曲线内部线性插值（点数不足 4 时先重采样）
      2. 跨模型补：Fig.15 里三条曲线在 50%/75% 几乎重合，后画的把先画的
         盖住了，数字化只能读到最上面那条。这种点用"同横坐标其他模型的均值"
         补，物理含义就是"它们在这个档位本来就分不开"。
    """
    with open("results/paper_digitized.json", encoding="utf-8") as fh:
        d = json.load(fh)

    raw = {}
    for fig, blob in d.items():
        for pk, panel in blob["panels"].items():
            metric = "nmse" if pk == "a" else "rmse"
            for model, vals in panel["series"].items():
                raw[(fig, metric, model)] = np.array(
                    [np.nan if v is None else v for v in vals], dtype=float)

    for k, v in list(raw.items()):
        if len(v) != 4:
            idx = np.arange(len(v))
            good = ~np.isnan(v)
            if good.sum() >= 2:
                raw[k] = np.interp(np.linspace(0, len(v) - 1, 4), idx[good], v[good])
            else:
                raw[k] = np.full(4, np.nan)

    for (fig, metric, model), v in list(raw.items()):
        if not np.isnan(v).any():
            continue
        others = [raw[(fig, metric, m)] for m in KS
                  if m != model and (fig, metric, m) in raw]
        if not others:
            continue
        mat = np.vstack(others)
        for i in np.where(np.isnan(v))[0]:
            col = mat[:, i][~np.isnan(mat[:, i])]
            if len(col):
                v[i] = col.mean()
        raw[(fig, metric, model)] = v
    return raw


def series(ours, model, key, order):
    d = ours["models"][model]
    return np.array([d[str(t)][key] for t in order])


def main():
    paper = load_paper()
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.8), dpi=170)

    for col, (mode, fignum, order, ticks, xlab) in enumerate(PANELS):
        with open("results/supp_%s_paper.json" % mode, encoding="utf-8") as fh:
            ours = json.load(fh)
        for row, key in enumerate(("nmse", "rmse")):
            ax = axes[row, col]
            xs = np.arange(len(order))
            for m in KS:
                c = COLORS[m]
                pv = paper[(fignum, key, m)][:len(order)]
                ov = series(ours, m, key, order)
                ax.plot(xs, pv, "--", lw=1.4, color=c, alpha=0.6,
                        marker="s", ms=3.4, mfc="none")
                ax.plot(xs, ov, "-", lw=2.0, color=c, marker="o", ms=4.4)
            ax.set_xticks(xs)
            ax.set_xticklabels(ticks, fontsize=8.5)
            ax.set_xlabel(xlab, fontsize=9)
            ax.set_ylabel(key.upper(), fontsize=9.5)
            ax.tick_params(labelsize=8.5)
            ax.grid(alpha=0.22, lw=0.7)
            ax.set_title("Fig.%s%s  %s" % (fignum, "ab"[row], xlab), fontsize=9.5)

    # 图例：颜色=模型，线型=来源
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLORS[m], lw=2, label=m) for m in KS]
    handles += [Line2D([], [], color="#444", lw=1.4, ls="--", marker="s",
                       ms=3.4, mfc="none", label="paper (digitized)"),
                Line2D([], [], color="#444", lw=2.0, marker="o", ms=4.4,
                       label="ours (re-run)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.8,
               frameon=False, bbox_to_anchor=(0.5, -0.004))
    fig.suptitle("Paper (dashed, digitized from the PDF) vs our re-run (solid) "
                 "- same axes, same units", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out = "figs/paper_vs_ours_axes.png"
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)

    rows = []
    for mode, fignum, order, ticks, _ in PANELS:
        with open("results/supp_%s_paper.json" % mode, encoding="utf-8") as fh:
            ours = json.load(fh)
        for key in ("nmse", "rmse"):
            for m in KS:
                pv = paper[(fignum, key, m)][:len(order)]
                ov = series(ours, m, key, order)
                for i, t in enumerate(ticks):
                    rows.append((fignum, key.upper(), m, t, pv[i], ov[i],
                                 (ov[i] - pv[i]) / pv[i] * 100))
    with open("results/paper_vs_ours.csv", "w", encoding="utf-8") as fh:
        fh.write("fig,metric,model,x,paper,ours,dev_pct\n")
        for r in rows:
            fh.write("%s,%s,%s,%s,%.5f,%.5f,%+.1f\n" % r)
    print("saved results/paper_vs_ours.csv (%d rows)" % len(rows))

    print("\n逐点偏差（ours 相对 paper）:")
    for fignum in ("14", "15", "16"):
        for m in KS:
            sub = [r[6] for r in rows if r[0] == fignum and r[2] == m]
            if sub:
                print("  Fig.%s %-10s 中位 %+6.1f%%   |Δ|最大 %5.1f%%"
                      % (fignum, m, np.median(sub), max(abs(x) for x in sub)))


if __name__ == "__main__":
    main()
