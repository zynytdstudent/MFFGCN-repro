# -*- coding: utf-8 -*-
"""论文 Fig.9 / Fig.10 的三联对照图。

左: NMSE   中: RMSE   右: 差距分解

右图是这张图存在的理由。NMSE 的定义 NMSE = RMSE²/E[t²] 决定了
"我们/论文"的 NMSE 比值必然等于 (RMSE 比值)² × (E_论文/E_我们)，
所以单看原始比值是看不出病因的。右图直接把两个因子并排画出来：
    左柱 = (RMSE 比)²   模型本身差多少
    右柱 = NMSE 比       图上看到的总差距
    红虚线 = 口径因子    纯粹因为两边 E[t²] 不同
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# 本地是 src/06_figures/，服务器上脚本常被直接放在 src/ 下跑，两种都认
for _c in (os.path.join(HERE, "..", ".."), os.path.join(HERE, ".."),
           os.path.join(HERE, "..", "..", "..")):
    if os.path.isdir(os.path.join(_c, "results")):
        ROOT = os.path.abspath(_c)
        break
else:
    raise SystemExit("找不到 results/ 目录")
R = os.path.join(ROOT, "results")
F = os.path.join(ROOT, "figs")

COLOR = {"RadioUNet": "#1f77b4", "RME-GAN": "#ff7f0e", "MFFGCN": "#2ca02c"}
MODELS = ["RadioUNet", "RME-GAN", "MFFGCN"]
LEV = {"dist": ["50", "70", "90", "110"], "side": ["90", "110", "130", "150"]}
XL = {"dist": "Distance from map centre (px)",
      "side": "Side length of restricted region (px)"}
FIGNO = {"dist": "9", "side": "10"}
E_FULL = 0.06879          # 全 8080 张测试图（作参照）
# 我们的 E[t^2] 从结果 json 的 et2 字段现读（supp_eval 直接算的），
# 不再写死——写死过一次，改成 --span 采样后忘了更新，脚注就错了。


def draw(mode):
    pap = json.load(open(os.path.join(R, "paper_fig9_10_digitized.json")))[FIGNO[mode]]
    ours = json.load(open(os.path.join(R, "supp_%s_paper.json" % mode)))
    lv = LEV[mode]
    xs = np.arange(len(lv))
    e_ours = float(np.mean([ours["models"]["MFFGCN"][l]["et2"] for l in lv]))

    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.35), dpi=170,
                           gridspec_kw={"width_ratios": [1, 1, 0.9]})

    for k, key in ((0, "nmse"), (1, "rmse")):
        step, wid = 0.30, 0.135
        for j, m in enumerate(MODELS):
            pv = [v for _, v in pap["a" if key == "nmse" else "b"][m]]
            ov = [ours["models"][m][l][key] for l in lv]
            xc0 = xs + (j - 1) * step
            ax[k].bar(xc0 - wid / 2 - 0.006, pv, wid, facecolor="none",
                      edgecolor=COLOR[m], lw=1.2, hatch="////",
                      label="%s — paper" % m if k == 0 else None)
            ax[k].bar(xc0 + wid / 2 + 0.006, ov, wid, color=COLOR[m],
                      alpha=0.92,
                      label="%s — ours" % m if k == 0 else None)
        ax[k].set_xticks(xs)
        ax[k].set_xticklabels(lv, fontsize=9)
        ax[k].set_xlabel(XL[mode], fontsize=9)
        ax[k].set_ylabel(key.upper(), fontsize=9.5)
        ax[k].set_title("(%s) %s" % ("ab"[k], key.upper()), fontsize=10.5)
        ax[k].grid(axis="y", alpha=0.22, lw=0.7)
        ax[k].tick_params(labelsize=8.5)
        ax[k].set_ylim(0, None)

    # ---------- (c) 分解 ----------
    axc = ax[2]
    ratios, rmses, epap = [], [], []
    for m in MODELS:
        pn = float(np.mean([v for _, v in pap["a"][m]]))
        pr = float(np.mean([v for _, v in pap["b"][m]]))
        on = float(np.mean([ours["models"][m][l]["nmse"] for l in lv]))
        opr = float(np.mean([ours["models"][m][l]["rmse"] for l in lv]))
        ratios.append(on / pn)
        rmses.append((opr / pr) ** 2)
        epap.append(pr * pr / pn)
    fden = float(np.mean(epap)) / e_ours

    xc = np.arange(len(MODELS))
    wc = 0.38
    axc.bar(xc - wc / 2, rmses, wc, color=[COLOR[m] for m in MODELS],
            label="$(RMSE_{ours}/RMSE_{paper})^2$")
    axc.bar(xc + wc / 2, ratios, wc, facecolor="white",
            edgecolor=[COLOR[m] for m in MODELS], lw=1.8,
            label="NMSE$_{ours}$/NMSE$_{paper}$")
    axc.axhline(fden, color="crimson", ls="--", lw=1.5,
                label="$E[t^2]_{paper}/E[t^2]_{ours}$ = %.2f" % fden)
    axc.axhline(1.0, color="gray", lw=1.0)
    for i, (r, f) in enumerate(zip(ratios, rmses)):
        axc.text(i - wc / 2, f * 1.10, "%.2f" % f, ha="center", fontsize=8.8)
        axc.text(i + wc / 2, r * 1.10, "%.2f" % r, ha="center", fontsize=8.8,
                 fontweight="bold")
    axc.set_yscale("log")
    axc.set_ylim(0.80, 6.2)
    axc.set_yticks([1, 1.5, 2, 3, 4, 5])
    axc.set_yticklabels(["1", "1.5", "2", "3", "4", "5"], fontsize=8.5)
    axc.minorticks_off()
    axc.set_xticks(xc)
    axc.set_xticklabels(MODELS, fontsize=9)
    axc.set_ylabel("ours / paper", fontsize=9.5)
    axc.set_title("(c) the two ratios differ by $E[t^2]$", fontsize=10.5)
    axc.grid(axis="y", alpha=0.22, lw=0.7)
    axc.legend(fontsize=7.6, loc="upper left", frameon=False)

    fig.suptitle("Fig.%s protocol — ours vs paper   (Setup 3 weights, n=200 maps)"
                 % FIGNO[mode], fontsize=11.5)
    fig.text(0.5, 0.005,
             "NMSE = RMSE$^2$/E[t$^2$].   E[t$^2$]: paper %.4f, ours %.4f "
             "(n=200 maps spread evenly over the test set; the full 8080 maps give %.4f)"
             % (float(np.mean(epap)), e_ours, E_FULL),
             ha="center", fontsize=8.2, color="#555")
    fig.tight_layout(rect=[0, 0.045, 1, 0.92])
    out = os.path.join(F, "cmp_%s_v2.png" % mode)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.abspath(out))


if __name__ == "__main__":
    for m in ("dist", "side"):
        draw(m)
