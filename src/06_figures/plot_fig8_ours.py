# -*- coding: utf-8 -*-
"""Fig.8 复现图（独立成图，形式与论文一致，不与论文叠画）。

论文 Fig.8：受限区域（Setup 2 / Setup 3）下 7 个方法的 NMSE 与 RMSE，
两栏上下排列，(a) NMSE (b) RMSE，每个模型两根柱（Setup 2 蓝 / Setup 3 橙），
图例放右上角，y 轴与论文同尺度（NMSE 0-0.14，RMSE 0-0.095）。

数据来源：results/fig8_metrics.json（由各 run 的 test_metrics.json 汇总，
test 集 8080 张全量）。

用法： python plot_fig8_ours.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# 脚本在本地是 src/06_figures/，在服务器上被平铺到 src/，两种都要能找到根目录
ROOT = HERE
for _ in ("..", os.path.join("..", "..")):
    _c = os.path.abspath(os.path.join(HERE, _))
    if os.path.isdir(os.path.join(_c, "results")):
        ROOT = _c
        break
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figs")

C2, C3 = "#1f77b4", "#ff7f0e"

PANELS = [
    ("nmse", "NMSE", 0.145, [0, .02, .04, .06, .08, .10, .12], "(a)"),
    ("rmse", "RMSE", 0.100, [0, .02, .04, .06, .08], "(b)"),
]


def main():
    d = json.load(open(os.path.join(RES, "fig8_metrics.json"), encoding="utf-8"))
    models = d["models"]
    xs = np.arange(len(models))
    w = 0.38

    plt.rcParams.update({"font.family": "DejaVu Sans",
                         "axes.spines.top": True, "axes.spines.right": True})

    fig, axes = plt.subplots(2, 1, figsize=(6.3, 7.9))
    for ax, (key, ylab, ymax, yticks, tag) in zip(axes, PANELS):
        v2 = [d["metrics"][m]["s2"][key] for m in models]
        v3 = [d["metrics"][m]["s3"][key] for m in models]
        ax.bar(xs - w / 2, v2, w, color=C2, label="Setup 2")
        ax.bar(xs + w / 2, v3, w, color=C3, label="Setup 3")
        ax.set_xticks(xs)
        ax.set_xticklabels(models, fontsize=8.6)
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.set_ylim(0, ymax)
        ax.set_yticks(yticks)
        ax.tick_params(labelsize=9.5)
        ax.set_xlabel("Model", fontsize=11, labelpad=2)
        ax.set_ylabel(ylab, fontsize=11)
        ax.legend(fontsize=10, loc="upper right", frameon=True,
                  handlelength=1.6, borderpad=0.6)
        ax.text(0.5, -0.255, tag, transform=ax.transAxes, ha="center",
                fontsize=14)

    fig.tight_layout(h_pad=4.2)
    out = os.path.join(FIG, "fig8_ours.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)

    # 数值自检：NMSE = RMSE^2 / E[t^2]，21 格反推出的 E[t^2] 必须是同一个常数
    et2 = [d["metrics"][m][s]["rmse"] ** 2 / d["metrics"][m][s]["nmse"]
           for m in models for s in ("s2", "s3")]
    assert max(et2) - min(et2) < 5e-4, et2
    print("自检通过：21 格反推 E[t^2] = %.5f ± %.1e（同一分母）"
          % (sum(et2) / len(et2), max(et2) - min(et2)))


if __name__ == "__main__":
    main()
