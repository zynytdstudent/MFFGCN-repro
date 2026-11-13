# -*- coding: utf-8 -*-
"""论文 Fig.9 / Fig.10 柱状图数字化。

两图结构完全一样：左panel = NMSE，右panel = RMSE，横轴 4 档，
每档 3 根柱（RadioUNet 蓝 / RME-GAN 橙 / MFFGCN 绿），tab10 配色。

做法：
  1. 用"灰色像素"（R≈G≈B 且不太亮）找轴框，避开柱子的饱和色；
  2. 用 y 轴刻度文字的文本行（左边缘 band）线性标定；
  3. 柱子从底 spine 向上逐列数连续同色段 —— 图例色块不接触底部，天然排除。

自检（很重要）：论文 NMSE 与 RMSE 必须满足 RMSE² = NMSE × E[t²]，
用三个模型各自反推 E[t²]，应当互相一致。实测三者都落在 0.053±0.001，
与表 III 反推的 0.0527 一致 —— 说明刻度标定正确。

用法：python digit_fig9_10.py
输出：results/paper_fig9_10_digitized.json
"""
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "..", "reports", "fig_compare", "paper")
OUT = os.path.join(HERE, "..", "..", "results", "paper_fig9_10_digitized.json")

COL = {"RadioUNet": (31, 119, 179), "RME-GAN": (255, 127, 14), "MFFGCN": (44, 159, 44)}

# panel 的 x 范围（不含轴框），以及 (底行, 顶刻度行, 底刻度值, 顶刻度值)
PANELS = {"a": (108, 471), "b": (625, 998)}
CAL = {"a": (424, 41, 0.0040, 0.0070), "b": (424, 22, 0.014, 0.020)}
Y_BOT = 428          # 柱子底部（轴框内最后一根柱像素；429 已被 spine 描边覆盖）


def columns(path):
    """返回 (RGB array, {panel: {model: [(x, value), ...]}})。"""
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    out = {}
    for tag, (x0, x1) in PANELS.items():
        y_bot_cal, y_top_cal, v_bot, v_top = CAL[tag]
        out[tag] = {}
        for name, c in COL.items():
            m = np.abs(a - c).sum(2) < 40
            tops = {}
            for x in range(x0, x1):
                y, h = Y_BOT, 0
                while y > 0 and m[y, x]:
                    h += 1
                    y -= 1
                if h > 5:
                    tops[x] = Y_BOT - h + 1
            xs = sorted(tops)
            groups = []
            for x in xs:
                if groups and x - groups[-1][-1] <= 2:
                    groups[-1].append(x)
                else:
                    groups.append([x])
            vals = []
            for g in groups:
                if len(g) < 4:
                    continue
                ty = float(np.median([tops[x] for x in g]))
                v = v_bot + (y_bot_cal - ty) / (y_bot_cal - y_top_cal) * (v_top - v_bot)
                vals.append((int(np.mean(g)), round(v, 5)))
            out[tag][name] = vals
    return out


def main():
    res = {}
    for f in ("09", "10"):
        res[f.lstrip("0")] = columns(os.path.join(PAPER, "paper_fig%s.png" % f))

    # ---- 自检：反推 E[t^2] ----
    print("自检 E[t^2] = RMSE^2 / NMSE")
    for fk, blob in res.items():
        for name in COL:
            n = [v for _, v in blob["a"][name]]
            r = [v for _, v in blob["b"][name]]
            e = [(ri * ri) / ni for ni, ri in zip(n, r)]
            print("  Fig.%s %-10s  %s   mean=%.5f"
                  % (fk, name, " ".join("%.4f" % x for x in e), np.mean(e)))

    json.dump(res, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
