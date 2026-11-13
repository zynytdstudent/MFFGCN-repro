# -*- coding: utf-8 -*-
"""digit_fig7.py - 从论文 Fig.7 位图反解 7 条 NMSE 曲线。

为什么单独写一个：digit_paper.py 只处理 Fig.14/15/16，用的是"从轴线外侧
找刻度短线"的定位法和 tab:blue/orange/green 三色；Fig.7 是**双纵轴对数图**
且 7 条曲线（含纯蓝 #0000ff 与 tab:brown），那两个假设都不成立。

本脚本的做法（已逐条核对过）：
  1. 轴框：暗行/暗列直方图 → panel a: x∈[70,436]、y∈[20,383]；
     panel b: x∈[598,963]、同 y。
  2. 纵轴标定：左轴外侧 25–68 px 的文本带里，只有两个大刻度带
     （33–43 与 263–273）→ 10⁻¹ @ y=38、10⁻² @ y=268，即 230 px/decade，
     value(row) = 10 ** (-1 - (row - 38) / 230)。
  3. 横轴刻度：底轴下方 384–386 行的暗列 → n=50..300 的 6 个 x 位置
     （87.5 / 153.5 / 220 / 286.5 / 352.5 / 419.5）。
  4. 取色：先统计饱和像素的众数色，得到 7 种曲线的准确 RGB（不是猜的）。
  5. 每个 x 位置取列 ±2 px 内同色像素行号的**中位数**（标记点比线粗，
     中位数对标记不敏感）；图例区（x≥215 且 y≤145）整体屏蔽。

输出：results/paper_fig7_digitized.json
"""
import json
import os

import numpy as np
from PIL import Image

FIG = "reports/fig_compare/paper/paper_fig07.png"
COLORS = {
    "AE": (31, 119, 179),
    "DeepAE": (255, 127, 14),
    "UNet": (44, 159, 44),
    "RadioUNet": (214, 38, 39),
    "Pix2pix+": (147, 102, 189),
    "RME-GAN": (140, 85, 75),
    "MFFGCN": (0, 0, 255),
}
XS = [87.5, 153.5, 220.0, 286.5, 352.5, 419.5]
COUNTS = [50, 100, 150, 200, 250, 300]

Y_TOP, Y_ONE = 38.0, 230.0      # 10^-1 所在行；每 decade 的像素数


def value(row):
    return 10.0 ** (-1.0 - (row - Y_TOP) / Y_ONE)


def main():
    a = np.asarray(Image.open(FIG).convert("RGB")).astype(int)
    res = {"counts": COUNTS, "series": {}}
    for name, rgb in COLORS.items():
        m = np.abs(a - np.array(rgb)).sum(2) < 28
        m[:21, :] = False              # 上框线以上
        m[383:, :] = False             # 下框线以下
        m[:, :71] = False
        m[:, 436:] = False             # 只看 panel a
        m[20:145, 215:437] = False     # 屏蔽图例
        ys, xs = np.where(m)
        vals, rows = [], []
        for x in XS:
            sel = ys[np.abs(xs - x) <= 2]
            if len(sel) == 0:
                vals.append(None)
                continue
            r = float(np.median(sel))
            rows.append(round(r, 1))
            vals.append(round(value(r), 5))
        res["series"][name] = {"nmse": vals, "row_pix": rows}
        print("%-10s %s" % (name, " ".join(
            "  None " if v is None else "%7.4f" % v for v in vals)))

    out = "results/paper_fig7_digitized.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
