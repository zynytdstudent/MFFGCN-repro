# -*- coding: utf-8 -*-
"""把论文 Fig.6 与我们的 loss 图各切成 (a)(b)(c) 三个子图，便于逐 setup 并排对比。

分界点由「全白竖缝」探测得到（见探针输出），这里写死以免每次重算。
"""
import os
from PIL import Image

ROOT = r"D:\Projects\MFFGCN"
PAPER = os.path.join(ROOT, "reports", "fig_compare", "paper")
FIGS = os.path.join(ROOT, "figs")

# 论文：缝 641-720 / 1339-1419；我们：缝 723-741 / 1455-1479
CUTS = {
    os.path.join(PAPER, "paper_fig06.png"): [0, 680, 1379, 2058],
    os.path.join(FIGS, "fig6_replica_ours.png"): [0, 732, 1467, 2213],
}


def trim(im, bg_thr=248):
    """去掉四周近白边。"""
    g = im.convert("L")
    px = g.load()
    w, h = g.size
    l, r, t, b = 0, w, 0, h
    while l < w and all(px[l, y] > bg_thr for y in range(0, h, 3)):
        l += 1
    while r > l and all(px[r - 1, y] > bg_thr for y in range(0, h, 3)):
        r -= 1
    while t < h and all(px[x, t] > bg_thr for x in range(0, w, 3)):
        t += 1
    while b > t and all(px[x, b - 1] > bg_thr for x in range(0, w, 3)):
        b -= 1
    return im.crop((max(0, l - 4), max(0, t - 4), min(w, r + 4), min(h, b + 4)))


for src, xs in CUTS.items():
    im = Image.open(src).convert("RGB")
    tag = "paper" if "paper/" in src.replace("\\", "/") else "ours"
    for k in range(3):
        y0 = 42 if tag == "ours" else 0      # 切掉我们图顶部的总标题，避免被分栏截断
        part = trim(im.crop((xs[k], y0, xs[k + 1], im.height)))
        name = "fig6_%s_%s.png" % (tag, "abc"[k])
        out = os.path.join(PAPER if tag == "paper" else FIGS, name)
        part.save(out, optimize=True)
        print(name, part.size)
