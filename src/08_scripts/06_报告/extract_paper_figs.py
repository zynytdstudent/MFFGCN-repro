# -*- coding: utf-8 -*-
"""从 paper.pdf 抽出 Fig.6-Fig.16 的实验图（矢量/位图混合），供逐张对照使用。"""
import os
import re
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CAND = [os.path.join(ROOT, "data", "paper.pdf"),
        os.path.join(HERE, ".tmp_srv", "paper.pdf"),
        os.path.expanduser(r"~\WorkBuddy\2026-10-22-23-58-31\.tmp_srv\paper.pdf")]
SRC = next((p for p in CAND if os.path.exists(p)), CAND[0])
OUT = os.path.join(ROOT, "reports", "fig_compare", "paper")
DPI = 300

os.makedirs(OUT, exist_ok=True)
doc = pymupdf.open(SRC)


def cap_blocks(page):
    res = {}
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        txt = "".join(s["text"] for ln in b["lines"] for s in ln["spans"]).strip()
        m = re.match(r"Fig\.\s*(\d+)\.", txt)
        if m:
            res.setdefault(int(m.group(1)), b["bbox"])
    return res


def draw_union(page, upper, lower, x0, x1):
    box = None
    for d in page.get_drawings():
        r = d["rect"]
        if r.width < 1 and r.height < 1:
            continue
        if r.y1 < upper - 1 or r.y0 > lower + 1:
            continue
        if r.x1 < x0 - 20 or r.x0 > x1 + 20:
            continue
        box = pymupdf.Rect(r) if box is None else (box | r)
    return box


report = []
for pno in range(doc.page_count):
    page = doc[pno]
    caps = cap_blocks(page)
    imgs = []
    for im in page.get_images(full=True):
        try:
            imgs.append(page.get_image_bbox(im))
        except Exception:
            pass
    order = sorted(caps.items(), key=lambda kv: kv[1][1])
    for i, (n, bbox) in enumerate(order):
        if n < 6 or n > 16:
            continue
        cap_y0, cx0, cx1 = bbox[1], bbox[0], bbox[2]
        upper = order[i - 1][1][3] if i > 0 else 40.0
        # 图注整体落在左栏或右栏时，把搜索区间锁在该栏内，避免把邻栏表格/正文吃进来
        if cx0 > 290:
            cx0, cx1 = max(cx0, 286.0), 560.0
            side = "right"
        elif cx1 < 290:
            cx0, cx1 = 30.0, 288.0
            side = "left"
        else:                      # 跨栏图，放宽到整版心
            cx0, cx1 = 30.0, 560.0
            side = "both"

        box = None
        # 1) 首选位图（Fig.11 / Fig.12 是整块位图）
        for ib in imgs:
            if ib.y1 <= cap_y0 + 2 and ib.y1 >= upper - 2 and ib.height > 60:
                box = pymupdf.Rect(ib) if box is None else (box | ib)
        # 2) 矢量绘图
        dv = draw_union(page, upper, cap_y0, cx0, cx1)
        if dv is not None and dv.height > 20:
            box = dv if box is None else (box | dv)
        # 3) 兜底：整个栏区间
        if box is None or box.height < 40:
            box = pymupdf.Rect(cx0 - 8, upper + 2, cx1 + 8, cap_y0 - 2)

        box = pymupdf.Rect(box.x0 - 5, box.y0 - 5, box.x1 + 5, cap_y0 - 2)
        if side == "right":                  # 右栏图：左侧须让开邻栏表格边线
            box.x0 = max(box.x0, 302.0)
        box &= pymupdf.Rect(cx0 - 34, 0, cx1 + 34, page.rect.height)
        box &= page.rect
        pix = page.get_pixmap(clip=box, dpi=DPI)
        fn = os.path.join(OUT, "paper_fig%02d.png" % n)
        pix.save(fn)
        report.append((n, pno + 1, tuple(round(v) for v in box), "%dx%d" % (pix.width, pix.height)))

for r in sorted(report):
    print(r)
print("files:", len(os.listdir(OUT)))
