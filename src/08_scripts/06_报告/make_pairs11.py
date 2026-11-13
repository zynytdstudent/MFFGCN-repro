# -*- coding: utf-8 -*-
"""把论文原图与我们的对应图上下拼成一张对照图。"""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = r"D:\Projects\MFFGCN"
PAPER = os.path.join(ROOT, "reports", "fig_compare", "paper")
OUT = os.path.join(ROOT, "reports", "fig_compare")
W = 1500
BAR = 52
FONT = r"C:\Windows\Fonts\msyh.ttc"


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except Exception:
        return ImageFont.load_default()


def fit(path, width):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = width / w
    if h * scale > 1700:                 # 竖长图按高度收，避免网页里变成一条细长条
        scale = 1700 / h
        w2, h2 = int(w * scale), 1700
    else:
        w2, h2 = width, max(1, int(h * scale))
    return im.resize((w2, h2), Image.LANCZOS)


def label(dr, y, text, color, width):
    dr.rectangle([0, y, width, y + BAR], fill=color)
    dr.text((22, y + 13), text, font=font(24), fill=(255, 255, 255))
    return y + BAR


def stack(paper_name, ours_path, out_name, paper_txt, ours_txt):
    a = fit(os.path.join(PAPER, paper_name), W)
    b = fit(ours_path, W) if ours_path else None
    pad = 16
    H = BAR + a.height + pad * 2 + (BAR + b.height + pad if b else 0)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    y = label(dr, 0, paper_txt, (31, 35, 40), W)
    canvas.paste(a, ((W - a.width) // 2, y + pad))
    if b:
        y2 = y + pad + a.height + pad
        y2 = label(dr, y2, ours_txt, (46, 90, 136), W)
        canvas.paste(b, ((W - b.width) // 2, y2 + pad))
    canvas.save(os.path.join(OUT, out_name), optimize=True)
    print(out_name, canvas.size)


# 论文图 → 我们的图；没有对应就传 None
F = os.path.join(ROOT, "figs")
R = os.path.join(F, "report")
PAIRS = [
    ("paper_fig06.png", os.path.join(F, "fig6_replica_ours.png"), "cmp_p06.png",
     "PAPER  Fig.6  training loss, 3 setups (MFFGCN vs RadioUNet)",
     "OURS  same two models, 50 epochs, our loss definition"),
    ("paper_fig07.png", None, "cmp_p07.png",
     "PAPER  Fig.7  NMSE / RMSE vs number of samples (50-300)", None),
    ("paper_fig08.png", os.path.join(R, "figA_main.png"), "cmp_p08.png",
     "PAPER  Fig.8  NMSE / RMSE in restricted regions, 7 methods x Setup 2/3",
     "OURS  same ranking, 3 setups x 7 methods (older metric version)"),
    ("paper_fig09.png", None, "cmp_p09.png",
     "PAPER  Fig.9  NMSE / RMSE vs distance between restricted region and center", None),
    ("paper_fig10.png", None, "cmp_p10.png",
     "PAPER  Fig.10  NMSE / RMSE vs side length of restricted region", None),
    ("paper_fig11.png", os.path.join(F, "fig5_quali.png"), "cmp_p11.png",
     "PAPER  Fig.11  6 methods x 3 setups: estimation + deviation",
     "OURS  MFFGCN only: GT / obs / input / coarse / refined / error"),
    ("paper_fig12.png", os.path.join(R, "figG_residual.png"), "cmp_p12.png",
     "PAPER  Fig.12  GT, zoom-in, graph nodes, GCN-refined result",
     "OURS  coarse vs refined NMSE (no graph-node drawing yet)"),
    ("paper_fig13.png", os.path.join(R, "figE_ablation.png"), "cmp_p13.png",
     "PAPER  Fig.13  NMSE vs propagation steps x neighbours (4/8/16)",
     "OURS  component ablation, 4 variants (no step/neighbour sweep)"),
    ("paper_fig14.png", os.path.join(R, "figC_noise.png"), "cmp_p14.png",
     "PAPER  Fig.14  impulse noise intensity 0.1%-1.0%, 3 methods",
     "OURS  SNR clean/30/20/10 dB, MFFGCN x 3 setups"),
    ("paper_fig15.png", None, "cmp_p15.png",
     "PAPER  Fig.15  impact of map resolution", None),
    ("paper_fig16.png", os.path.join(R, "figD_budget.png"), "cmp_p16.png",
     "PAPER  Fig.16  number of missing buildings 1-4, 3 methods",
     "OURS  observation budget x0.5-x4 (different axis, same idea)"),
]

for p, o, out, pt, ot in PAIRS:
    stack(p, o, out, pt, ot)
