"""把论文原图与复现图上下拼接，用于严格对照。"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

PAPER = r"D:\Projects\MFFGCN\reports\ppt_work\assets\paper\figs"
OURS = r"D:\Projects\MFFGCN\figs"
OUT = r"D:\Projects\MFFGCN\reports\fig_compare"
os.makedirs(OUT, exist_ok=True)

W = 1400          # 统一宽度
GAP = 18
BAR = 34


def load(path, width=W):
    im = Image.open(path).convert("RGB")
    h = int(im.height * width / im.width)
    return im.resize((width, h), Image.LANCZOS)


def label(text, width=W, color=(20, 20, 20)):
    im = Image.new("RGB", (width, BAR), (250, 250, 250))
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 20)
    except OSError:
        f = ImageFont.load_default()
    d.text((10, 6), text, fill=color, font=f)
    return im


def stack(items, out_path):
    ims = []
    for kind, payload in items:
        if kind == "label":
            ims.append(label(payload))
        else:
            ims.append(load(payload))
    total_h = sum(i.height for i in ims) + GAP * (len(ims) - 1)
    canvas = Image.new("RGB", (W, total_h), (255, 255, 255))
    y = 0
    for i in ims:
        canvas.paste(i, (0, y))
        y += i.height + GAP
    canvas.save(out_path, quality=95)
    print("saved", out_path, canvas.size)


# Fig 6：训练 loss 曲线
stack([
    ("label", "PAPER  Fig.6  (100 epochs)"),
    ("img", os.path.join(PAPER, "fig06.png")),
    ("label", "OURS  Fig.6 replica  (50 epochs, our loss definition)"),
    ("img", os.path.join(OURS, "fig6_replica_ours.png")),
], os.path.join(OUT, "cmp_fig06_loss.png"))

# Fig 11：定性可视化
stack([
    ("label", "PAPER  Fig.11  (6 methods x 3 setups, Estimation + Deviation)"),
    ("img", os.path.join(PAPER, "fig11.png")),
    ("label", "OURS  fig5_quali  (1 method x 3 setups, no deviation colormap)"),
    ("img", os.path.join(OURS, "fig5_quali.png")),
], os.path.join(OUT, "cmp_fig11_quali.png"))
