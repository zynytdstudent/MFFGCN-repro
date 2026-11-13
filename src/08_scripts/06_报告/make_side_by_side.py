# -*- coding: utf-8 -*-
"""左右对比图：左边论文原图，右边我们的复现图，等高并排。

严格口径（用户 17:25 要求）：
  1. 图型必须与论文一致（折线对折线、柱状对柱状、图像网格对图像网格）；
  2. 必须是同一个实验，不能拿别的实验的结果凑数。
不满足就右侧留「未复现」占位卡，并写明这张图需要什么实验。
"""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = r"D:\Projects\MFFGCN"
PAPER = os.path.join(ROOT, "reports", "fig_compare", "paper")
OUT = os.path.join(ROOT, "reports", "fig_compare")
F = os.path.join(ROOT, "figs")
R = os.path.join(F, "report")
FONT = r"C:\Windows\Fonts\msyh.ttc"

H = 780          # 图区目标高度
BAR = 46         # 顶部标签条
GAP = 26         # 左右间距
PAD = 14         # 上下留白
PH_W = 620       # 「未复现」占位框宽度
WMAX = 2400      # 整张对比图的最大宽度


def font(sz):
    try:
        return ImageFont.truetype(FONT, sz)
    except Exception:
        return ImageFont.load_default()


def to_h(path, h):
    im = Image.open(path).convert("RGB")
    if im.height <= h:
        return im.copy()
    w = max(1, int(im.width * h / im.height))
    return im.resize((w, h), Image.LANCZOS)


def side(paper_name, ours_path, out_name, p_txt, o_txt, miss=None):
    a = to_h(os.path.join(PAPER, paper_name), H)
    if ours_path:
        b = to_h(ours_path, H)
        b_txt = o_txt
        b_fill = (46, 90, 136)
    else:
        PH = max(a.height, 520)
        b = Image.new("RGB", (PH_W, PH), (246, 247, 249))
        d0 = ImageDraw.Draw(b)
        d0.rectangle([0, 0, PH_W - 1, PH - 1], outline=(214, 218, 224), width=2)
        # 标题
        t1 = "本轮未复现"
        d0.text(((PH_W - d0.textlength(t1, font=font(34))) / 2, PH // 2 - 92),
                t1, font=font(34), fill=(120, 128, 138))
        # 缺的实验说明，按行折
        lines, cur = [], ""
        f18 = font(19)
        for ch in (miss or ""):
            if d0.textlength(cur + ch, font=f18) > PH_W - 70:
                lines.append(cur)
                cur = ch
            else:
                cur += ch
        if cur:
            lines.append(cur)
        y = PH // 2 - 20
        for ln in lines[:5]:
            d0.text(((PH_W - d0.textlength(ln, font=f18)) / 2, y), ln,
                    font=f18, fill=(122, 130, 140))
            y += 31
        b_txt = "我们的复现"
        b_fill = (150, 156, 164)

    # 整张图太宽就整体等比缩小，两边一起缩，保持等高
    total = a.width + GAP + b.width
    if total > WMAX:
        k = WMAX / total
        a = a.resize((max(1, int(a.width * k)), max(1, int(a.height * k))), Image.LANCZOS)
        b = b.resize((max(1, int(b.width * k)), max(1, int(b.height * k))), Image.LANCZOS)

    body = max(a.height, b.height)
    W = a.width + GAP + b.width
    canvas = Image.new("RGB", (W, BAR + PAD * 2 + body), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    top = BAR + PAD - 6
    bot = top + body + 12
    for x0, w in ((0, a.width), (a.width + GAP, b.width)):
        d.rectangle([x0, top, x0 + w - 1, bot], fill=(247, 248, 250),
                    outline=(228, 231, 236), width=1)

    d.rectangle([0, 0, a.width, BAR], fill=(31, 35, 40))
    d.text((16, 9), p_txt, font=font(25), fill=(255, 255, 255))
    x2 = a.width + GAP
    d.rectangle([x2, 0, W, BAR], fill=b_fill)
    d.text((x2 + 16, 9), b_txt, font=font(25), fill=(255, 255, 255))

    canvas.paste(a, (0, BAR + PAD + (body - a.height) // 2))
    canvas.paste(b, (x2, BAR + PAD + (body - b.height) // 2))
    canvas.save(os.path.join(OUT, out_name), optimize=True)
    print(out_name, canvas.size)


# 第三项 ours=None 表示未复现，第六项写缺什么
PAIRS = [
    ("fig6_paper_a.png", os.path.join(F, "fig6_ours_a.png"), "sbs_06a.png",
     "论文 Fig.6(a)　Setup 1 训练 loss", "我们的复现　Setup 1", None),
    ("fig6_paper_b.png", os.path.join(F, "fig6_ours_b.png"), "sbs_06b.png",
     "论文 Fig.6(b)　Setup 2 训练 loss", "我们的复现　Setup 2", None),
    ("fig6_paper_c.png", os.path.join(F, "fig6_ours_c.png"), "sbs_06c.png",
     "论文 Fig.6(c)　Setup 3 训练 loss", "我们的复现　Setup 3", None),
    ("paper_fig07.png", os.path.join(F, "fig7_ours.png"), "sbs_07.png",
     "论文 Fig.7　样本数扫描（折线）", "我们的复现　同 7 方法同轴折线", None),
    ("paper_fig08.png", os.path.join(F, "fig8_ours.png"), "sbs_08.png",
     "论文 Fig.8　受限区域 7 方法（柱状）", "我们的复现　同指标同轴柱状", None),
    ("paper_fig09.png", os.path.join(F, "fig9_ours.png"), "sbs_09.png",
     "论文 Fig.9　盲区到中心距离（柱状）", "我们的复现　同三方法同轴柱状", None),
    ("paper_fig10.png", os.path.join(F, "fig10_ours.png"), "sbs_10.png",
     "论文 Fig.10　盲区边长（柱状）", "我们的复现　同三方法同轴柱状", None),
    ("paper_fig11.png", os.path.join(F, "fig11_ours.png"), "sbs_11.png",
     "论文 Fig.11　6 方法 × 3 setup 可视化", "我们的复现　同 6 方法同 3 setup", None),
    ("paper_fig12.png", os.path.join(F, "fig12_ours.png"), "sbs_12.png",
     "论文 Fig.12　图结构与 GCN 精修（图像）", "我们的复现　同为 16×16 节点 kNN 图", None),
    ("paper_fig13.png", os.path.join(F, "fig13_ours.png"), "sbs_13.png",
     "论文 Fig.13　传播步数 × 邻居数（折线）", "我们的复现　5 层 GCN，步数 1–5", None),
    ("paper_fig14.png", os.path.join(F, "fig14_ours.png"), "sbs_14.png",
     "论文 Fig.14　冲激噪声（折线）", "我们的复现　同三方法同强度折线", None),
    ("paper_fig15.png", os.path.join(F, "fig15_ours.png"), "sbs_15.png",
     "论文 Fig.15　地图降采样率（折线）", "我们的复现　同三方法同轴折线", None),
    ("paper_fig16.png", os.path.join(F, "fig16_ours.png"), "sbs_16.png",
     "论文 Fig.16　缺失建筑（柱状）", "我们的复现　1–4 栋建筑从输入地图抹掉", None),
]

for p, o, out, pt, ot, ms in PAIRS:
    side(p, o, out, pt, ot, ms)
