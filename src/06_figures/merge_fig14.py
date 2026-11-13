# -*- coding: utf-8 -*-
"""把页面上那张 Fig.14 四联图与新的诊断柱状图拼成一张。

为什么不用 matplotlib 重排整张：
    上半张（论文 Fig.14 原始截图 + 我们的复现）是之前逐像素对好的资产，
    重画会丢细节；下半张是新诊断。这里只做三件事：
      1. 把右上角"结论相反"红角标改成现在的结论（琥珀色）；
      2. 中间插一条深蓝标题带，交代下半张的口径（同一份权重、只换注入方式）；
      3. 底部加一行结论文本。

用法：先在对照页上把 Fig.14 那张图抠下来存成 figs/fig14_onpage.png，
       再跑 python merge_fig14.py，输出 figs/fig14_merged.png
"""
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
# 脚本可能在 src/06_figures/，也可能被平铺到服务器 src/，两种都要能找到 figs/
ROOT = HERE
for _rel in ("..", os.path.join("..", "..")):
    _c = os.path.abspath(os.path.join(HERE, _rel))
    if os.path.isdir(os.path.join(_c, "figs")):
        ROOT = _c
        break
TOP = os.path.join(ROOT, "figs", "fig14_onpage.png")
BOT = os.path.join(ROOT, "figs", "fig14_diag.png")
OUT = os.path.join(ROOT, "figs", "fig14_merged.png")

HDR = (40, 78, 118)          # 原图标题带的深蓝
AMBER = (178, 106, 0)        # 与页面药丸同色系
INK = (32, 36, 42)
FONT = "C:/Windows/Fonts/msyh.ttc"


def font(sz, bold=False):
    p = "C:/Windows/Fonts/msyhbd.ttc" if bold else FONT
    return ImageFont.truetype(p, sz, index=0)


def main():
    top = Image.open(TOP).convert("RGB")
    bot = Image.open(BOT).convert("RGB")
    W = max(top.width, bot.width)
    band_h, foot_h, gap = 40, 58, 10

    canvas = Image.new("RGB", (W, top.height + band_h + gap + bot.height
                               + foot_h), "white")
    d = ImageDraw.Draw(canvas)

    # ---- 1. 上半张：先在原图上覆盖旧角标，再整体贴进画布
    #      （顺序错了会被 paste 覆盖回去，这里踩过一次）
    dt = ImageDraw.Draw(top)
    dt.rectangle([2290, 0, top.width, 42], fill=HDR)
    tag = "已定位·注入形态"
    f = font(17)
    tw = dt.textlength(tag, font=f)
    x0 = top.width - tw - 22
    dt.rounded_rectangle([x0 - 8, 9, x0 + tw + 6, 34], radius=4, fill=AMBER)
    dt.text((x0, 11), tag, font=f, fill="white")
    canvas.paste(top, (0, 0))

    # ---- 2. 中间标题带
    y = top.height
    d.rectangle([0, y, W, y + band_h], fill=HDR)
    t1 = "第三轮诊断：同一份 s3 权重，只换噪声注入口径（n=60，不重训）"
    d.text((14, y + 9), t1, font=font(19, True), fill="white")
    t2 = "缺建筑 ×1.1–1.6 几乎无害   →   凭空多出的假建筑 ×7.4–15.8 才是杀手"
    tw2 = d.textlength(t2, font=font(16))
    d.text((W - tw2 - 14, y + 11), t2, font=font(16), fill=(255, 214, 150))

    # ---- 3. 下半张
    canvas.paste(bot, ((W - bot.width) // 2, y + band_h + gap))

    # ---- 4. 结论脚注
    fy = y + band_h + gap + bot.height + 14
    foot = ("结论：1% 双向翻转的劣化被放大到 ×13.6，但加一层 3×3 中值滤波就回到 ×2.91，"
            "落进论文 Fig.14 的 ×1.9–2.45 同量级 —— 分歧在噪声注入形态，不在模型结构。")
    d.text((14, fy), foot, font=font(16), fill=INK)

    canvas.save(OUT)
    print("saved", OUT, canvas.size)


if __name__ == "__main__":
    main()
