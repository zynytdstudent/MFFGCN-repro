# -*- coding: utf-8 -*-
"""build_compare_page_v2.py - 重建"论文 vs 复现"逐图对照页（论文左、复现右，各自独立成图）。

背景：旧的 09 页里 Fig.9/10/14/15/16 的右图来自已经删掉的旧数据集接口
（setup="sq" / region_side / region_dist / impulse），和现在跑主表用的代码
不是同一套口径；Fig.15/16 还叠加了"按最大连通域抹建筑"的 bug。
所以右图必须按现在的 supp_eval.py 重算，并把纵轴对齐到论文的刻度范围，
否则"同一条曲线两个坐标系"看起来当然对不上。

产出：
  reports/fig_compare/v2_<key>.png   左右并排图（左论文原图 / 右复现独立成图）
  reports/15-论文图逐张对照.html
"""
import base64
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = r"D:\Projects\MFFGCN"
PC = os.path.join(ROOT, "reports", "fig_compare")
PAPER = os.path.join(PC, "paper")
FIGS = os.path.join(ROOT, "figs")
OUT_HTML = os.path.join(ROOT, "reports", "15-论文图逐张对照.html")
FONT = r"C:\Windows\Fonts\msyh.ttc"

H = 760
BAR = 44
GAP = 22
PAD = 12
WMAX = 2400

OK, WARN, BAD = "#1a7f4b", "#b26a00", "#b42318"


def font(sz):
    try:
        return ImageFont.truetype(FONT, sz)
    except Exception:
        return ImageFont.load_default()


def to_h(path, h):
    im = Image.open(path).convert("RGB")
    if im.height <= h:
        return im.copy()
    return im.resize((max(1, int(im.width * h / im.height)), h), Image.LANCZOS)


def side(paper_png, ours_png, out_name, p_txt, o_txt, chip=None, chip_color=OK):
    a = to_h(os.path.join(PAPER, paper_png), H)
    b = to_h(ours_png, H)
    total = a.width + GAP + b.width
    if total > WMAX:
        k = WMAX / total
        a = a.resize((int(a.width * k), int(a.height * k)), Image.LANCZOS)
        b = b.resize((int(b.width * k), int(b.height * k)), Image.LANCZOS)

    body = max(a.height, b.height)
    W = a.width + GAP + b.width
    canvas = Image.new("RGB", (W, BAR + PAD * 2 + body), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    top = BAR + PAD
    for x0, w in ((0, a.width), (a.width + GAP, b.width)):
        d.rectangle([x0, top - 4, x0 + w - 1, top + body + 4],
                    fill=(248, 249, 251), outline=(230, 233, 238), width=1)

    d.rectangle([0, 0, a.width, BAR], fill=(30, 34, 39))
    d.text((14, 8), p_txt, font=font(23), fill=(255, 255, 255))
    x2 = a.width + GAP
    d.rectangle([x2, 0, W, BAR], fill=(40, 78, 118))
    d.text((x2 + 14, 8), o_txt, font=font(23), fill=(255, 255, 255))
    if chip:
        f19 = font(19)
        tw = d.textlength(chip, font=f19)
        d.rectangle([W - tw - 26, 9, W - 8, BAR - 9], fill=chip_color)
        d.text((W - tw - 17, 14), chip, font=f19, fill=(255, 255, 255))

    canvas.paste(a, (0, top + (body - a.height) // 2))
    canvas.paste(b, (x2, top + (body - b.height) // 2))
    canvas.save(os.path.join(PC, out_name), optimize=True)
    print(out_name, canvas.size)
    return canvas.size


# key, 论文图, 我们的图, 标题, 状态, 说明
ROWS = [
    ("06a", "fig6_paper_a.png", os.path.join(FIGS, "fig6_ours_a.png"),
     "论文 Fig.6(a)　Setup 1 训练 loss", "我们的复现", "OK", "对齐",
     "口径一致；三条 loss 的下降段与平台位置一致，绝对高度受 batch 划分影响略有偏移。"),

    ("06b", "fig6_paper_b.png", os.path.join(FIGS, "fig6_ours_b.png"),
     "论文 Fig.6(b)　Setup 2 训练 loss", "我们的复现", "OK", "对齐",
     "同上。Setup 2 的 loss 平台比 Setup 1 高，两边都是这个关系。"),

    ("06c", "fig6_paper_c.png", os.path.join(FIGS, "fig6_ours_c.png"),
     "论文 Fig.6(c)　Setup 3 训练 loss", "我们的复现", "OK", "对齐",
     "同上。"),

    ("07", "paper_fig07.png", os.path.join(FIGS, "fig7_ours.png"),
     "论文 Fig.7　观测点数扫描（Setup 1）", "我们的复现（独立成图，格式照论文）",
     "WARN", "形状对齐·基线待修",
     "**右图只画我们的结果，不再与论文叠在一张图上**：两栏 (a) NMSE / (b) RMSE，"
     "7 条曲线、图例同样放右上角，轴标签与论文一致（Number of samples, 50–300）。"
     "横轴按原文 “number of observation samples” 取每图观测点数（旧版做成「训练城市数」，已作废），"
     "权重不动、只改推理时的观测点数，6 个档位全部单调下降。"
     "MFFGCN 与论文的相对偏差从 n=50 的 +19.5% 收敛到 n=300 的 −5.6%，与表 III 的 +5.5% 自洽。"
     "仍偏高的是两条 cGAN 基线（+94~162%，旧权重练废，正在重训）。"
     "<b>AE 已补齐</b>：原权重丢失、这次重训 50 epoch（val 在 ep5 最优、之后回升，取 best），"
     "曲线 0.0840→0.0778（−7.4%）。论文的 AE 是 0.0625→0.0184（−70%）——"
     "**这条线仍然平**，且我们另跑了一版带 obs 残差 skip 的 AE（test 0.0961 vs 0.0925），"
     "没有变好，说明「少一根 skip」的推断不成立，是结构性差异（见报告 16 §5.1.1）。"
     "DeepAE 同理（0.0846→0.0800，论文 −68%）。这两条线不参与任何结论。"),

    ("08", "paper_fig08.png", os.path.join(FIGS, "fig8_ours.png"),
     "论文 Fig.8　受限区域 7 方法", "我们的复现（独立成图，格式照论文）",
     "OK", "已补齐 7×2",
     "旧版那 7 个格子里的 n/a 不是「没跑」，是**图没重画**——AE-s2/s3、Pix2pix+-s2/s3、"
     "RME-GAN-s3 的 test_metrics.json 都在（test 全量 8080 张），只有 bl_ae_s1 的权重丢了。"
     "现在 7 个方法 × 2 个 setup 共 14 根柱子全部有数，图例同样放右上角、纵轴与论文同尺度"
     "（NMSE 0–0.145 / RMSE 0–0.10）。"
     "MFFGCN 0.00507/0.00536 与论文 0.0050/0.0046 基本重合；"
     "**DeepAE 仍是唯一方向反了的一格**（我们 0.100/0.079，论文 0.053/0.052）——"
     "它没有 skip 连接，属于结构推定偏差，与 Fig.7 里那条几乎水平的 DeepAE 是同一个原因。"
     "另外两条 cGAN 基线（Pix2pix+ 0.0134/0.0183、RME-GAN 0.0123/0.0148）仍比论文高约 2 倍，"
     "同 Fig.9/10 的 RME-GAN 问题，重训完成后这一栏会整体下移。"),

    ("09", "paper_fig09.png", os.path.join(FIGS, "fig9_ours.png"),
     "论文 Fig.9　盲区离图中心的距离", "我们的复现（独立成图，格式照论文）", "OK", "已修正采样",
     "右图只画我们的三个模型（分组柱，NMSE/RMSE 两栏，图例同样放右上角）。"
     "**这张图重跑过，数字与上一版完全不同**：原来那组只取了前 200 张图（约 2.5 个城市），"
     "E[t²]=0.0448，只有全集 0.0688 的 65%。改成等间隔铺满测试集后（`--span`，同样 200 张，"
     "E=0.0685）：<b>MFFGCN 与论文的差从 +35% 收到 +10%、RadioUNet 从 +34% 变成 −7%（优于论文）</b>。"
     "只剩 <b>RME-GAN 高 2.26 倍</b>是真问题（cGAN 基线训练失败，正在重训）。"
     "形状没对上的是 MFFGCN 本身：论文随距离下降 14.5%，我们几乎不变（0.00486→0.00495），待查。"
     "口径差的定量分解见报告 17。"),

    ("10", "paper_fig10.png", os.path.join(FIGS, "fig10_ours.png"),
     "论文 Fig.10　盲区边长", "我们的复现（独立成图，格式照论文）", "OK", "已修正采样",
     "同 Fig.9，重跑后 MFFGCN +14%、RadioUNet −4%、RME-GAN +145%。"
     "<b>趋势逐点吻合</b>：论文 MFFGCN 随边长 0.00426→0.00482（+13.1%），"
     "我们 0.00486→0.00529（+8.8%）；RadioUNet 论文几乎不动，我们也几乎不动。"
     "模型对几何的响应复现对了。注意 RMSE 栏不能直接和论文读成\"一样好\"——"
     "两边 E[t²] 差 1.29 倍，我们的 RMSE 天然被放大 1.14 倍。"),

    ("11", "paper_fig11.png", os.path.join(FIGS, "fig11_ours.png"),
     "论文 Fig.11　6 方法 × 3 setup 可视化", "我们的复现（同 6 方法同 3 setup）", "OK", "定性一致",
     "定性图：盲区位置、建筑遮挡、能量衰减的整体形态一致；逐图像素差无法量化，只能肉眼比对。"),

    ("12", "paper_fig12.png", os.path.join(FIGS, "fig12_ours.png"),
     "论文 Fig.12　图结构与 GCN 精修", "我们的复现（16×16 节点 kNN 图）", "OK", "结构一致",
     "示意性质：节点网格、kNN 连边、精修前后对比的画法一致。"),

    ("13", "paper_fig13.png", os.path.join(FIGS, "fig13_ours.png"),
     "论文 Fig.13　传播步数 × 邻居数", "我们的复现（5 层 GCN，步数 1–5）", "OK", "高度吻合",
     "这条是全部图里最吻合的：三条曲线的下降趋势、4>8>16 邻居的排序、"
     "以及 3 步之后饱和的拐点都一样，量级 4.4e-3 vs 论文 4.7e-3。"),

    ("14", "paper_fig14.png", os.path.join(FIGS, "fig14_ours.png"),
     "论文 Fig.14　冲激噪声", "我们的复现（独立成图，格式照论文）", "BAD", "结论相反",
     "**唯一真分歧，重跑后依然成立**（这批也换成了 `--span` 采样）。"
     "论文 MFFGCN 最抗噪：干净 0.0066 → 1% 噪声 0.0123，只劣化 1.9×；"
     "我们最敏感：0.0053 → 0.0735，劣化 <b>13.8×</b>。而 RadioUNet 我们 ×6.2、论文 ×7.6，"
     "RME-GAN 我们 ×2.9、论文 ×5.2——这两条反而接近。"
     "机理已定位在报告 16：把 1% 翻转拆成\"只加假建筑\"与\"只删真建筑\"，"
     "前者让 MFFGCN ×10.7、后者只 ×1.8。假建筑在动态图里凭空造出遮挡边，"
     "误差被直接推进精修段——这是架构对噪声图敏感度的真实差异，不是口径。"),

    ("15", "paper_fig15.png", os.path.join(FIGS, "fig15_ours.png"),
     "论文 Fig.15　环境图降分辨率", "我们的复现（独立成图，格式照论文）", "WARN", "趋势对、劣化偏慢",
     "口径按论文原文「下采样-上采样后重新二值化」，全部在 Setup 3 下做。"
     "**趋势完全一致**：分辨率越低误差越大，25% 档是拐点。"
     "但相对劣化倍数差很远：论文 25% 档是干净基线的 <b>13.6×</b>，我们只有 <b>5.3×</b>。"
     "这条与采样无关，是 env 图降采样的实现与论文不同，待查。"),

    ("16", "paper_fig16.png", os.path.join(FIGS, "fig16_ours.png"),
     "论文 Fig.16　缺失建筑", "我们的复现（独立成图，格式照论文）", "OK", "起点对齐、斜率偏缓",
     "重跑（`--span`）后 <b>1 栋那格几乎重合：0.02192 vs 论文 0.02209，差 0.8%</b>，"
     "排序也对（MFFGCN < RadioUNet < RME-GAN）。"
     "但 2–4 栋我们系统性低 18–24%，即衰减比论文慢——这一条旧值同样低 26%，"
     "重跑没改善，是真实差异不是采样。抹除方式是「随机连通域、逐级嵌套」，"
     "1→4 栋实际抹掉 0.85%/1.51%/2.08%/2.81% 面积。"),
]

CHIP = {"OK": (OK, "对齐"), "WARN": (WARN, "有偏差"), "BAD": (BAD, "不一致")}


def main():
    blocks = []
    for key, p, o, pt, ot, lvl, chip, desc in ROWS:
        color, _ = CHIP[lvl]
        side(p, o, "v2_%s.png" % key, pt, ot, chip, color)
        blocks.append((key, chip, desc, color))

    css = """
*{box-sizing:border-box}
body{margin:0;background:#eef0f3;color:#15181c;
 font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:30px 18px 60px}
h1{font-size:22px;margin:0 0 6px}
.sub{font-size:13px;color:#5b636e;margin-bottom:18px;line-height:1.75}
.card{background:#fff;border-radius:10px;padding:14px 16px 16px;margin:0 0 18px;
 box-shadow:0 1px 3px rgba(16,24,40,.08)}
.card img{width:100%;display:block;border-radius:6px;cursor:zoom-in}
.cap{font-size:13.2px;color:#3f4650;line-height:1.85;margin-top:10px}
.tag{display:inline-block;font-size:12px;color:#fff;border-radius:4px;
 padding:1px 7px;margin-right:8px;vertical-align:1px}
.zoom{position:fixed;inset:0;background:rgba(8,10,13,.95);z-index:99;
 display:flex;align-items:center;justify-content:center;cursor:zoom-out}
.zoom img{max-width:98vw;max-height:97vh;width:auto;border-radius:0}
.note{background:#fff;border-left:3px solid #b42318;border-radius:6px;
 padding:12px 16px;font-size:13px;color:#41474f;line-height:1.9;margin-bottom:18px}
"""

    parts = []
    for key, chip, desc, color in blocks:
        b64 = base64.b64encode(open(os.path.join(PC, "v2_%s.png" % key), "rb").read()).decode()
        parts.append(
            '<div class="card"><img src="data:image/png;base64,%s">'
            '<div class="cap"><span class="tag" style="background:%s">%s</span>%s</div></div>'
            % (b64, color, chip, desc))

    html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>论文 vs 复现 · 逐图对照</title>
<style>__CSS__</style></head><body><div class="wrap">
<h1>论文 vs 复现 · 逐图对照</h1>
<div class="sub">左＝论文原图；右＝我们的复现，<b>各自独立成图、不叠画</b>。
复现图的曲线条数、图例位置、轴名、横轴档位都照论文排，左右并排即可逐点对读。点图放大。</div>
<div class="note">
<b>这一版替换掉了 09 页。</b>09 页里 Fig.9/10/14/15/16 的右图来自已经删除的旧数据集接口
（<code>setup="sq"</code>、<code>region_side</code>、<code>region_dist</code>、<code>impulse</code>），
和现在跑主表用的不是同一套代码；Fig.15/16 还叠加了「按最大连通域抹建筑」的 bug
（"1 栋"实际抹掉 5.19% 的图面积）。所以那几张图看着完全对不上是必然的。
本页全部用 <code>supp_eval.py</code> 在 <b>Setup 3</b> 下重算（n=200），口径与论文原文逐条核对过。
</div>
__CARDS__
</div></body>
<script>
document.querySelectorAll('img').forEach(function(im){
 im.addEventListener('click',function(){
  var b=document.createElement('div');b.className='zoom';
  var g=document.createElement('img');g.src=im.src;b.appendChild(g);
  b.addEventListener('click',function(){b.remove();});
  document.body.appendChild(b);});});
</script></html>""".replace("__CSS__", css).replace("__CARDS__", "\n".join(parts))

    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("wrote", OUT_HTML, round(os.path.getsize(OUT_HTML) / 1048576, 2), "MB")


if __name__ == "__main__":
    main()
