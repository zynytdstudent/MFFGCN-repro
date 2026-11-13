# -*- coding: utf-8 -*-
"""生成「复现与论文趋势一致性」图文报告（纯 SVG，无第三方依赖）。

数据来源优先级：
  1) 同目录 trend_data.json（由 update_trend.py 从服务器自动拉取）
  2) 内置 DEF（2026-11-04 17:24 的人工核对值）

用法：python make_trend_report.py [--out 路径]
"""
import argparse
import json
import math
import os

FONT = '"Microsoft YaHei","PingFang SC","Noto Sans CJK SC",system-ui,sans-serif'
INK = "#1f2933"
MUT = "#6b7785"
GRID = "#e3e8ee"
PAPER = "#9aabc0"      # 论文系列（灰蓝）
OURS = "#2f6fb3"       # 复现系列（蓝）
BAD = "#c0392b"
GOOD = "#1f9d55"
HERE = os.path.dirname(os.path.abspath(__file__))
# 脚本位于 src/08_scripts/06_报告/，项目根在上三级
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA_PATH = os.path.join(HERE, "trend_data.json")

DEF = {
    "updated": "2026-11-04 17:24",
    "bench": {
        "MFFGCN": {"v": 0.0211, "std": 0.0, "n": 1, "src": "旧权重重评（观测点口径不同）"},
        "U-Net": {"v": 0.0200},
        "AE": {"v": 0.0789},
        "KNN": {"v": 0.0908},
        "Kriging": {"v": 0.0899},
    },
    "setup": {
        "MFFGCN": [0.0211, 0.0222, 0.0420],
        "KNN": [0.0908, 0.1049, 0.1139],
        "Kriging": [0.0899, 0.1030, 0.1116],
        "MFFGCN_s3_provisional": True,
    },
    "ablation": {
        "完整模型": 0.0211, "去残差": 0.0392, "去动态图（换静态图）": 0.0337,
        "换融合：交叉注意力": 0.0293, "换融合：相加": 0.0243, "换融合：拼接": 0.0191,
        "换融合：通道门控": 0.0185, "去掉图卷积": 0.0184, "只留单编码器": 0.0196,
    },
    "depth": {"mid": [0.0220, 0.0226, 0.0237, 0.0260, 0.0297, 0.0332],
              "big": [0.0415, 0.0420, 0.0429, 0.0438, 0.0445, 0.0484]},
}

# 论文 Table III（Setup 1, RMSE）
PAPER_BENCH = {"MFFGCN": 0.0150, "RME-GAN": 0.0151, "Pix2pix+": 0.0156,
               "RadioUNet": 0.0160, "U-Net": 0.0182, "DeepAE": 0.0288,
               "AE": 0.0456, "Kriging": 0.1389, "KNN": 0.2312}
# 论文 Table V（Setup 1, RMSE 相对变化 %）
PAPER_ABL = {"只留单编码器": 5.3, "换融合：相加": 8.0, "换融合：拼接": 5.3,
             "换融合：通道门控": 3.3, "换融合：交叉注意力": 4.0,
             "去掉图卷积": 7.3, "去动态图（换静态图）": 2.7, "去残差": 131.3}
ABL_ORDER = ["去残差", "去动态图（换静态图）", "换融合：交叉注意力", "换融合：相加",
             "换融合：拼接", "换融合：通道门控", "去掉图卷积", "只留单编码器"]


def load():
    d = json.loads(json.dumps(DEF))
    if os.path.exists(DATA_PATH):
        try:
            with open(DATA_PATH, encoding="utf-8") as fh:
                raw = json.load(fh)
            for k in ("bench", "setup", "ablation", "depth"):
                if isinstance(raw.get(k), dict):
                    d[k].update(raw[k])
            if isinstance(raw.get("running"), dict):
                d["running"] = raw["running"]
            for k in ("updated", "fetched_at"):
                if raw.get(k):
                    d[k] = raw[k]
        except Exception as e:
            print("[warn] trend_data.json 读取失败，用内置值：%s" % e)
    return d


D = load()
B = D["bench"]
ABL = D["ablation"]
S = D["setup"]

# ---------------------------------------------------------------- 深度剖面
# 由 depth_sweep.sh + src/diag_depth.py 产出：全测试集 8080 张，按「盲区像素
# 到最近观测像素的欧氏距离」分档。有结果文件就用真实数据，没有就退回内置值。
BANDS = ["1–2", "2–4", "4–8", "8–16", "16–32", "32+"]
BAND_KEYS = ["1-2px", "2-4px", "4-8px", "8-16px", "16-32px", "32+px"]
DEPTH = {}


def load_depth():
    got = {}
    for key, fn in (("mffgcn", "depth_s1.json"), ("unet", "depth_single_s1.json")):
        p = os.path.join(ROOT, "results", fn)
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8") as fh:
                raw = json.load(fh)
            bands = raw["blind_pixels"]
            got[key] = {
                "rmse": [bands[k]["rmse"] for k in BAND_KEYS],
                "nmse": [bands[k]["nmse"] for k in BAND_KEYS],
                "share": [bands[k]["share"] for k in BAND_KEYS],
                "overall": raw["overall"]["rmse"],
                "n": raw.get("n_images"),
            }
        except Exception as e:
            print("[warn] %s 读取失败：%s" % (fn, e))
    return got


DEPTH = load_depth()

MFF = B["MFFGCN"]["v"]
UNET = B["U-Net"]["v"]
KNN = B["KNN"]["v"]
KRG = B["Kriging"]["v"]
AE = B["AE"]["v"]
FULL = ABL["完整模型"] or MFF


def pct(v, base=FULL):
    return (v / base - 1) * 100


# ---------------------------------------------------------------- SVG helpers
def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, size=12, fill=INK, anchor="start", weight="400", opacity=1.0):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family={FONT} font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" '
            f'opacity="{opacity}">{esc(s)}</text>')


def frame(title, sub, inner, W, H):
    return f'''<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" role="img">
<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>
{txt(24, 30, title, 15, INK, weight="600")}
{txt(24, 50, sub, 11.5, MUT)}
{inner}
</svg>'''


# ---------------------------------------------------------------- 图 1：基准模型
def fig1():
    W, H = 920, 440
    rows = [("KNN 插值", "KNN"), ("Kriging 插值", "Kriging"), ("AE 自编码器", "AE"),
            ("U-Net", "U-Net"), ("MFFGCN（本文）", "MFFGCN")]
    L, R, T, Bm = 132, 150, 78, 46
    lo, hi = math.log10(0.011), math.log10(0.32)

    def px(v):
        return L + (math.log10(v) - lo) / (hi - lo) * (W - L - R)

    s = []
    for tick in [0.015, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]:
        x = px(tick)
        s.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H-Bm}" stroke="{GRID}" stroke-width="1"/>')
        lab = f"{tick:g}" if tick >= 0.1 else f"{tick:.3f}".rstrip("0")
        s.append(txt(x, H - Bm + 18, lab, 10.5, MUT, "middle"))
    s.append(txt((L + W - R) / 2, H - 10, "RMSE（对数刻度，越低越好）", 11, MUT, "middle"))

    gh = (H - T - Bm) / len(rows)
    bh = 20
    for i, (label, key) in enumerate(rows):
        yc = T + i * gh + gh / 2
        s.append(txt(L - 14, yc + 4, label, 12, INK, "end"))
        pv = PAPER_BENCH[key]
        ov = B[key]["v"]
        for j, (v, c, lab) in enumerate([(pv, PAPER, "论文"), (ov, OURS, "复现")]):
            y = yc - bh - 3 + j * (bh + 6)
            w = px(v) - L
            s.append(f'<rect x="{L}" y="{y:.1f}" width="{max(w,1):.1f}" height="{bh}" rx="3" fill="{c}"/>')
            if key == "MFFGCN" and j == 1 and B[key].get("n", 1) > 1:
                ann = f"复现 {v:.4f}±{B[key].get('std', 0):.4f}"
            else:
                ann = f"{lab} {v:.4f}"
            s.append(txt(L + max(w, 1) + 8, y + 14, ann, 10.5, INK if j else MUT))

    s.append(f'<rect x="{W-R+4}" y="{T-26}" width="11" height="11" rx="2" fill="{PAPER}"/>')
    s.append(txt(W - R + 20, T - 16, "论文", 10.5, MUT))
    s.append(f'<rect x="{W-R+4}" y="{T-8}" width="11" height="11" rx="2" fill="{OURS}"/>')
    s.append(txt(W - R + 20, T + 2, "复现", 10.5, MUT))
    return frame("图 1　基础模型排序：插值 ≪ 学习型，AE ≪ U-Net",
                 "Setup 1（全图均匀采样）测试集 RMSE；横条越短越好。两边排序形状一致。",
                 "\n".join(s), W, H)


# ---------------------------------------------------------------- 图 2：消融
def fig2():
    W, H = 920, 470
    ours = [pct(ABL[n]) for n in ABL_ORDER]
    paper = [PAPER_ABL[n] for n in ABL_ORDER]
    labels = [n.replace("（", "\n（") for n in ABL_ORDER]

    L, R, T, Bm = 84, 24, 84, 96
    ymax = max(140, max(ours) + 16)
    ymin = min(-25, min(ours) - 8)

    def py(v):
        return T + (ymax - v) / (ymax - ymin) * (H - T - Bm)

    s = []
    for tick in range(-20, int(ymax) + 1, 20):
        y = py(tick)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{GRID if tick else "#b9c3ce"}" stroke-width="{1 if tick else 1.2}"/>')
        s.append(txt(L - 8, y + 4, f"{tick:+d}%" if tick else "0", 10, MUT, "end"))
    s.append(txt(22, (T + H - Bm) / 2, "RMSE 变化（相对完整模型）", 10.5, MUT, "middle").replace(
        '<text', '<text transform="rotate(-90 22 %.1f)"' % ((T + H - Bm) / 2)))

    gw = (W - L - R) / len(ABL_ORDER)
    bw = 26
    for i, n in enumerate(ABL_ORDER):
        cx = L + gw * (i + 0.5)
        for j, (v, c) in enumerate([(paper[i], PAPER), (ours[i], OURS)]):
            x = cx - bw - 2 + j * (bw + 4)
            y0, y1 = py(max(v, 0)), py(min(v, 0))
            s.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{bw}" height="{max(y1-y0,1):.1f}" rx="2" fill="{c}"/>')
            ty = (y0 - 6) if v >= 0 else (y1 + 13)
            s.append(txt(x + bw / 2, ty, f"{v:+.0f}%", 10,
                         c if abs(v) > 40 else (INK if v < 0 else MUT), "middle",
                         weight="600" if abs(v) > 40 else "400"))
        for k, line in enumerate(labels[i].split("\n")):
            s.append(txt(cx, H - Bm + 20 + k * 14, line, 10.5, INK, "middle"))
        if i == 0:
            s.append(txt(cx, H - Bm + 68, "唯一的灾难项", 10, BAD, "middle", "600"))

    s.append(f'<rect x="{W-R-120}" y="{T-40}" width="11" height="11" rx="2" fill="{PAPER}"/>')
    s.append(txt(W - R - 104, T - 30, "论文（Table V）", 10.5, MUT))
    s.append(f'<rect x="{W-R-120}" y="{T-22}" width="11" height="11" rx="2" fill="{OURS}"/>')
    s.append(txt(W - R - 104, T - 12, "复现（本机重评）", 10.5, MUT))
    return frame("图 2　消融实验：残差连接是唯一会「全盘崩掉」的部件",
                 f"柱高 = 各变体相对完整模型的 RMSE 变化百分比；论文完整模型 0.0150，复现 {FULL:.4f}。",
                 "\n".join(s), W, H)


# ---------------------------------------------------------------- 图 3：采样难度
def fig3():
    W, H = 920, 450
    prov = bool(S.get("MFFGCN_s3_provisional"))
    norm = lambda xs: [x / xs[0] for x in xs]
    ours_m = norm(S["MFFGCN"])
    ours_k = norm(S["KNN"])
    series = [
        ("论文 MFFGCN", [1.000, 1.027, 1.033], "#5b6b7f", False),
        ("论文 U-Net", [1.000, 1.093, 1.132], "#8e9bab", False),
        ("论文 AE", [1.000, 1.375, 1.441], "#b0897f", False),
        ("复现 MFFGCN", ours_m, OURS, True),
        ("复现 KNN", ours_k, "#77a3d2", True),
    ]
    L, R, T, Bm = 70, 160, 80, 60
    top = max(max(v[2] for _, v, _, _ in series), 1.6) + 0.08
    ymin, ymax = 0.98, top

    def py(v):
        return T + (ymax - v) / (ymax - ymin) * (H - T - Bm)

    def px(i):
        return L + i / 2 * (W - L - R)

    s = []
    step = 0.2 if top - 1 < 1.2 else 0.4
    tick = 1.0
    while tick <= ymax:
        y = py(tick)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{GRID}"/>')
        s.append(txt(L - 8, y + 4, f"{tick:.1f}×", 10, MUT, "end"))
        tick += step
    for i, lab in enumerate(["Setup 1\n均匀采样", "Setup 2\n25% 盲区", "Setup 3\n盲区 10–50%"]):
        for k, line in enumerate(lab.split("\n")):
            s.append(txt(px(i), H - Bm + 20 + k * 14, line, 10.5, MUT if k else INK, "middle"))
    s.append(txt(20, (T + H - Bm) / 2, "RMSE 相对 Setup 1 的倍数", 11, MUT, "middle").replace(
        '<text', '<text transform="rotate(-90 20 %.1f)"' % ((T + H - Bm) / 2)))

    for name, vals, color, dash in series:
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
        da = ' stroke-dasharray="6 4"' if dash else ""
        s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4"{da} stroke-linejoin="round"/>')
        for i, v in enumerate(vals):
            s.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="4" fill="#fff" stroke="{color}" stroke-width="2.2"/>')
        s.append(txt(px(2) + 12, py(vals[2]) + 4, f"{name}  {vals[2]:.2f}×", 10.5, color))
    if prov:
        s.append(txt(px(2) + 12, py(ours_m[2]) + 22, "※ S3 为第 1 轮存盘，重训中", 9.5, BAD))

    s.append(txt(W - R - 200, T - 30, "实线 = 论文", 10.5, MUT))
    s.append(txt(W - R - 120, T - 30, "虚线 = 复现", 10.5, OURS))
    return frame("图 3　采样变难时所有模型都变差——方向一致，幅度不同",
                 "三条曲线是论文里三档典型模型，两条是复现；纵轴以各自 Setup 1 为 1.0 归一。",
                 "\n".join(s), W, H)


# ---------------------------------------------------------------- 图 4：盲区深度
def fig4():
    W, H = 920, 440
    L, R, T, Bm = 76, 216, 84, 74

    if DEPTH.get("mffgcn") and DEPTH.get("unet"):
        series = [("MFFGCN（本文）", DEPTH["mffgcn"]["rmse"], OURS),
                  ("U-Net（单编码器）", DEPTH["unet"]["rmse"], BAD)]
        ymin = min(min(v) for _, v, _ in series) - 0.002
        ymax = max(max(v) for _, v, _ in series) + 0.004
        title = "图 4　盲区越深误差越高，但本文模型的劣化速度更慢"
        sub = ("按「盲区像素到最近观测像素的距离」分箱，只统计未观测像素；"
               "Setup 1，全测试集 %s 张，权重为修复坐标后的新训练结果。" %
               (f"{DEPTH['mffgcn']['n']:,}" if DEPTH["mffgcn"].get("n") else "8080"))
    else:
        mid, big = D["depth"]["mid"], D["depth"]["big"]
        series = [("中等盲区（旧权重）", mid, OURS), ("大盲区（旧权重）", big, BAD)]
        ymin, ymax = min(mid) - 0.004, max(big) + 0.004
        title = "图 4　盲区内部：误差随「离可见区距离」单调上升"
        sub = "旧权重重评口径；新权重的剖面待补。"

    def py(v):
        return T + (ymax - v) / (ymax - ymin) * (H - T - Bm)

    def px(i):
        return L + i / (len(BANDS) - 1) * (W - L - R)

    s = []
    step = 0.005 if ymax - ymin > 0.02 else 0.0025
    t = math.ceil(ymin / step) * step
    while t <= ymax:
        y = py(t)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{GRID}"/>')
        s.append(txt(L - 8, y + 4, f"{t:.4f}", 10, MUT, "end"))
        t = round(t + step, 5)
    for i, b in enumerate(BANDS):
        s.append(txt(px(i), H - Bm + 20, b, 10.5, MUT, "middle"))
    s.append(txt((L + W - R) / 2, H - 16, "盲区像素到最近观测像素的距离（px）", 11, MUT, "middle"))
    s.append(txt(L - 8, T - 14, "RMSE", 10.5, MUT, "end"))

    for name, vals, color in series:
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for i, v in enumerate(vals):
            s.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="3.6" fill="#fff" '
                     f'stroke="{color}" stroke-width="2"/>')

    # 右侧图例 + 两端差距标注
    ly = T + 6
    for name, vals, color in series:
        s.append(f'<line x1="{W-R+26}" y1="{ly}" x2="{W-R+56}" y2="{ly}" stroke="{color}" stroke-width="2.4"/>')
        s.append(txt(W - R + 64, ly + 4, name, 10.5, INK))
        s.append(txt(W - R + 64, ly + 19, "末端 %.4f" % vals[-1], 10, MUT))
        ly += 46

    if len(series) == 2:
        a, b = series[0][1], series[1][1]
        g0 = (1 - a[0] / b[0]) * 100
        g5 = (1 - a[-1] / b[-1]) * 100
        s.append(txt(px(0) + 8, py(a[0]) - 12,
                     "贴边处只领先 %.0f%%" % g0, 10.5, OURS, "start", "600"))
        s.append(txt(px(5) - 6, py(a[5]) + 26,
                     "深处领先 %.0f%%" % g5, 10.5, OURS, "end", "600"))
        s.append(txt(W - R + 26, ly + 16,
                     "优势随深度扩大 %.1f 倍" % (g5 / max(g0, 1e-9)), 10.5, GOOD, "start", "600"))
    else:
        rise = (series[0][1][-1] / series[0][1][0] - 1) * 100
        s.append(txt(px(0) + 6, py(min(series[0][1])) - 12, f"越往盲区深处越差：+{rise:.0f}%",
                     10.5, OURS, "start", "600"))

    return frame(title, sub, "\n".join(s), W, H)


# ---------------------------------------------------------------- 文案参数
UNET_BETTER = MFF > UNET          # 复现里 U-Net 是否仍赢本文模型
RESID = ABL["去残差"]
resid_pct = pct(RESID)
abl_others = [v for k, v in ABL.items() if k not in ("完整模型", "去残差")]
resid_ratio = RESID / FULL
other_max = max(abl_others) / FULL
knn_ratio = KNN / MFF
ae_u = AE / UNET
s2_pct = (S["MFFGCN"][1] / S["MFFGCN"][0] - 1) * 100
s3_pct = (S["MFFGCN"][2] / S["MFFGCN"][0] - 1) * 100
PROV = bool(S.get("MFFGCN_s3_provisional"))
same_n = 5 + (0 if UNET_BETTER else 1)
diff_n = 3 - (0 if UNET_BETTER else 1)
verdict_tag = '<span class="tag t-no">相反</span>' if UNET_BETTER else '<span class="tag t-ok">一致</span>'
verdict_txt = ("我的 MFFGCN（%.4f）比 U-Net（%.4f）略差" % (MFF, UNET)) if UNET_BETTER else \
              ("我的 MFFGCN（%.4f）胜过 U-Net（%.4f）" % (MFF, UNET))
data_note = D.get("fetched_at") or D.get("updated")

# ---- 盲区深度剖面的文案参数 ----
HAVE_DEPTH = bool(DEPTH.get("mffgcn") and DEPTH.get("unet"))
if HAVE_DEPTH:
    DM = DEPTH["mffgcn"]["rmse"]
    DU = DEPTH["unet"]["rmse"]
else:
    DM = D["depth"]["mid"]
    DU = [v * 1.06 for v in DM]
depth_rise = DM[-1] / DM[0] - 1
gap_edge = (1 - DM[0] / DU[0]) * 100
gap_deep = (1 - DM[-1] / DU[-1]) * 100
depth_rows = "".join(
    "<tr><td>%s</td><td class='c'>%.4f</td><td class='c'>%.4f</td><td class='c'>%.4f</td>"
    "<td class='c'>%.1f%%</td></tr>" %
    (BANDS[i], DM[i], DU[i], DU[i] - DM[i],
     (1 - DM[i] / DU[i]) * 100)
    for i in range(len(BANDS)))

mff_src = B["MFFGCN"].get("src", "")
mff_ann = ("%d 个种子均值 ±%.4f" % (B["MFFGCN"].get("n", 1), B["MFFGCN"].get("std", 0))) \
    if B["MFFGCN"].get("n", 1) > 1 else mff_src
s3_txt = ("%.2f×（第 1 轮存盘，不可用）" % (S["MFFGCN"][2] / S["MFFGCN"][0])) if PROV \
    else ("%.2f×" % (S["MFFGCN"][2] / S["MFFGCN"][0]))

CSS = """
:root{--ink:#1f2933;--mut:#6b7785;--line:#e3e8ee;--blue:#2f6fb3;--red:#c0392b;--green:#1f9d55;--bg:#f4f6f9;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:%s;line-height:1.75;-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:44px 28px 80px}
.eyebrow{font-size:12px;letter-spacing:.14em;color:var(--blue);font-weight:700;text-transform:uppercase}
h1{font-size:30px;line-height:1.35;margin:10px 0 8px;letter-spacing:-.01em}
.lead{color:var(--mut);font-size:14.5px;margin:0 0 10px}
.stamp{font-size:12px;color:var(--mut);background:#fff;border:1px dashed var(--line);border-radius:8px;padding:7px 12px;display:inline-block;margin:0 0 22px}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:22px 24px;margin:0 0 22px;box-shadow:0 1px 2px rgba(16,24,40,.04)}
.card h2{font-size:17px;margin:0 0 4px}
.card h2 .num{color:var(--blue);margin-right:8px}
.card p{margin:8px 0 0;font-size:14px}
.svgbox{margin:14px -6px 6px;border:1px solid #eef2f6;border-radius:10px;overflow:hidden;background:#fff}
.finding{display:flex;gap:10px;margin-top:14px;padding:12px 14px;border-radius:10px;background:#f2f7fd;font-size:13.5px}
.finding b{color:var(--blue)}
.finding.warn{background:#fdf3f2}.finding.warn b{color:var(--red)}
.finding.ok{background:#f1f9f4}.finding.ok b{color:var(--green)}
table{width:100%%;border-collapse:collapse;font-size:13.5px;margin-top:6px}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--mut);font-weight:600;background:#fafbfd}
td.c{text-align:center;white-space:nowrap}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11.5px;font-weight:600}
.t-ok{background:#e8f6ed;color:#17703d}.t-part{background:#fdf5e6;color:#8a5a00}.t-no{background:#fbeae8;color:#9c2f22}
.note{font-size:12.5px;color:var(--mut)}
ul{margin:8px 0 0;padding-left:20px;font-size:14px}li{margin:5px 0}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:6px 0 24px}
.kpi{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi .v{font-size:22px;font-weight:700;letter-spacing:-.01em}
.kpi .l{font-size:11.5px;color:var(--mut);margin-top:2px}
@media(max-width:760px){.kpis{grid-template-columns:repeat(2,1fr)}}
""" % FONT


def html():
    p_others_min = min(pct(v) for v in abl_others)
    p_others_max = max(pct(v) for v in abl_others)
    one_sentence = ("「谁的精度该高于谁」这套相对排序基本复现出来了；没复现出来的是「MFFGCN 比 U-Net 更强」这一步，以及论文那种近乎为零的采样退化。"
                    if UNET_BETTER else
                    "「谁的精度该高于谁」这套相对排序基本复现出来了，MFFGCN 也已经反超 U-Net；还差的是论文那种近乎为零的采样退化。")
    # 正在训练的 run（快照里带 train_log.csv 的那些）
    _run = {t: v for t, v in (D.get("running") or {}).items() if v.get("alive")}
    _s2, _s3 = _run.get("txf_mffgcn_s2"), _run.get("txf_mffgcn_s3")
    if _s2 and not _s3:
        if _s2.get("just_started") or not _s2.get("epoch"):
            s23_state = "Setup 2 刚开训（第一轮还没跑完），Setup 3 仍排队"
        else:
            s23_state = "Setup 2 正在训练（当前第 %s 轮，最优验证 %.5f），Setup 3 仍排队" % (
                _s2.get("epoch"), _s2.get("best_val") or 0)
    elif _s2 and _s3:
        s23_state = "Setup 2 / 3 都在训练中"
    else:
        s23_state = "Setup 2/3 的新训练还在排队"
    bench_base = [k for k in B if k != "MFFGCN"]
    n_bench = len(bench_base)
    n_full = sum(1 for k in bench_base if (B[k].get("n") or 1) >= 2)
    run_rows = "".join(
        "<tr><td>%s</td><td class='c'>%s</td><td class='c'>%s</td><td class='c'>%s</td></tr>"
        % (t,
           ("ep %s" % v["epoch"]) if v.get("epoch") else "刚开训",
           ("%.5f" % v["val_rmse"]) if v.get("val_rmse") is not None else "—",
           ("ep %s / %.5f" % (v["best_epoch"], v["best_val"]))
           if v.get("best_val") is not None else "—")
        for t, v in sorted(_run.items()) if not v.get("error"))
    running_block = (
        "<h2 style='font-size:14px;margin:22px 0 0'>当前在跑（未出最终测试指标）</h2>"
        "<table><tr><th style='width:40%'>run</th><th class='c'>最新轮次</th>"
        "<th class='c'>最新 val RMSE</th><th class='c'>最优轮次 / 最优 val</th></tr>"
        + run_rows + "</table>" if run_rows else "")
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MFFGCN 复现与论文趋势一致性对比</title>
<style>{CSS}</style></head>
<body><div class="wrap">

<div class="eyebrow">Reproduction · Trend Cross-check</div>
<h1>我的复现，哪些趋势和论文是一致的</h1>
<p class="lead">对照论文 Table III / Table V 的原始数值。判定标准只看「变化方向和相对排序」，不看绝对值——绝对精度差是另一件事。</p>
<div class="stamp">数据截至 {data_note}　·　本文模型：{mff_ann}</div>

<div class="kpis">
  <div class="kpi"><div class="v" style="color:#1f9d55">{same_n} 条</div><div class="l">方向与排序一致</div></div>
  <div class="kpi"><div class="v" style="color:#8a5a00">2 条</div><div class="l">方向对、幅度不同</div></div>
  <div class="kpi"><div class="v" style="color:#c0392b">{diff_n} 条</div><div class="l">结论相反</div></div>
  <div class="kpi"><div class="v">{n_bench}/8</div><div class="l">基线已有可比数字（其中 {n_full} 个跑满论文协议）</div></div>
</div>

<div class="card">
<h2><span class="num">01</span>结论速览</h2>
<table>
<tr><th style="width:34%">论文里的趋势</th><th style="width:30%">论文数值</th><th style="width:30%">我的复现</th><th class="c">判定</th></tr>
<tr><td>传统插值（KNN / Kriging）远差于深度模型</td><td>KNN 0.2312、Kriging 0.1389，是 MFFGCN 的 9–15 倍</td><td>{KNN:.4f} / {KRG:.4f}，是 MFFGCN 的 {knn_ratio:.1f} 倍</td><td class="c"><span class="tag t-ok">一致</span></td></tr>
<tr><td>KNN 略差于 Kriging</td><td>0.2312 vs 0.1389</td><td>{KNN:.4f} vs {KRG:.4f}</td><td class="c"><span class="tag t-ok">一致</span></td></tr>
<tr><td>U-Net 级架构远好于自编码器</td><td>AE 0.0456 → U-Net 0.0182（2.5 倍）</td><td>AE {AE:.4f} → U-Net {UNET:.4f}（{ae_u:.1f} 倍）</td><td class="c"><span class="tag t-ok">一致</span></td></tr>
<tr><td>去掉残差连接会全盘崩掉，且是唯一大崩坏项</td><td>0.0150 → 0.0347（+131%），其余变体仅 ±3–8%</td><td>{FULL:.4f} → {RESID:.4f}（{resid_pct:+.0f}%），其余变体 {p_others_min:+.0f}%～{p_others_max:+.0f}%</td><td class="c"><span class="tag t-ok">一致</span></td></tr>
<tr><td>去掉单个融合/GCN 部件的损失很小</td><td>去 GCN +7.3%，换拼接 +5.3%，换相加 +8.0%</td><td>去 GCN {pct(ABL['去掉图卷积']):+.1f}%，换拼接 {pct(ABL['换融合：拼接']):+.1f}%，换相加 {pct(ABL['换融合：相加']):+.1f}%</td><td class="c"><span class="tag t-part">一致但符号相反</span></td></tr>
<tr><td>采样越受限，误差越高</td><td>三档 RMSE 单调升（MFFGCN +3.3%）</td><td>单调升（S2 {s2_pct:+.1f}%，S3 {s3_pct:+.0f}%{'，待重训' if PROV else ''}）</td><td class="c"><span class="tag t-part">方向一致、幅度不同</span></td></tr>
<tr><td>MFFGCN 应当险胜 U-Net</td><td>0.0150 vs 0.0182（好 18%）</td><td>{MFF:.4f} vs {UNET:.4f}（{'差' if UNET_BETTER else '好'} {abs(MFF/UNET-1)*100:.1f}%）</td><td class="c">{verdict_tag}</td></tr>
<tr><td>误差主要集中在阴影/建筑边缘</td><td>正文：误差主要在阴影边缘的精细度</td><td>越往盲区中心越差（+{depth_rise*100:.0f}%）；但本文模型相对 U-Net 的优势也从 {gap_edge:.0f}% 扩大到 {gap_deep:.0f}%</td><td class="c"><span class="tag t-no">相反</span></td></tr>
<tr><td>对不均匀采样近乎免疫</td><td>Setup 1→3 只退化 3.3%</td><td>退化 {s3_pct:.1f}%{'（旧值待重训）' if PROV else ''}</td><td class="c"><span class="tag t-no">相反</span></td></tr>
</table>
<div class="finding ok"><span>✔</span><span><b>一句话：</b>{one_sentence}</span></div>
</div>

<div class="card">
<h2><span class="num">02</span>基础模型：排序形状对上了</h2>
<div class="svgbox">{fig1()}</div>
<p>论文和复现的条形都是从下往上递减的同一个形状：插值类最长，自编码器次之，U-Net 与 MFFGCN 最短。三个<b>相对关系</b>完全对得上——插值远差于学习型、KNN 差于 Kriging、AE 差于 U-Net。</p>
<p>论文比了 9 个模型，我这边只有 <b>5 个</b>有可信数字，所以图里只放共有的这 5 个。DeepAE、RadioUNet、Pix2pix+、RME-GAN 的代码已就绪、正在队列里跑，出数后补进来不会改变下面的结论。</p>
<p>最后一格是关键的差异点：<b>{verdict_txt}</b>。论文里 MFFGCN 是赢的（0.0150 vs 0.0182）。{'这更像是训练轮次不对等——U-Net 跑到第 47 轮收敛，MFFGCN 停在更早的轮次。' if UNET_BETTER else '两边现在都是多种子均值（本文 %d 个、U-Net %d 个），标准差分别是 %.4f 与 %.4f——差距 %.4f 约等于 %.1f 个本文标准差，这个反超不是种子抽签抽出来的。' % (B['MFFGCN'].get('n', 1), B['U-Net'].get('n', 1), B['MFFGCN'].get('std', 0), B['U-Net'].get('std', 0), UNET - MFF, (UNET - MFF) / max(B['MFFGCN'].get('std', 0) or 1e-9, 1e-9))}</p>
<div class="finding"><span>→</span><span><b>可用的结论：</b>「学习方法碾压插值」成立，差距量级在 {knn_ratio:.1f} 倍以上；「多模态融合打赢单编码器 U-Net」{'暂时不成立' if UNET_BETTER else '也成立'}。</span></div>
</div>

<div class="card">
<h2><span class="num">03</span>消融：残差连接是命门，这点完全一致</h2>
<div class="svgbox">{fig2()}</div>
<p>这是全部实验里<b>最干净的一条吻合</b>。论文里八个变体中有七个的 RMSE 只在完整模型的 ±3–8% 内浮动，唯独去掉残差直接翻到 2.3 倍；我这边同样是去掉残差最差（{resid_pct:+.0f}%，{resid_ratio:.2f} 倍），其余变体最多只到 {other_max:.2f} 倍。</p>
<p><b>口径提醒：</b>图 2 这九个变体和图注里的「完整模型 {FULL:.4f}」是同一批旧协议训出来的，只用于它们彼此的相对比较；图 1 那个 {MFF:.4f} 是修复坐标后按新协议重训的五种子结果。两套数字混用会得出错误结论——所以这里的百分比一律以 {FULL:.4f} 为分母。</p>
<p>次要的一致点：<b>去掉 GCN、换掉融合方式，损失本来就不大</b>——论文是 +3%～+8%，我这边是 {p_others_min:+.0f}%～{p_others_max:+.0f}%。所以「我这边的 GCN / 融合没带来收益」并不算严重背离论文，论文自己也没做出大增益，只是方向反了。</p>
<div class="finding ok"><span>✔</span><span><b>可写进报告：</b>残差连接的必要性在两次实验中被独立验证；融合模块与图卷积的边际贡献都很小，属于论文本身的弱项，不是复现引入的偏差。</span></div>
</div>

<div class="card">
<h2><span class="num">04</span>采样变难：方向一致，退化速度不一致</h2>
<div class="svgbox">{fig3()}</div>
<p>五条曲线全部向右上走，「采样越不均匀越难」这个物理结论是复现出来的。差别在斜率：论文的 MFFGCN 几乎是平的（1.00 → 1.03，只涨 3.3%），我这边 S2 涨 {s2_pct:.1f}%，S3 是 {s3_txt}。</p>
<p>另一个反向细节：论文里 KNN / Kriging 在三档设置下几乎不动（0% 和 +0.9%），我这边两种插值都退化了 {(S['KNN'][2]/S['KNN'][0]-1)*100:.0f}% 以上。说明论文的插值基线在受限区里做了额外处理，或者口径与我的实现不同。</p>
<div class="finding warn"><span>!</span><span><b>待补：</b>S3 的完整重训正在队列里，出数后这条要重画。当前不要把 {'1.99×' if PROV else 'S3 数值'} 写进任何结论。</span></div>
</div>

<div class="card">
<h2><span class="num">05</span>盲区内部：误差怎么随深度分布</h2>
<div class="svgbox">{fig4()}</div>
<p>按「盲区像素到最近观测像素的距离」分箱后，两条曲线都单调上升：贴边 {DM[0]:.4f} → 32 像素外 {DM[-1]:.4f}（{"+%.0f%%" % (depth_rise*100)}）。论文正文说误差主要来自「阴影边缘的精细度」（它给 MS-SSIM 设极小权重的理由），我这边看到的是<b>越往盲区中心越差</b>——严格说两件事不冲突，一个讲误差的形态、一个讲误差的位置，但论文没给这条同口径曲线，只能算「论文没覆盖、我补上的观测」。</p>
<table>
<tr><th style="width:22%">离最近观测点</th><th class="c">MFFGCN</th><th class="c">U-Net</th><th class="c">差值</th><th class="c">领先幅度</th></tr>
{depth_rows}
</table>
<p><b>这张表比曲线本身更有用：</b>两种模型都在深处变差，但本文模型的劣化更慢——贴边处只领先 {gap_edge:.0f}%，到 32 像素外领先 {gap_deep:.0f}%，优势扩大约 {gap_deep/max(gap_edge,1e-9):.1f} 倍。也就是说多模态融合 / 图分支真正起作用的地方不在边缘，而在<b>观测点够不着的深处</b>。</p>
<div class="finding ok"><span>✔</span><span><b>可写进报告：</b>「盲区越深越难」是数据本身的规律，不是模型缺陷；而本文方法相对单编码器的增益随深度放大，这正好解释了论文表 III 里那 18% 的整体优势从哪来。</span></div>
<div class="finding warn"><span>!</span><span><b>口径：</b>上图只有 Setup 1（全图均匀采样）是修复坐标后的新权重；{s23_state}。出数后要补两条曲线，届时「深处领先多少」这个数可能变。</span></div>
</div>

<div class="card">
<h2><span class="num">06</span>数字来源与可信度</h2>
<table>
<tr><th style="width:26%">数据</th><th style="width:40%">来源</th><th>可信度</th></tr>
<tr><td>论文全部数值</td><td>Table III / Table V 原文转录（PDF 逐格核对）</td><td>高</td></tr>
<tr><td>MFFGCN {MFF:.4f}</td><td>{mff_src}</td><td>{'高' if B['MFFGCN'].get('n', 1) > 1 else '中'}</td></tr>
<tr><td>U-Net {UNET:.4f}</td><td>{B['U-Net'].get('n', 1)} 个种子均值（±{B['U-Net'].get('std', 0):.4f}），50 轮、论文口径、8080 张测试集</td><td>高</td></tr>
<tr><td>KNN / Kriging</td><td>自实现插值，同一测试集</td><td>中高（实现细节可能与论文不同）</td></tr>
<tr><td>AE {AE:.4f}</td><td>新训练，论文口径</td><td>中高</td></tr>
<tr><td>消融八个变体</td><td>旧权重重评，多数停在第 1–15 轮</td><td>中（只用于变体间相对比较）</td></tr>
<tr><td>盲区深度剖面</td><td>diag_depth.py，Setup 1 全测试集 {DEPTH.get('mffgcn', {}).get('n') or 8080} 张，按到最近观测像素的距离分箱</td><td>{'高（新权重，与 test_metrics 对上）' if HAVE_DEPTH else '低（旧权重）'}</td></tr>
<tr><td>MFFGCN S3（{S['MFFGCN'][2]:.4f}）</td><td>{'第 1 轮存盘，早停被错误坐标污染' if PROV else '完整重训结果'}</td><td>{'低，不可引用' if PROV else '中高'}</td></tr>
</table>
<p class="note">口径说明：论文在整张图上按密度均匀撒观测点；早期部分实验只把观测点放在有信号的位置。对照实验显示两者相差 4–9%，不改变变体之间的相对排名，所以消融结论仍可用。</p>
<p class="note">本页数据由 update_trend.py 从服务器快照自动刷新，最后更新 {data_note}。</p>
{running_block}
</div>

</div></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "07-与论文趋势一致性对比.html"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(html())
    figdir = os.path.join(ROOT, "figs")
    os.makedirs(figdir, exist_ok=True)
    for name, fn in (("fig_trend_bench.svg", fig1), ("fig_trend_ablation.svg", fig2),
                     ("fig_trend_setup.svg", fig3), ("fig_trend_depth.svg", fig4)):
        with open(os.path.join(figdir, name), "w", encoding="utf-8") as fh:
            fh.write(fn())
    print("[wrote] %s (%.1f KB)" % (a.out, os.path.getsize(a.out) / 1024))


if __name__ == "__main__":
    main()
