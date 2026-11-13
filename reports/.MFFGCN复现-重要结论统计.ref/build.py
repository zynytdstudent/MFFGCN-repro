# -*- coding: utf-8 -*-
"""生成《MFFGCN 复现：重要结论统计》xlsx。"""
import math
import os

try:
    import openpyxl
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "openpyxl>=3.1.0"])
    import openpyxl

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.formatting.rule import CellIsRule
from openpyxl.worksheet.datavalidation import DataValidation


def xl_color(css_hex: str) -> str:
    value = css_hex.removeprefix("#").upper()
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB, got: {css_hex}")
    return "FF" + value


XL_HEADER_BG = xl_color("#4472C4")
XL_TITLE = xl_color("#2F5597")
XL_SUB = xl_color("#7F7F7F")
XL_BORDER = xl_color("#BFBFBF")
XL_ZEBRA = xl_color("#F2F6FC")
XL_WHITE = xl_color("#FFFFFF")
XL_GREEN_BG, XL_GREEN_FG = xl_color("#C6EFCE"), xl_color("#006100")
XL_YELLOW_BG, XL_YELLOW_FG = xl_color("#FFEB9C"), xl_color("#9C6500")
XL_RED_BG, XL_RED_FG = xl_color("#FFC7CE"), xl_color("#9C0006")

OUT = r"D:\Projects\MFFGCN\reports\MFFGCN复现-重要结论统计.xlsx"

# (序号, 结论, 关键数据, 判定, 要点)
ROWS = [
    (1, "领先全部基线，五个种子全胜",
     "0.00454，对比 U-Net 0.00583（−22.1%）、RadioUNet 0.00594（−22.9%）；最差一次 0.00556 仍优于 U-Net 最好一次 0.00583",
     "成立", "种子区间不重叠，不是运气；但方差大 8 倍，部署建议多跑几种子取优"),
    (2, "绝对精度接近论文",
     "NMSE 0.00478（五种子均值）vs 论文 0.0043，差 11%；最好一次 0.00454，差 5.6%",
     "部分", "论文的 NMSE 归一化定义未核实，不能当作「复现成功」的依据"),
    (3, "深度学习远优于传统插值",
     "KNN 0.1200、Kriging 0.1175，与 MFFGCN 相差 26 倍",
     "成立", "只站得住「一个数量级」；我们的插值实现强 8 倍，论文的具体倍数不可引用"),
    (4, "残差精修把退化压缩 49%（核心发现）",
     "s1→s3 粗估计涨 34.3%（0.00594→0.00798），精修后仅涨 17.6%（0.00454→0.00534）；各级降幅 23.6% / 27.3% / 33.0%",
     "成立", "盲区越大补得越多，这就是论文「退化更慢」一直缺的机制解释"),
    (5, "精修能补稀疏，补不了观测错误",
     "同一权重：clean −23.6%、20 dB −14.2%、10 dB 仅 −2.6%",
     "成立", "GCN 建图依赖观测点，观测值被污染后信息源本身失效，这是能力边界"),
    (6, "残差最关键；融合方式几乎不影响；双编码器增益实为 GCN 增益",
     "sigmoid +4.7%、spatial +7.2%；粗估计 0.00594 ≈ 单编码器 0.00583；论文 w/o Residual 为 0.0043→0.0231（×5.4）",
     "成立", "w/o Residual 仍缺干净数据，只能靠精修的 23–33% 贡献间接反推"),
    (7, "算力省 14.5%，但参数与时延都上涨",
     "FLOPs 40.32 G（−14.5%）、参数 18.23 M（+37%）、时延 5.17 ms（三者最慢，+1.4%）",
     "成立", "论文「时延基本持平」应改写为「同量级、略慢 1.4%」，不能算优势项"),
    (8, "盲区越大优势越大（论文主卖点）",
     "论文 Setup 2/3 自身 RMSE 仅 +2.67% / +3.33%；本复现 NMSE 由 0.00454 涨到 0.00534（+17.6%）",
     "未验证", "缺 s2/s3 基线对照，算不出相对领先；口径不同（RMSE vs NMSE），趋势差距明显"),
]

TITLE_ROW, SUB_ROW, HEAD_ROW = 1, 2, 4
FIRST_DATA = HEAD_ROW + 1
LAST_DATA = FIRST_DATA + len(ROWS) - 1
NOTE_ROW = LAST_DATA + 2

HEADERS = ["序号", "结论", "关键数据", "判定", "要点"]
WIDTHS = {"A": 6, "B": 30, "C": 54, "D": 10, "E": 40}
CHARS_PER_LINE = {"B": 14, "C": 26, "E": 19}   # 中文按 2 个宽度单位估算

wb = Workbook()
ws = wb.active
ws.title = "结论统计"

thin = Side(style="thin", color=XL_BORDER)
box = Border(left=thin, right=thin, top=thin, bottom=thin)

# ---------- 标题区 ----------
ws.merge_cells(f"A{TITLE_ROW}:E{TITLE_ROW}")
c = ws[f"A{TITLE_ROW}"]
c.value = "MFFGCN 复现：重要结论统计"
c.font = Font(name="微软雅黑", size=16, bold=True, color=XL_TITLE)
c.alignment = Alignment(horizontal="left", vertical="center")
ws.row_dimensions[TITLE_ROW].height = 30

ws.merge_cells(f"A{SUB_ROW}:E{SUB_ROW}")
c = ws[f"A{SUB_ROW}"]
c.value = ("数据截至 2026-11-06 ｜ 共 8 条 ｜ 判定口径：成立＝数据支持；部分＝方向对但未核实；"
          "未验证＝缺对照数据；需改写＝与论文说法冲突")
c.font = Font(name="微软雅黑", size=10, color=XL_SUB)
c.alignment = Alignment(horizontal="left", vertical="center")

# ---------- 表头 ----------
for col, name in zip("ABCDE", HEADERS):
    cell = ws[f"{col}{HEAD_ROW}"]
    cell.value = name
    cell.font = Font(name="微软雅黑", size=11, bold=True, color=XL_WHITE)
    cell.fill = PatternFill("solid", fgColor=XL_HEADER_BG)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = box
ws.row_dimensions[HEAD_ROW].height = 26

# ---------- 数据区 ----------
for i, row in enumerate(ROWS):
    r = FIRST_DATA + i
    for col, val in zip("ABCDE", row):
        cell = ws[f"{col}{r}"]
        cell.value = val
        cell.border = box
        cell.font = Font(name="微软雅黑", size=10)
        if col == "A":
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.number_format = "0"
        elif col == "D":
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.font = Font(name="微软雅黑", size=10, bold=True)
        else:
            cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
        if i % 2 == 1 and col != "D":
            cell.fill = PatternFill("solid", fgColor=XL_ZEBRA)

    lines = max(
        math.ceil(len(str(row[k])) / CHARS_PER_LINE[col]) if col in CHARS_PER_LINE else 1
        for k, col in [(1, "B"), (2, "C"), (4, "E")]
    )
    ws.row_dimensions[r].height = max(30, lines * 14 + 8)

for col, w in WIDTHS.items():
    ws.column_dimensions[col].width = w

# ---------- 判定列：下拉 + 条件格式 ----------
dv = DataValidation(type="list", formula1='"成立,部分,未验证,需改写"', allow_blank=True,
                    showDropDown=False)
dv.error = "请从下拉中选择判定"
dv.errorTitle = "判定取值"
ws.add_data_validation(dv)
dv.add(f"D{FIRST_DATA}:D{LAST_DATA}")

rng = f"D{FIRST_DATA}:D{LAST_DATA}"
ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"成立"'],
                                             fill=PatternFill("solid", fgColor=XL_GREEN_BG),
                                             font=Font(color=XL_GREEN_FG, bold=True)))
ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"部分"'],
                                             fill=PatternFill("solid", fgColor=XL_YELLOW_BG),
                                             font=Font(color=XL_YELLOW_FG, bold=True)))
ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"未验证"'],
                                             fill=PatternFill("solid", fgColor=XL_RED_BG),
                                             font=Font(color=XL_RED_FG, bold=True)))
ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"需改写"'],
                                             fill=PatternFill("solid", fgColor=XL_RED_BG),
                                             font=Font(color=XL_RED_FG, bold=True)))

# ---------- 脚注 ----------
ws.merge_cells(f"A{NOTE_ROW}:E{NOTE_ROW}")
c = ws[f"A{NOTE_ROW}"]
c.value = ("数据来源：runs/txf_*/test_metrics.json（主表、消融）、eval_ROB/metrics_rob_*.json（噪声、预算）、"
           "src/interp_*.json（插值）、五种子 runs/seed*｜ 完整分析见 reports/08-实验图表汇总与分析.html")
c.font = Font(name="微软雅黑", size=9, color=XL_SUB)
c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
ws.row_dimensions[NOTE_ROW].height = 30

# ---------- 页面设置 ----------
ws.page_setup.orientation = "landscape"
ws.page_setup.fitToWidth = 1
ws.page_setup.fitToHeight = 0
ws.sheet_properties.pageSetUpPr.fitToPage = True
ws.print_options.horizontalCentered = True

wb.properties.title = "MFFGCN 复现：重要结论统计"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
wb.save(OUT)
print("saved", OUT)
