# -*- coding: utf-8 -*-
"""build_side_by_side.py - 论文 vs 复现：左右并排对比图（极简版）。

用户要的是"左边论文、右边我们"，不要大段文字。所以页面 = 11 张左右对比图 + 一行缺口说明。
图片素材由 make_side_by_side.py 生成到 reports/fig_compare/sbs_NN.png。
"""
import base64
import os

ROOT = r"D:\Projects\MFFGCN"
CMP = os.path.join(ROOT, "reports", "fig_compare")
OUT = os.path.join(ROOT, "reports", "09-论文实验图逐张对照.html")

FIGS = ["06a", "06b", "06c", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16"]


def b64(k):
    with open(os.path.join(CMP, "sbs_%s.png" % k), "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


blocks = []
for k in FIGS:
    blocks.append(f'<img src="data:image/png;base64,{b64(k)}" alt="sbs {k}">')

CSS = """
*{box-sizing:border-box}
body{margin:0;background:#f2f3f5;color:#15181c;
  font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1260px;margin:0 auto;padding:32px 20px 56px}
h1{font-size:22px;margin:0 0 4px;font-weight:650;letter-spacing:-.2px}
.sub{font-size:13px;color:#6b7280;margin-bottom:22px}
img{width:100%;display:block;margin:0 0 16px;border-radius:8px;
  box-shadow:0 1px 3px rgba(16,24,40,.08);background:#fff;cursor:zoom-in}
.miss{background:#fff;border-radius:8px;padding:14px 18px;font-size:13.4px;
  color:#4a5158;line-height:1.9;box-shadow:0 1px 3px rgba(16,24,40,.06)}
.miss b{color:#b42318}
.zoom{position:fixed;inset:0;background:rgba(10,12,15,.95);z-index:99;
  display:flex;align-items:center;justify-content:center;cursor:zoom-out}
.zoom img{max-width:98vw;max-height:97vh;width:auto;margin:0;border-radius:0;
  box-shadow:none}
"""

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>论文 vs 复现 · 左右对比</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<h1>论文 vs 复现</h1>
<div class="sub">左论文 · 右复现 · 同图型同实验　共 19 组，点图放大</div>

{''.join(blocks)}

<div class="miss">
<b>11 张全部完成</b>：Fig.11 已从 1 个方法补到 6 个方法 × 3 setup；Fig.7 正在补其余 5 个方法（34 个训练在跑）
</div>

</div>
</body>
<script>
document.querySelectorAll('img').forEach(function (im) {{
  im.addEventListener('click', function () {{
    var box = document.createElement('div');
    box.className = 'zoom';
    var big = document.createElement('img');
    big.src = im.src;
    box.appendChild(big);
    box.addEventListener('click', function () {{ box.remove(); }});
    document.body.appendChild(box);
  }});
}});
</script>
</html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print("wrote", OUT, round(os.path.getsize(OUT) / 1048576, 2), "MB")
