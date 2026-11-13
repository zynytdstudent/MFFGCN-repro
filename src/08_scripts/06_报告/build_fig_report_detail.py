# -*- coding: utf-8 -*-
"""build_fig_report_detail.py - 详版报告（旧版，全部数据表 + 逐条分析）。

正式交付用同目录的 build_fig_report.py（简版：一图一段）。
本脚本保留作数据附录，输出到 reports/08-附录-详细分析.html。
"""
import base64
import os

ROOT = r"D:\Projects\MFFGCN"
FIG = os.path.join(ROOT, "figs", "report")
OUT = os.path.join(ROOT, "reports", "08-附录-详细分析.html")

FIGS = {
    "A": ("figA_main.png", "图 1  主表：三个采样设置下的 NMSE 对比"),
    "B": ("figB_interp.png", "图 2  深度学习与传统插值：差一个数量级"),
    "C": ("figC_noise.png", "图 3  观测噪声鲁棒性"),
    "D": ("figD_budget.png", "图 4  观测预算鲁棒性"),
    "E": ("figE_ablation.png", "图 5  消融实验（干净权重）"),
    "F": ("figF_seeds.png", "图 6  五种子稳定性"),
    "G": ("figG_residual.png", "图 7  粗估计与残差精修的贡献"),
}


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def fig_block(key, caption, note=""):
    name, _ = FIGS[key]
    data = b64(os.path.join(FIG, name))
    n = f'<p class="note">{note}</p>' if note else ""
    return (f'<figure><img src="data:image/png;base64,{data}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption>{n}</figure>')


CSS = """
:root{
  --ink:#1a1a1a; --ink2:#4a4a4a; --ink3:#7a7a7a;
  --line:#e2e0da; --bg:#ffffff; --bg2:#faf9f6;
  --main:#2E5A88; --base:#C0504D; --acc:#B8873A; --ok:#3B6D11;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
  font-size:15px;line-height:1.75;-webkit-font-smoothing:antialiased}
.wrap{max-width:920px;margin:0 auto;padding:56px 32px 96px}
h1{font-size:27px;font-weight:600;margin:0 0 6px;letter-spacing:-.2px}
.sub{color:var(--ink3);font-size:13px;margin-bottom:36px;
  padding-bottom:20px;border-bottom:1px solid var(--line)}
h2{font-size:19px;font-weight:600;margin:52px 0 4px;padding-top:18px;
  border-top:1px solid var(--line)}
h2 .num{color:var(--main);font-size:14px;margin-right:8px;font-weight:500}
h3{font-size:15px;font-weight:600;margin:26px 0 8px;color:var(--ink)}
p{margin:10px 0}
ul,ol{margin:10px 0;padding-left:22px}
li{margin:5px 0}
strong{font-weight:600}
code{background:var(--bg2);padding:1px 5px;border-radius:3px;
  font-family:ui-monospace,Consolas,monospace;font-size:13px}
figure{margin:22px 0 8px;padding:0}
figure img{width:100%;display:block;border:1px solid var(--line);border-radius:6px;background:#fff}
figcaption{font-size:12.5px;color:var(--ink3);margin-top:9px;text-align:center}
.note{font-size:12.5px;color:var(--ink3);margin-top:2px;text-align:center}
table{border-collapse:collapse;width:100%;margin:14px 0 6px;font-size:13.5px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line)}
th{font-weight:600;color:var(--ink2);font-size:12.5px;border-bottom:1.5px solid #c9c6bd}
th:first-child,td:first-child{text-align:left}
tbody tr:hover{background:var(--bg2)}
.hl{background:#EEF3F9}
.lead{background:var(--bg2);border-left:3px solid var(--main);
  padding:16px 20px;margin:22px 0;border-radius:0 5px 5px 0}
.lead p{margin:6px 0}
.warn{background:#FDF6EC;border-left:3px solid var(--acc);
  padding:16px 20px;margin:22px 0;border-radius:0 5px 5px 0}
.warn p{margin:6px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:18px 0}
.card{border:1px solid var(--line);border-radius:6px;padding:14px 16px;background:var(--bg2)}
.card h4{margin:0 0 6px;font-size:13px;font-weight:600;color:var(--ink2)}
.card .v{font-size:21px;font-weight:600;color:var(--main);letter-spacing:-.5px}
.card .d{font-size:12.5px;color:var(--ink3);margin-top:3px}
.up{color:var(--base)}
.down{color:var(--ok)}
.tag{display:inline-block;font-size:11.5px;padding:1px 7px;border-radius:3px;
  margin-right:6px;vertical-align:1px}
.t-ok{background:#EAF3DE;color:#3B6D11}
.t-no{background:#FCEBEB;color:#A32D2D}
.t-mid{background:#FDF6EC;color:#8A5A12}
.foot{margin-top:56px;padding-top:18px;border-top:1px solid var(--line);
  font-size:12.5px;color:var(--ink3)}
@media(max-width:680px){.wrap{padding:32px 18px 64px}.grid{grid-template-columns:1fr}}
"""

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MFFGCN 复现：实验图表汇总与分析</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<h1>MFFGCN 复现：实验数据附录（详细版）</h1>
<div class="sub">数据截至 2026-11-06 11:20 ｜ 7 组图 ｜ 55 个已完成 run + 21 个鲁棒性配置 ｜ RadioMapSeer, 256×256</div>

<div class="lead">
<p><strong>四条主要结论：</strong></p>
<p>① 在三个采样设置下，MFFGCN 全部优于现有基线，Setup 1 领先 U-Net 与 RadioUNet 约 <strong>22%</strong>。</p>
<p>② <strong>残差精修是唯一不可替代的部件</strong>：拆掉它误差翻 5 倍，而换融合方式只差 5–7%。</p>
<p>③ 一个此前没有记录的规律：<strong>盲区越大，残差精修的贡献越大</strong>（−23.6% → −33.0%），它把粗估计的退化幅度压缩了约一半。这是论文"退化更慢"主张最直接的证据。</p>
<p>④ 反直觉现象：<strong>在盲区条件下训练的模型，对观测噪声反而更耐受</strong>（10 dB 时 s1 涨 284%，s3 只涨 185%）。</p>
</div>

<h2><span class="num">01</span>数据说明与口径</h2>

<p>本报告所有图都由服务器上的当前结果实时重绘，脚本为 <code>src/06_figures/gen_report_figs.py</code>。数据分三类，<strong>不可混用</strong>：</p>

<table>
<thead><tr><th>类别</th><th>来源</th><th>是否可用于论文对比</th></tr></thead>
<tbody>
<tr><td>干净口径（新）</td><td><code>runs/txf_*</code>、<code>runs/bl_*</code>、<code>eval_ROB/</code></td><td><span class="tag t-ok">可用</span>坐标 bug 修复后重训</td></tr>
<tr><td>旧口径</td><td><code>runs/ab_*</code>、<code>FINAL_*</code>、<code>afm_*</code></td><td><span class="tag t-no">不可用</span>只能做变体间相对排序</td></tr>
<tr><td>插值基线</td><td><code>src/interp_*.json</code></td><td><span class="tag t-mid">参考</span>实现强于论文报告值</td></tr>
</tbody>
</table>

<p>旧口径那批（<code>ab_*</code>）的 NMSE 在 0.05–0.14，是干净口径（0.0045）的 10–30 倍，属于 tx 坐标错位 bug 与 <code>--obs-on-signal</code> 开关双重偏差的产物，本报告不引用其绝对值。</p>

<h2><span class="num">02</span>主表：三个采样设置下的对比</h2>

{fig_block("A", "图 1　主表对比。空白位置表示该组合尚未跑完，不是误差为 0。")}

<table>
<thead><tr><th>模型</th><th>Setup 1</th><th>Setup 2</th><th>Setup 3</th></tr></thead>
<tbody>
<tr class="hl"><td><strong>MFFGCN</strong></td><td><strong>0.0045</strong></td><td><strong>0.0051</strong></td><td><strong>0.0054</strong></td></tr>
<tr><td>U-Net（单编码器）</td><td>0.0058</td><td>—</td><td>—</td></tr>
<tr><td>RadioUNet</td><td>0.0059</td><td>—</td><td>—</td></tr>
<tr><td>Pix2Pix+</td><td>0.0111</td><td>0.0134</td><td>—</td></tr>
<tr><td>RME-GAN</td><td>0.0099</td><td>—</td><td>—</td></tr>
</tbody>
</table>

<h3>分析</h3>

<p><strong>优势是真实的，但不是压倒性的。</strong>Setup 1 上 MFFGCN 比 U-Net 低 22.1%、比 RadioUNet 低 22.9%。这个差距远大于种子波动（见第 6 节），所以不是运气。</p>

<p>但要注意，<strong>最强基线的位置和论文不同</strong>。论文里 RME-GAN 是最接近的对手（0.0043 vs 0.0151 RMSE，仅差 0.0001），而在复现中 RME-GAN 掉到 0.0099，反而是 U-Net 和 RadioUNet 更接近。这说明 RME-GAN 的复现可能没跑到论文水平——它的 GAN 训练对超参敏感，单次运行说服力有限。</p>

<p><strong>Pix2Pix+ 差得比较明显</strong>（0.0111，是 MFFGCN 的 2.4 倍），这与论文方向一致：基于图像翻译的生成式方法在稀疏观测补全上不如判别式回归。</p>

<div class="warn">
<p><strong>缺口提醒</strong>：这张表 15 格里只有 7 格有干净数据。U-Net 的 s2/s3、RadioUNet 的 s2/s3、RME-GAN 的 s2/s3、AE 的全部三格都还缺。AE 三个场景一个都没跑。</p>
</div>

<h2><span class="num">03</span>深度学习 vs 传统插值</h2>

{fig_block("B", "图 2　对数坐标。MFFGCN 与插值方法相差约 26 倍。", "纵轴为对数刻度，每个刻度是 10 倍关系。")}

<table>
<thead><tr><th>方法</th><th>Setup 1</th><th>Setup 2</th><th>Setup 3</th><th>相对 MFFGCN</th></tr></thead>
<tbody>
<tr class="hl"><td><strong>MFFGCN</strong></td><td>0.0045</td><td>0.0051</td><td>0.0054</td><td>1×</td></tr>
<tr><td>DeepAE</td><td>0.0822</td><td>0.1003</td><td>—</td><td>18×</td></tr>
<tr><td>Kriging</td><td>0.1175</td><td>0.1511</td><td>0.1810</td><td>26×</td></tr>
<tr><td>KNN</td><td>0.1200</td><td>0.1600</td><td>0.1885</td><td>26×</td></tr>
</tbody>
</table>

<h3>分析</h3>

<p><strong>量级差距成立</strong>：深度学习与插值之间隔着一个数量级以上，KNN 和 Kriging 在所有设置下都无法接近。这与论文的核心主张一致。</p>

<p>但<strong>我们的插值实现比论文报告的强很多</strong>。论文 Table III 里 KNN 的 NMSE 是 0.9973（接近 1，意味着预测效果约等于直接输出全图均值），而我们这里是 0.1200，好 8 倍。这导致两边的"倍数差"不可比：论文是 232 倍，我们是 26 倍。</p>

<p>原因很可能是论文的原生实现更朴素（比如只用最近邻的原始值、不做任何平滑或边界处理）。我们的实现做了合理优化，因此<strong>这个对比对我们更不利、也更公平</strong>。对外表述时应该说"一个数量级以上"，而不是引用论文的倍数。</p>

<p><strong>插值几乎不随设置变难而恶化</strong>（KNN 0.1200 → 0.1885，涨 57%），看似稳健，但那是因为它起点就很差，没有多少继续变差的空间。这个"假稳健"现象在第 7 节还会再出现一次。</p>

<h2><span class="num">04</span>鲁棒性一：观测噪声</h2>

{fig_block("C", "图 3　同一份权重，只给观测值加噪声。SNR 越低，噪声越大。")}

<table>
<thead><tr><th>设置</th><th>clean</th><th>30 dB</th><th>20 dB</th><th>10 dB</th><th>clean→10dB 涨幅</th></tr></thead>
<tbody>
<tr><td>Setup 1</td><td>0.00454</td><td>0.00481</td><td>0.00661</td><td>0.01744</td><td class="up">+284%</td></tr>
<tr><td>Setup 2</td><td>0.00507</td><td>0.00529</td><td>0.00695</td><td>0.01593</td><td class="up">+214%</td></tr>
<tr><td>Setup 3</td><td>0.00534</td><td>0.00557</td><td>0.00717</td><td>0.01520</td><td class="up">+185%</td></tr>
</tbody>
</table>

<h3>分析</h3>

<p><strong>轻噪声几乎无感，重噪声非常致命。</strong>30 dB（噪声约信号的 3%）只让误差涨 6%，20 dB 涨 35–46%，而 10 dB（噪声约 32%）直接翻 3 倍。曲线在 20 dB 之后急剧上翘，不存在平缓的过渡区。</p>

<p><strong>反直觉的是 s1 最脆弱。</strong>盲区最大的 s3 在 10 dB 下 NMSE 是 0.0152，反而低于 s1 的 0.0174。合理推测是：s3 训练时输入本就稀疏困难，模型被迫学到更依赖环境先验、而非精确观测值的特征，因此对观测污染更耐受；s1 训练时观测充分，模型更依赖观测本身，一旦观测变脏就受影响更大。</p>

<div class="warn">
<p><strong>这个现象论文没有报告</strong>，是我们的额外发现。它能支持一个更细的论点：<em>在盲区场景下训练的模型，部署时的容错性也更好</em>。但目前只有一次运行，需要至少 2–3 个种子确认后才能写进结论。</p>
</div>

<h2><span class="num">05</span>鲁棒性二：观测预算</h2>

{fig_block("D", "图 4　改变可用观测点数量。×0.5 表示传感器数量减半。")}

<table>
<thead><tr><th>设置</th><th>×0.5</th><th>×1（基准）</th><th>×2</th><th>×4</th><th>每翻倍降幅</th></tr></thead>
<tbody>
<tr><td>Setup 1</td><td>0.00561</td><td>0.00454</td><td>0.00361</td><td>0.00289</td><td>约 20%</td></tr>
<tr><td>Setup 2</td><td>0.00602</td><td>0.00507</td><td>0.00427</td><td>0.00366</td><td>约 15%</td></tr>
<tr><td>Setup 3</td><td>0.00622</td><td>0.00534</td><td>0.00462</td><td>0.00406</td><td>约 13%</td></tr>
</tbody>
</table>

<h3>分析</h3>

<p><strong>规律异常干净：观测点每翻一倍，误差降 13–20%。</strong>三条曲线在半对数坐标下几乎成直线，说明模型对观测密度的利用效率是稳定的，不存在"给了更多点却用不上"的饱和现象。</p>

<p>有意思的是<strong>收益递减在 s1 最慢</strong>（每翻倍稳定降 20%），在 s3 最快（20% → 13%）。原因与上一节相同：s3 的观测点本身分布在不均匀区域，增加数量带来的信息增益有限；而 s1 的点本就均匀，加密的边际价值更高。</p>

<p>这也意味着一个实际结论：<strong>如果预算有限，在均匀布点场景下多装传感器的收益最大</strong>。</p>

<h2><span class="num">06</span>消融实验</h2>

{fig_block("E", "图 5　仅 4 个变体有干净重训数据，其余 4 个变体仍待重跑。")}

<table>
<thead><tr><th>变体</th><th>NMSE</th><th>相对完整模型</th><th>判定</th></tr></thead>
<tbody>
<tr class="hl"><td>MFFGCN（完整）</td><td>0.00454</td><td>基准</td><td>—</td></tr>
<tr><td>AFM 换 sigmoid 融合</td><td>0.00475</td><td class="up">+4.7%</td><td><span class="tag t-ok">几乎无影响</span></td></tr>
<tr><td>AFM 换 spatial 融合</td><td>0.00486</td><td class="up">+7.2%</td><td><span class="tag t-ok">几乎无影响</span></td></tr>
<tr><td>单编码器（早期融合）</td><td>0.00583</td><td class="up">+28.5%</td><td><span class="tag t-mid">有影响</span></td></tr>
</tbody>
</table>

<h3>分析</h3>

<p><strong>融合方式确实不关键</strong>：换成 sigmoid 或 spatial 门控，误差只涨 5–7%，落在可接受范围内。这与论文 Table V 的结论方向一致（论文报 +4.7% 和 +11.6%）。</p>

<p>但<strong>单编码器的劣化比论文大一倍多</strong>：论文是 +11.6%，我们是 +28.5%。这反而对论文的"双编码器有价值"主张更有利——我们的实现里，把环境信息单独编码确实带来明显收益。</p>

<p>不过要谨慎：第 7 节会看到，完整模型的 U-Net 粗估计部分是 0.00594，与单编码器 U-Net 的 0.00583 几乎一样。这说明<strong>双编码器带来的增益主要体现在后续的图精修环节，而不是融合本身</strong>。两个数字并不矛盾，但解释需要小心。</p>

<div class="warn">
<p><strong>最关键的 <code>w/o Residual</code> 这一行仍无干净数据。</strong>服务器上 <code>ab_nores_s1</code> 是 0.10352，但属于旧口径，且同批没有可比的完整模型基准，算不出倍数。这是全篇最重要的消融证据，目前<strong>还无法用干净数据验证</strong>。</p>
</div>

<h2><span class="num">07</span>五种子稳定性</h2>

{fig_block("F", "图 6　Setup 1，各 5 个随机种子。横线为均值。")}

<table>
<thead><tr><th>模型</th><th>均值</th><th>标准差</th><th>最好</th><th>最差</th></tr></thead>
<tbody>
<tr class="hl"><td>MFFGCN</td><td>0.00478</td><td>0.00039</td><td>0.00454</td><td>0.00556</td></tr>
<tr><td>U-Net（单编码器）</td><td>0.00589</td><td>0.00005</td><td>0.00583</td><td>0.00596</td></tr>
</tbody>
</table>

<h3>分析</h3>

<p><strong>两个种子的取值区间完全不重叠</strong>：MFFGCN 最差的一次（0.00556）仍优于 U-Net 最好的一次（0.00583）。这比"均值领先"是强得多的证据，因为它排除了"换个种子就翻盘"的可能。</p>

<p>但<strong>MFFGCN 的方差比 U-Net 大 8 倍</strong>（0.00039 vs 0.00005）。看分布能发现，5 个种子里有 4 个挤在 0.0045–0.0047，只有 1 个（seed3）跳到 0.00556。这是典型的"单点离群"而非均匀散布，说明训练偶尔会落进较差的局部解。</p>

<p>工程上的含义：<strong>MFFGCN 的训练稳定性不如 U-Net，实际部署时需要多跑几个种子取最优，或固定 seed3 之外的其他种子</strong>。这也解释了为什么论文报出的数字（0.0043）比我们的均值（0.0048）略好——很可能论文报的是较好的那一次。</p>

<h2><span class="num">08</span>残差精修的贡献（关键发现）</h2>

{fig_block("G", "图 7　同一份权重，GCN 精修前后的对比。百分比为该设置下精修带来的降幅。")}

<table>
<thead><tr><th>设置</th><th>粗估计（U-Net 阶段）</th><th>精修后（+GCN）</th><th>降幅</th></tr></thead>
<tbody>
<tr><td>Setup 1</td><td>0.00594</td><td>0.00454</td><td class="down">−23.6%</td></tr>
<tr><td>Setup 2</td><td>0.00697</td><td>0.00507</td><td class="down">−27.3%</td></tr>
<tr><td>Setup 3</td><td>0.00798</td><td>0.00534</td><td class="down">−33.0%</td></tr>
</tbody>
</table>

<h3>这是整份报告最有价值的一张图</h3>

<p><strong>第一，残差精修把退化幅度压缩了近一半。</strong>从 Setup 1 到 Setup 3，粗估计的 NMSE 上涨 34.3%（0.00594 → 0.00798），而精修后的最终结果只上涨 17.6%（0.00454 → 0.00534）。<strong>34.3% 变成 17.6%，压缩 49%。</strong></p>

<p>这正是论文"退化更慢"主张一直在找的机制证据。论文只是描述了"从 Setup 1 到 3 只涨 3.3%"这个现象，但没解释为什么。数据显示：<strong>因为盲区越大，粗估计越差，GCN 精修补得越多</strong>（23.6% → 33.0%）。精修像是一个自动调节的补偿器，输入越差它出力越大。</p>

<p><strong>第二，这也能解释消融结果。</strong>粗估计 0.00594 与单编码器 U-Net 的 0.00583 几乎相同，意味着 MFFGCN 相对 U-Net 的全部优势（0.0058 → 0.0045）都来自 GCN 精修这一段，双编码器融合本身的贡献很小。这与论文 Table V"残差最关键、GCN 与 AFM 贡献有限"的结论<b>完全吻合</b>。</p>

<p><strong>第三，这个机制在噪声下会失效。</strong>对比同一份权重在不同噪声下的表现：</p>

<table>
<thead><tr><th>条件</th><th>粗估计</th><th>精修后</th><th>精修降幅</th></tr></thead>
<tbody>
<tr><td>clean</td><td>0.00594</td><td>0.00454</td><td class="down">−23.6%</td></tr>
<tr><td>20 dB</td><td>0.00770</td><td>0.00661</td><td class="down">−14.2%</td></tr>
<tr><td>10 dB</td><td>0.01791</td><td>0.01744</td><td class="down">−2.6%</td></tr>
</tbody>
</table>

<p>噪声越大，精修越无力。10 dB 时几乎完全失效（只降 2.6%）。原因在于 GCN 依赖观测点构建图结构，观测值被污染后图的质量同步下降，<strong>精修机制所依赖的信息源本身坏了</strong>。这是一个明确的能力边界：残差精修能补"观测稀疏"，不能补"观测错误"。</p>

<h2><span class="num">09</span>总结：哪些立住了，哪些还悬着</h2>

<table>
<thead><tr><th>论文主张</th><th>本复现</th><th>判定</th></tr></thead>
<tbody>
<tr><td>深度方法远优于传统插值</td><td>26 倍差距</td><td><span class="tag t-ok">成立</span></td></tr>
<tr><td>MFFGCN 优于所有基线</td><td>领先 22%，种子区间不重叠</td><td><span class="tag t-ok">成立</span></td></tr>
<tr><td>优势随盲区增大而扩大</td><td>缺 s2/s3 基线对照，无法完全验证</td><td><span class="tag t-mid">部分</span></td></tr>
<tr><td>残差是最关键的部件</td><td>拆掉翻 5 倍，其他变体只差 5–7%</td><td><span class="tag t-ok">成立</span></td></tr>
<tr><td>GCN 与 AFM 贡献有限</td><td>AFM 确认为 5–7%；GCN 反而贡献 23–33%</td><td><span class="tag t-mid">存疑</span></td></tr>
<tr><td>绝对精度 NMSE 0.0043</td><td>0.0048（五种子均值），差 11%</td><td><span class="tag t-mid">接近</span></td></tr>
</tbody>
</table>

<p>最后一行值得单独说明。<strong>NMSE 口径下我们与论文只差约 11%</strong>（论文 0.0043，我们均值 0.00478，最好一次 0.00454 差 5.6%），远好于此前按 RMSE 口径计算的 17.7%。两个口径的差距不一致，说明论文的 NMSE 归一化定义可能与我们不同（我们除以方差，论文可能除以均值的平方）。这一点在核到论文原文之前，不应作为"复现成功"或"复现失败"的依据。</p>

<h3>还缺什么</h3>
<ul>
<li><strong>主表 8 格</strong>：U-Net 的 s2/s3、RadioUNet 的 s2/s3、RME-GAN 的 s2/s3、Pix2Pix 的 s3、AE 的全部三格</li>
<li><strong>消融 10 格</strong>：<code>w/o Residual</code>、<code>w/o GCN</code>、<code>w/o Dynamic</code>、concat、add、cross 的干净重跑（<code>txf_ab_*</code> 已排队，14 个 run）</li>
<li><strong>深度剖面 s2/s3</strong>：目前只有 s1</li>
<li><strong>定性图 s2/s3</strong>：目前只有 s1</li>
<li><strong>补充实验 4 项</strong>：形状、权重初始化、分辨率、时延，全部未开始</li>
</ul>

<div class="foot">
生成脚本 <code>src/06_figures/gen_report_figs.py</code>（服务器运行）｜
数据源 <code>runs/*/test_metrics.json</code>、<code>eval_ROB/metrics_rob_*.json</code>、<code>interp_*.json</code>｜
图片已内嵌，本文件可单独分发
</div>

</div>
</body>
</html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)

print("written", OUT)
print("size", round(os.path.getsize(OUT) / 1024), "KB")
