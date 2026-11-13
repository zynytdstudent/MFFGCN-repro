# -*- coding: utf-8 -*-
"""build_fig_report.py - 简版报告：一图一段，末尾对照论文结论。

设计约束（用户要求）：
  1. 每节 = 一张图 + 一段文字，不堆表格；
  2. 最后一节回答"哪些结论对得上论文"；
  3. 全部数据表和分析细节移入 reports/08-附录-详细分析.html（build_fig_report_detail.py）。
图片以 base64 内嵌，产出的 HTML 可单独分发。
"""
import base64
import os

ROOT = r"D:\Projects\MFFGCN"
FIG = os.path.join(ROOT, "figs", "report")
OUT = os.path.join(ROOT, "reports", "08-实验图表汇总与分析.html")

FIGS = {
    "A": ("figA_main.png", "图 1　三个采样设置下的 NMSE"),
    "B": ("figB_interp.png", "图 2　深度学习 vs 传统插值（对数轴）"),
    "C": ("figC_noise.png", "图 3　观测噪声鲁棒性"),
    "D": ("figD_budget.png", "图 4　观测预算鲁棒性"),
    "E": ("figE_ablation.png", "图 5　消融实验（干净权重）"),
    "F": ("figF_seeds.png", "图 6　五种子稳定性"),
    "G": ("figG_residual.png", "图 7　粗估计 vs 残差精修"),
}


def b64(name):
    with open(os.path.join(FIG, name), "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def fig(key):
    name, cap = FIGS[key]
    return (f'<figure><img src="data:image/png;base64,{b64(name)}" alt="{cap}">'
            f'<figcaption>{cap}</figcaption></figure>')


CSS = """
:root{
  --ink:#1a1a1a; --ink2:#4a4a4a; --ink3:#7a7a7a;
  --line:#e4e2dc; --bg:#ffffff; --bg2:#faf9f6;
  --main:#2E5A88; --base:#C0504D; --acc:#B8873A; --ok:#3B6D11;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
  font-size:15.5px;line-height:1.8;-webkit-font-smoothing:antialiased}
.wrap{max-width:880px;margin:0 auto;padding:56px 32px 90px}
h1{font-size:26px;font-weight:600;margin:0 0 8px;letter-spacing:-.2px}
.sub{color:var(--ink3);font-size:13px;margin-bottom:34px;
  padding-bottom:18px;border-bottom:1px solid var(--line)}
h2{font-size:18px;font-weight:600;margin:54px 0 2px}
h2 .num{color:var(--main);font-size:13px;margin-right:10px;font-weight:500;
  font-variant-numeric:tabular-nums}
p{margin:12px 0}
strong{font-weight:600}
code{background:var(--bg2);padding:1px 5px;border-radius:3px;
  font-family:ui-monospace,Consolas,monospace;font-size:13px}
figure{margin:20px 0 4px}
figure img{width:100%;display:block;border:1px solid var(--line);border-radius:6px;background:#fff}
figcaption{font-size:12.5px;color:var(--ink3);margin-top:8px;text-align:center}
.takeaway{background:var(--bg2);border-left:3px solid var(--main);
  padding:14px 18px;margin:18px 0 0;border-radius:0 5px 5px 0}
.takeaway p{margin:0}
table{border-collapse:collapse;width:100%;margin:20px 0 6px;font-size:13.5px;line-height:1.6}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--ink2);font-size:12.5px;border-bottom:1.5px solid #c9c6bd}
tbody tr:hover{background:var(--bg2)}
td.c{text-align:center;white-space:nowrap}
.tag{display:inline-block;font-size:11.5px;padding:1px 7px;border-radius:3px;white-space:nowrap}
.t-ok{background:#EAF3DE;color:#3B6D11}
.t-no{background:#FCEBEB;color:#A32D2D}
.t-mid{background:#FDF6EC;color:#8A5A12}
.up{color:var(--base)}
.down{color:var(--ok)}
.foot{margin-top:56px;padding-top:16px;border-top:1px solid var(--line);
  font-size:12.5px;color:var(--ink3);line-height:1.7}
@media(max-width:680px){.wrap{padding:32px 18px 60px}table{font-size:12.5px}}
"""

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MFFGCN 复现：七张图看结论</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<h1>MFFGCN 复现：七张图看结论</h1>
<div class="sub">2026-11-06 ｜ RadioMapSeer 256×256 ｜ 每节一张图 + 一段结论 ｜ 末节对照论文</div>

<p>一句话总览：<strong>模型确实更强（领先 22%），但真正被数据解释清楚的是"残差精修"这一段——它把盲区带来的退化压掉了一半。论文关于"优势随盲区扩大"的主张，目前还缺基线对照，未能验证。</strong></p>

<h2><span class="num">01</span>主表：领先 22%，但最强对手换人了</h2>
{fig("A")}
<div class="takeaway"><p>Setup 1 上 MFFGCN（0.0045）比 U-Net（0.0058）低 22.1%、比 RadioUNet（0.0059）低 22.9%，这个差距远超种子波动（最大 8%），不是运气。但论文里最接近的对手是 RME-GAN，在我们这里它只跑到 0.0099，明显没复现出论文水平；反而是两个轻量 U-Net 更接近。<strong>结论：优势成立，但"打败了谁"要按我们自己的表来说，不能照抄论文的对手排序。</strong>（这张表 15 格里有干净数据的只有 7 格。）</p></div>

<h2><span class="num">02</span>深度学习 vs 插值：差一个数量级</h2>
{fig("B")}
<div class="takeaway"><p>KNN 与 Kriging 在 0.12–0.19，MFFGCN 在 0.0045，相差 26 倍——深度学习与插值之间确实隔着一个数量级以上，方向与论文完全一致。但要小心倍数：论文 Table III 里 KNN 的 NMSE 接近 1（约 0.997，相当于直接输出全图均值），我们的实现只 0.1200，好了 8 倍，所以论文那个两百多倍的差距不能直接引用。<strong>对外只说"一个数量级以上"。</strong>另外插值的"稳定"是假稳健——它起点太差，没有继续变差的空间。</p></div>

<h2><span class="num">03</span>噪声：轻噪声无感，10 dB 崩三倍</h2>
{fig("C")}
<div class="takeaway"><p>30 dB 只让误差涨 6%，20 dB 涨 35–46%，10 dB 直接翻 3 倍（+185% 至 +284%），20 dB 之后陡然上翘，没有平缓过渡区。反直觉的是<strong>盲区最大的 s3 反而最耐受</strong>（+185%，s1 是 +284%）：s3 训练时观测本就稀疏，模型被迫更依赖环境先验而不是观测值本身。这条规律论文没有报告，且目前只有单次运行，要 2–3 个种子确认后才能写进结论。</p></div>

<h2><span class="num">04</span>观测预算：每翻一倍，误差降 13–20%</h2>
{fig("D")}
<div class="takeaway"><p>三条曲线在半对数坐标下几乎成直线：观测点每翻倍，NMSE 稳定降 13–20%，没有出现"给了更多点却用不上"的饱和。收益在 s1 最大（约 20%）、s3 最小（约 13%），原因同上——均匀布点加密的边际价值更高。<strong>落到工程上：预算有限时，优先在均匀布点的场景加传感器。</strong></p></div>

<h2><span class="num">05</span>消融：融合方式不重要，残差才重要</h2>
{fig("E")}
<div class="takeaway"><p>把 AFM 换成 sigmoid 或 spatial 门控，误差只涨 4.7% 和 7.2%，落在噪声范围内，说明<strong>融合方式确实不是关键</strong>——论文的原话是"AFM 比直接 concat 再好 4–6%"，量级对得上。单编码器涨 28.5%（反过来说双编码器降 22.2%），比论文报的"降 9–10%"大一倍多，看似对"双编码器有价值"更有利——但第 7 节会说明，这部分增益其实来自后面的图精修。<strong>最关键的 <code>w/o Residual</code> 一格仍缺干净数据，这是全篇唯一还悬着的决定性证据。</strong></p></div>

<h2><span class="num">06</span>五种子：区间不重叠，但方差大 8 倍</h2>
{fig("F")}
<div class="takeaway"><p>MFFGCN 最差的一次（0.00556）仍优于 U-Net 最好的一次（0.00583），5 个种子全部胜出——这比"均值领先"强得多，排除了"换个种子就翻盘"。但标准差是 0.00039 vs 0.00005，相差 8 倍，且由 seed3 单点离群造成（其余 4 个挤在 0.0045–0.0047）。<strong>含义：训练偶发落入较差局部解，实际使用建议多跑几个种子取最优。</strong>论文报的 0.0043 略好于我们的均值 0.0048，很可能报的是较好那一次。</p></div>

<h2><span class="num">07</span>残差精修：把退化压掉一半（最关键的一张图）</h2>
{fig("G")}
<div class="takeaway"><p>从 Setup 1 到 Setup 3，粗估计的 NMSE 上涨 34.3%（0.00594→0.00798），精修后只上涨 17.6%（0.00454→0.00534）——<strong>退化幅度被压缩了 49%</strong>，而且盲区越大补得越多（降幅 −23.6% → −27.3% → −33.0%）。这就是论文"退化更慢"主张一直缺的机制解释：精修像个自动调节的补偿器，输入越差出力越大。但它有明确边界：clean 下降 23.6%、20 dB 降 14.2%、10 dB 只剩 2.6%，<strong>能补"观测稀疏"，补不了"观测错误"</strong>。同一组数据还说明，粗估计（0.00594）几乎等于单编码器 U-Net（0.00583），所以 MFFGCN 相对 U-Net 的优势基本全部来自这一段。</p></div>

<h2><span class="num">08</span>与论文结论的对应关系</h2>

<p>论文的主张逐条对本复现的数据，判定分三档：<span class="tag t-ok">对得上</span> 数据支持；<span class="tag t-mid">部分</span> 方向对但没验证完；<span class="tag t-no">对不上</span> 数据与论文说法冲突。</p>

<table>
<thead><tr><th style="width:27%">论文结论（出处）</th><th style="width:24%">论文原文数字</th><th style="width:29%">本复现</th><th style="width:20%">判定</th></tr></thead>
<tbody>
<tr>
  <td>深度方法远优于传统插值（Table III / Fig. 7）</td>
  <td>KNN、Kriging 误差远高（该表为图片，未逐格取数）</td>
  <td>相差 26 倍，一个数量级以上；方向完全一致</td>
  <td class="c"><span class="tag t-ok">对得上</span><br><span style="color:#7a7a7a;font-size:12px">只引用"一个数量级"</span></td>
</tr>
<tr>
  <td>MFFGCN 优于全部 SOTA（Table IV）</td>
  <td>Setup 1：NMSE 0.0043、RMSE 0.0150</td>
  <td>NMSE 0.00478（五种子均值），差 11%；最好一次 0.00454，差 5.6%</td>
  <td class="c"><span class="tag t-ok">对得上</span><br><span style="color:#7a7a7a;font-size:12px">绝对值接近</span></td>
</tr>
<tr>
  <td>盲区变大时退化更慢，优势反而扩大</td>
  <td>Setup 2/3 自身 RMSE 仅 +2.67%/+3.33%；相对 SOTA 领先 6.67%/9.36%（RMSE）、11.76%/14.82%（NMSE）</td>
  <td>缺 s2/s3 基线，算不出相对领先；自身 NMSE 0.00454→0.00534 涨 17.6%（论文说的是 RMSE，口径不同，但趋势差异明显）</td>
  <td class="c"><span class="tag t-mid">未验证</span><br><span style="color:#7a7a7a;font-size:12px">趋势没复现</span></td>
</tr>
<tr>
  <td>残差输出是最关键部件（Table V）</td>
  <td>0.0043→0.0231（×5.4），RMSE 0.0150→0.0347</td>
  <td>该变体仍缺干净数据；间接证据：精修单独贡献 23–33%，且粗估计（0.00594）≈ 单编码器（0.00583）</td>
  <td class="c"><span class="tag t-mid">间接支持</span></td>
</tr>
<tr>
  <td>双编码器优于单编码器（Table V）</td>
  <td>NMSE 降 9–10%（三个设置）</td>
  <td>0.00583→0.00454，降 22.2%，幅度是论文两倍多；但增益实际出现在 GCN 段而非融合段</td>
  <td class="c"><span class="tag t-ok">对得上</span><br><span style="color:#7a7a7a;font-size:12px">归因不同</span></td>
</tr>
<tr>
  <td>AFM 优于直接 concat（Table V）</td>
  <td>再降 4–6%</td>
  <td>sigmoid +4.7%、spatial +7.2%（即 AFM 更优 4.5%/6.7%）</td>
  <td class="c"><span class="tag t-ok">对得上</span></td>
</tr>
<tr>
  <td>动态建图有用（w/o Dynamic）</td>
  <td>NMSE +6%（最不均匀场景）</td>
  <td>该变体无干净数据</td>
  <td class="c"><span class="tag t-mid">未验证</span></td>
</tr>
<tr>
  <td>FLOPs 少 14.5%，时延"基本持平"（Table VIII）</td>
  <td>40.32 G、18.23 M、5.17 ms</td>
  <td>FLOPs 完全一致；参数比 RadioUNet 高 37%；5.17 ms 是三者<strong>最慢</strong>（+1.4%）</td>
  <td class="c"><span class="tag t-no">需修正</span><br><span style="color:#7a7a7a;font-size:12px">应写"略慢"</span></td>
</tr>
</tbody>
</table>

<p>归纳：<strong>四条对得上</strong>（深度 vs 插值、整体领先、双编码器、AFM）、<strong>一条间接支持</strong>（残差最关键——用精修的 23–33% 贡献反推）、<strong>两条没验证</strong>（盲区越大优势越大、动态建图有用）、<strong>一条要改写</strong>（时延不是持平，是略慢 1.4%，参数还多了 37%）。</p>

<p>其中"盲区越大优势越大"这条是论文的核心卖点，也是我们最没验证住的一条：论文说自身 RMSE 只涨 3.33%，我们的 NMSE 涨了 17.6%。口径不同（RMSE vs NMSE）不能直接相减，但方向上的差距很明显，在补齐 s2/s3 基线之前不应替论文背书。另外"绝对精度差 11%"要先核到论文的 NMSE 归一化定义（除以方差还是均值平方）再判断，不宜当作复现成功或失败的依据。</p>

<div class="foot">
数据来源 <code>runs/*/test_metrics.json</code>、<code>eval_ROB/metrics_rob_*.json</code>、<code>interp_*.json</code>｜
出图脚本 <code>src/06_figures/gen_report_figs.py</code>（服务器运行）｜
全部数字表与逐条分析见 <code>reports/08-附录-详细分析.html</code>｜
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
