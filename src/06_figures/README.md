# 06_figures — 出图层

只读结果，不训练、不评估。跑完评估之后才能用。

## 文件

| 文件 | 输出 | 用途 |
|---|---|---|
| `make_figs.py` | `figs/fig1_setups.png` 等 5 张 | **论文主图** |
| `make_table.py` | LaTeX / markdown 表格 | 论文表格 |
| `plot_ablation.py` | 消融对比柱状图 | 表 V 可视化 |
| `plot_paper_vs_ours.py` | 论文数值 vs 复现数值 | 趋势一致性 |
| `plot_compare.py` / `plot_compare2.py` | 模型两两对比曲线 | |
| `plot_gap.py` | 泛化 gap 曲线 | train vs test 差距 |
| `plot_robustness.py` | 鲁棒性曲线（噪声 / 观测预算） | |

## make_figs.py 的五张主图

| 图 | 内容 |
|---|---|
| `fig1_setups.png` | 5 种采样设定的示意 |
| `fig2_paper.png` | 与论文数值对比 |
| `fig3_ablation.png` | 消融实验 |
| `fig4_best_last.png` | best vs last checkpoint |
| `fig5_quali.png` | 定性对比（真值/观测/预测/误差） |

## 输入依赖

```
make_figs.py  ← results/*.json + runs/*/test_metrics.json + eval_out/sample_*.png
```

所以顺序必须是：**训练 → 评估 → 出图**，跳步会拿到旧数据。

## 趋势对比报告（在 08_scripts/06_报告/）

与论文趋势的匹配度分析不在本目录，而是一个独立的 HTML 报告生成器：

```
08_scripts/06_报告/
    make_trend_report.py    纯 SVG 生成 4 张趋势图 + HTML
    update_trend.py         SSH 拉快照 → 更新 trend_data.json → 重渲染
    trend_data.json         数据源（含论文原始数值）
→ 输出 reports/07-与论文趋势一致性对比.html + figs/fig_trend_*.png
```

这个是自动化流程，`update_trend.py` 只在数据有变化时才重写报告。
