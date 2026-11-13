# 04_eval — 评估层

训练完之后才算。所有正式数字都出自这一层。

## 文件

| 文件 | 作用 | 输出 |
|---|---|---|
| `evaluate.py` | 单模型完整评估：分区指标 + 5 联定性图 | `metrics_<name>.json` + `sample_*.png` |
| `eval_all.py` | 批量评估多个 checkpoint | 多个 metrics json |
| `compare.py` | 两两对比 | json / md |
| `compare_all.py` | 汇总所有模型 → 横向对比表 | `comparison.json/md` |
| `reeval.py` | 用**统一口径**重评历史 run | 修 bug 后必须跑 |
| `agg_seeds.py` | 多种子聚合（均值 ± 标准差） | `<setup>_agg.json` |
| `agg_report.py` | 生成 markdown 报告 | `.md` |
| `collect_trend.py` | 扫全部 `runs/*/test_metrics.json` | JSON 快照 |

## 分区指标（本层的核心价值）

`evaluate.py` 把误差拆成三个区域分别统计：

| 区域 | 含义 | 看什么 |
|---|---|---|
| `all` | 全图 | 论文主指标 |
| `constr` | 盲区（不允许布点的地方） | 模型是真懂了还是只会抄观测 |
| `obs` | 已观测区（有传感器的像素） | 是否保住了真值 |

如果 `obs` 误差很小但 `constr` 误差很大 → 模型没学到传播规律，只是在插值。

同时报告 `coarse` 和 `fine` 两组，**两者之差就是 GCN 精修段的净收益**。

## 口径注意事项

- **跨区域比较请统一看 `nmse`**。`rmse` 的分母统一用了全图像素数
  （`n_img * 256 * 256`），不是该区域像素数，所以 `constr_rmse` / `obs_rmse`
  的绝对值偏小，只在同区域之间比较才有效。
- 评估用 `seed=2024`，与训练 seed 无关 —— 故意用一套独立的观测点抽样。
- `--obs-on-signal` 必须和训练时一致。

## 典型用法

```bash
# 单模型评估
python 04_eval/evaluate.py --ckpt runs/s1_seed0/best.pt --cache ../data/cache \
    --setup s1 --out ../eval_out --name s1 --obs-on-signal

# 修 bug 后重算全部历史结果
python 04_eval/reeval.py --cache ../data/cache

# 多种子聚合
python 04_eval/agg_seeds.py --pattern "runs/s1_seed*" --out ../results/s1_agg.json
```
