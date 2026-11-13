# runs — 训练输出

每次训练产生一个 `runs/<tag>/` 目录，`tag` 由 `--tag` 指定
（不指定则默认 `<setup>_seed<seed>`）。

> **本地这个目录是空的。** 全部训练产物在服务器
> `zjut148:/public2/czz/ZhangYu/project/mffgcn/runs/`。

## 一个 run 目录里的 4 个文件

| 文件 | 内容 | 谁写 | 谁读 |
|---|---|---|---|
| `train_log.csv` | 每 epoch 一行 | `train.py` | `status.sh`（画进度表） |
| `best.pt` | val NMSE 最低的权重 + args | `train.py` | `evaluate.py` |
| `last.pt` | 最后一轮的权重 | `train.py` | 断点续训 |
| `test_metrics.json` | 最终测试指标 | `train.py` 结尾 | **下游全部** |

## train_log.csv 的列

```
epoch, lr, train_loss, mse_coarse, mse_fine,
val_nmse, val_rmse, val_coarse_nmse, val_coarse_rmse, sec
```

`val_nmse` 是选模型的依据；`val_coarse_nmse` 是 U-Net 单段的水平，
两者之差就是 GCN 精修段的净贡献。

## test_metrics.json 的内容

```json
{
  "nmse": 0.0181,
  "rmse": 0.0421,
  "nmse_coarse": 0.0243,     // 只有 U-Net 时
  "rmse_coarse": 0.0488,
  "n_images": 8080,
  "best_epoch": 47,
  "total_minutes": 268.4,
  "setup": "s1"
}
```

`04_eval/collect_trend.py` 扫描所有 `runs/*/test_metrics.json` 汇总成快照，
`agg_seeds.py` 按 tag 前缀聚合多种子。

⚠ **`best.pt` 里含 `args`**，`evaluate.py::build_from_ckpt` 靠它重建网络结构。
删掉 args 会导致评估时结构对不上。

## 归档

历史废弃的 run 移到 `runs_archive/`（服务器上）。`src/runs/_legacy_buggy/`
下有一份 `MOVED.txt` 记录那些是坐标 bug 期间的产物，指标不可用。
