# 03_train — 训练层

## 文件

| 文件 | 作用 |
|---|---|
| `train.py` | **主训练入口**。MFFGCN 和所有消融变体都靠参数开关跑 |
| `baseline_train.py` | 深度基线（U-Net / Pix2pix 等）的训练 |
| `rme_gan_train.py` | RME-GAN 复现 |
| `sweep.py` | 批量扫参 / 扫种子 |

## train.py 一次迭代

```
DataLoader → move() 搬 GPU → fwd() 前向 → MFFGCNLoss → AMP 反传 → 更新
```

AMP 四步顺序不能错：

```python
scaler.scale(loss).backward()          # ① 放大
scaler.unscale_(opt)                   # ② 还原（必须在 clip 之前）
torch.nn.utils.clip_grad_norm_(..., 5) # ③ 裁剪
scaler.step(opt); scaler.update()      # ④ 更新 + 调缩放系数
```

如果先 clip 再 unscale，等于裁的是"被放大 65536 倍的梯度"，阈值形同虚设。

## 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--setup` | s1 | 5 种采样设定 |
| `--epochs` / `--decay-epoch` | 50 / 40 | 第 40 轮 lr ×0.1 |
| `--bs` / `--lr` | 32 / 1e-3 | |
| `--l1` / `--l2` / `--l3` | 0.1 / 1.0 / 1e-4 | coarse / fine / MS-SSIM 权重 |
| `--afm-mode` | softmax | 6 种融合变体 |
| `--gcn-mode` | dynamic | dynamic / static / none |
| `--no-residual` | 关 | 去残差消融 |
| `--arch` | mffgcn | mffgcn / single |
| `--k` | 16 | 图邻居数 |
| `--smoke` | 0 | >0 用小样本快速验证 |
| `--fixed-obs` | 关 | 关闭"每 epoch 重抽观测点" |

## 输出目录 `runs/<tag>/`

| 文件 | 内容 |
|---|---|
| `train_log.csv` | 每 epoch：lr, loss, mse_coarse, mse_fine, val_nmse, val_rmse, sec |
| `best.pt` | val NMSE 最低的权重，**含 args**（评估时用来重建结构） |
| `last.pt` | 最后一轮，断点续训用 |
| `test_metrics.json` | 最终测试指标，**下游所有分析都读它** |

## 两个容易忽略的设计

1. **每个 epoch 重抽观测点**（`set_epoch`）—— 这是数据增强，逼模型学"位置无关"
   的传播规律。`--fixed-obs` 关掉它，用于公平对比。
2. **测试用 best.pt 而不是最后一轮权重** —— 否则和"按 val 选模型"的协议不一致。

## 冒烟测试（改代码后先跑）

```bash
python 03_train/train.py --cache ../data/cache --setup s1 \
    --epochs 2 --smoke 1200 --bs 16 --workers 4 --val-batches 8 --tag debug
```
