# MFFGCN 代码结构导读

> 目的：能在 VSCode 里按图索骥地读代码，而不是从 36 个 py 文件里乱翻。
> 对应代码位置：**服务器 `zjut148:/public2/czz/ZhangYu/project/mffgcn/src/`**，本地镜像 `D:\Projects\MFFGCN\src\`。

---

## 0. 一句话概括

整个项目只有三件事：**一份数据缓存** → **一个双分支模型** → **一套评估脚本**。

36 个 py 文件里，**只有 5 个是主干**，其余 31 个都是围绕它的实验、评估和画图工具。

---

## 1. 分层：哪些文件是主干，哪些是外围

### 主干（必须读懂，按依赖顺序）

| 顺序 | 文件 | 行数 | 职责 | 类型 |
|---|---|---|---|---|
| ① | `prepare_data.py` | 132 | PNG → npy 缓存，一次性 | 独立脚本 |
| ② | `dataset.py` | 293 | 数据集 + 五种不均匀采样 | **被 import 的库** |
| ③ | `models.py` | 416 | MFFGCN 全部网络结构 | **被 import 的库** |
| ④ | `losses.py` | 132 | 损失函数 + NMSE/RMSE 指标 | **被 import 的库** |
| ⑤ | `train.py` | 222 | 训练主循环 | 独立脚本 |

读懂这 5 个，项目就通了 80%。**建议的阅读顺序就是 ②→③→④→⑤，`prepare_data.py` 最后看**（它只是格式转换，不影响理解模型）。

### 外围：评估与分析

| 文件 | 用途 |
|---|---|
| `evaluate.py` | 单模型评估：全局 / 盲区 / 观测区三套指标 |
| `sweep.py` | 批量重评所有 checkpoint，产出 `eval_sweep_*.jsonl` |
| `eval_all.py` | 早期版本的重评脚本（功能与 sweep 重叠，历史遗留） |
| `reeval.py` | 更细的重评：加盲区距离分带 |
| `agg_report.py` | 汇总成论文表 III / 表 V 对照表 |
| `agg_seeds.py` | 多 seed 取均值±标准差 |

### 外围：诊断（排查问题用，平时不看）

| 文件 | 诊断什么 |
|---|---|
| `diag_obs.py` | 模型到底有没有在用稀疏观测（把 obs 置零看误差变化） |
| `diag_bands.py` | 按地图区间分段统计 RMSE，定位误差跳变 |
| `diag_gap.py` | train/test 差距是真过拟合还是评估假象 |
| `oos_check.py` | 两种观测点放置策略的口径差异量化 |
| `check_s45.py` | 验证 Setup 4/5 的实现是否符合论文 |
| `viz_check.py` | 肉眼看模型输出的是什么 |

### 外围：基线与对比

| 文件 | 基线 |
|---|---|
| `baseline_train.py` + `baselines.py` | AE / DeepAE / RadioUNet / Pix2pix+ / RME-GAN |
| `rme_gan_train.py` | RME-GAN 单独复现（路径损耗预训练 + cGAN 微调） |
| `kriging_knn.py` | 传统插值（Kriging / KNN），不用训练 |
| `complexity.py` | 参数量 / MACs / 延迟 |

### 外围：画图

`make_figs.py`、`quali.py`、`plot_*.py`（6 个）、`make_table.py`。
其中 `plot_compare.py` / `plot_compare2.py` / `plot_paper_vs_ours.py` 功能重叠，是不同时期的版本，看最新的 `make_figs.py` 就够了。

---

## 2. 数据流：张量形状每步怎么变

这是理解代码最省力的入口。所有样本都是 256×256。

### 输入（`dataset.py` 返回的 dict）

| 键 | 形状 | 含义 |
|---|---|---|
| `obs` | `(B, 1, 256, 256)` | 稀疏观测图，只在观测点处有值，其余为 0 |
| `env` | `(B, 2, 256, 256)` | 通道0=建筑图，通道1=发射机位置（3×3 亮块） |
| `mask` | `(B, 1, 256, 256)` | 观测点指示图，1=有观测 |
| `target` | `(B, 1, 256, 256)` | 真值无线电地图，已归一化到 [0,1] |
| `index` | `(B,)` | 全局样本 id，用于回查是哪张图 |

### 第一段：双编码器 U-Net → 粗估计

`DualEncoderUNet.forward(obs, env)`，代码在 `models.py:196`

```
obs  1@256 ─enc_obs[0]→ 32@256 ─┐
                                 ├─ AFM → skip[0] 32@256 ──pool──→ 128
env  2@256 ─enc_env[0]→ 32@256 ─┘

             128 → 64@128  → AFM → skip[1]  64@128  → pool → 64
              64 → 128@64  → AFM → skip[2] 128@64   → pool → 32
              32 → 256@32  → AFM → skip[3] 256@32   ──pool──→ 16

              16 → bottleneck 512@16
```

解码器从瓶颈逐级上采样，每级拼接 skip：

```
512@16 → up → 32 → cat(skip[3] 256ch) → 256@32
256@32 → up → 64 → cat(skip[2] 128ch) → 128@64
128@64 → up →128 → cat(skip[1]  64ch) →  64@128
 64@128→ up →256 → cat(skip[0]  32ch) →  32@256
                                     → 1×1 conv
                                     → sigmoid
                                     → coarse 1@256
```

**关键点**：两个编码器全程平行，只在每级末尾用 AFM 融合一次。融合后的特征既送进 decoder 的 skip，也继续往下采样。

### 第二段：图精化网络 → 残差

`GraphRefineNet.forward(obs, coarse, mask)`，代码在 `models.py:317`

```
先做一次"实测值回填"：
    x_in = obs * mask + coarse * (1 - mask)
    有观测的地方用真观测，没观测的地方用粗估计

map_branch(cat[x_in, mask])  → 32@256   ┐
guide_branch(coarse)         → 32@256   ┴→ merge → 64@256

patch_embed: 4 个 stride-2 卷积
    64@256 → 64@128 → 128@64 → 192@32 → 256@16
    flatten 成 (B, 256, 256)   ← 256 个节点，每个 256 维
    + pos_embed（可学习位置编码）

建图 build_knn(z, k=16)：
    cdist 得到 256×256 距离矩阵
    topk 取最近 17 个，去掉自己 → 16 个邻居
    得到 knn_idx (B, 256, 16)

3 层 MRConv，每层带残差：
    z = z + MRConv(z, knn_idx)

上采样回原尺寸 Graph2Image：
    256@16 → 192@32 → 128@64 → 64@128 → 32@256 → 1@256
    → residual 1@256
```

### 第三段：相加得最终输出

`MFFGCN.forward`，代码在 `models.py:362`

```python
coarse = fusion(obs, env)          # 粗估计
out    = refine(obs, coarse, mask) # 残差
fine   = coarse + out              # 最终结果
return coarse, fine, out
```

三个返回值分别用于：粗估计的监督（`l1` 项）、精化结果的主监督（`l2` 项）、以及诊断时的残差分析。

---

## 3. 三块组件分别在做什么

### 3.1 AFM（`models.py:75`）—— 模型里最容易误读的地方

论文式 (4)(5)：`F_e' = W ⊙ F_e`，`W = softmax(conv3x3(F_e))`，然后与观测分支 concat 再卷积。

代码里 `torch.softmax(g, dim=1)` 是**沿通道维**做 softmax。当通道数 C=512 时，每个通道的平均权重只有 1/C ≈ 0.002，**等于把环境分支直接压没了**。

这就是为什么代码额外提供了 `sigmoid` 和 `spatial` 两种模式：

| `--afm-mode` | 门的归一化方式 | 效果 |
|---|---|---|
| `softmax` | 沿通道，和为 1 | 论文原式，但会衰减环境分支 |
| `sigmoid` | 每个通道独立 0~1 | 保留特征幅值 |
| `spatial` | 沿 H×W，和为 1 | 门保持单位量级 |
| `concat` | 无门，直接拼 | 消融用 |
| `add` | 逐元素相加后 1×1 卷积 | 消融用 |
| `cross` | 观测分支 query 环境分支的交叉注意力 | 消融用 |

实测 softmax 与 sigmoid 差异可以忽略（报告里已写明），但读代码时容易以为是自己理解错了，其实是论文没规定清楚。

### 3.2 MRConv（`models.py:221`）—— 最大相对图卷积

论文式 (9) 的实现，只有一行数学：

```python
rel   = x_nb - x                    # 邻居减自己：x_j - x_i
x_rel = rel.max(dim=2).values        # 取最大 → 只保留最突出的差异
out   = proj_self(x) + proj_rel(x_rel)
```

`max` 而不是 `sum`/`mean`，这是"Max-relative"的含义：只关心邻居中**相对自己变化最剧烈**的那个方向。

### 3.3 动态图 vs 静态图（`models.py:330`）

```python
if self.gcn_mode == "none":      # 消融：完全不要 GCN
    pass
elif self.gcn_mode == "static":  # 消融：用固定的网格位置建图
    knn_idx = self.build_knn(self.pos_embed.expand(B,-1,-1), self.k)
else:                            # 论文：用内容特征建图
    knn_idx = self.build_knn(z, self.k)
```

- **dynamic**：用当前样本的特征 `z` 算相似度，每个样本的图都不一样
- **static**：用可学习的位置编码建图，跟图像内容无关，所有样本共享同一张图

---

## 4. 一次训练的完整调用链

```
train.py:main()
 ├─ build_loaders()                    dataset.py:270
 │    ├─ RadioMapSeer(train)   →  40000 条
 │    ├─ RadioMapSeer(val)     →   8000 条
 │    └─ RadioMapSeer(test)    →   8080 条
 ├─ MFFGCN(...)                        models.py:350
 ├─ MFFGCNLoss(0.1, 1.0, 1e-4)         losses.py:63
 ├─ Adam + MultiStepLR(milestones=[40], gamma=0.1)
 ├─ GradScaler（AMP 混合精度）
 └─ for ep in 1..50:
      ├─ tr.dataset.set_epoch(ep)      ← 关键：重撒观测点
      ├─ for batch in tr:
      │    ├─ fwd() → coarse, fine, _   train.py:24
      │    ├─ crit(coarse, fine, target)
      │    ├─ scaler.scale(loss).backward()
      │    ├─ clip_grad_norm_(5.0)
      │    └─ scaler.step / update
      ├─ evaluate(model, va, max_batches=60)   ← 只取 60 个 batch 打分
      ├─ 若 val NMSE 创新低 → 存 best.pt
      └─ 写 train_log.csv
 └─ 训练结束
      ├─ 存 last.pt（第 50 轮）
      └─ 载入 best.pt，在 test 全量 8080 条上评一次 → test_metrics.json
```

**注意 `evaluate` 的 `max_batches=60`**：验证时只跑了 60/250 个 batch（8000/32≈250），所以 val NMSE 是抽样估计，有 0.5~0.7% 的抖动。这就是为什么 `best.pt` 的选点不稳定。

---

## 5. 消融实验怎么切换（全部靠命令行参数）

| 论文 Table V 的行 | 命令行 |
|---|---|
| 完整 MFFGCN | `--afm-mode softmax --gcn-mode dynamic` |
| w/o GCN | `--gcn-mode none` |
| w/o Dynamic（静态图） | `--gcn-mode static` |
| w/o Residual | `--no-residual` |
| Single Encoder | `--arch single` |
| AFM→Concat | `--afm-mode concat` |
| AFM→Add | `--afm-mode add` |
| AFM→Cross-attn | `--afm-mode cross` |
| AFM→Channel softmax | `--afm-mode softmax`（默认） |
| AFM→Sigmoid | `--afm-mode sigmoid` |
| K 邻居数敏感性 | `--k 4/8/16/32` |

**代码里没有第二套模型实现**，所有消融都是同一个 `MFFGCN` 类换参数。这是读代码时最该注意的一点：不要在 `models.py` 里找"消融版的模型"，它不存在。

`--arch single` 是唯一例外，走 `SingleEncoderUNet`（`models.py:374`），因为它连编码器结构都不同。

---

## 6. 建议的阅读路径

如果你只有 1 小时：

1. **`models.py` 第 350-372 行**（`MFFGCN.forward`）—— 先看清整体是"粗估计 + 残差"两段式，10 行
2. **`models.py` 第 196-215 行**（`DualEncoderUNet.forward`）—— 双编码器怎么并行、AFM 在哪融合
3. **`models.py` 第 317-344 行**（`GraphRefineNet.forward`）—— 图像怎么变成 256 个节点、再变回图像
4. **`dataset.py` 第 161-217 行**（`__getitem__`）—— 观测点怎么撒、五种 setup 的分支
5. **`train.py` 第 150-205 行**（训练循环）—— 损失怎么算、best 怎么选
6. **`losses.py` 第 63-84 行**（损失函数）—— 三项加权

跳读时可以先忽略：所有 `plot_*.py`、所有 `diag_*.py`、`baseline_*.py`、`rme_gan_train.py`。

---

## 7. 在 VSCode 里怎么看

工作区 `MFFGCN.code-workspace` 有两个 folder，**优先用上面那个**：

| Folder | 实际位置 | 是否最新 |
|---|---|---|
| `MFFGCN · 服务器 zjut148` | 服务器的 `/public2/.../mffgcn` | ✅ 永远是真相，改完直接生效 |
| `MFFGCN · 本地副本 D:` | `D:\Projects\MFFGCN` | ⚠️ 需要手动同步 |

**两边不会自动同步**。工作区还开了 `files.autoSave: afterDelay 2000`，意味着你在本地那栏改代码，2 秒后存的是 D 盘，服务器不知道。

需要把服务器最新代码拉到本地时，双击项目根目录的 `sync-from-server.cmd`，或在 VSCode 里跑任务 `① 同步代码 ← 服务器`。
