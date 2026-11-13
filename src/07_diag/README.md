# 07_diag — 诊断层

排查问题、做补充分析用。**正式流程不跑这些**，但出了问题第一个来这里。

## 文件分类

### 排查"为什么指标不对"

| 文件 | 查什么 |
|---|---|
| `selftest.py` | 最小自检：损失能不能跑通、反向有没有梯度、参数有没有 NaN |
| `viz_check.py` | 把一批样本画出来，肉眼确认 obs/target/env 对齐 |
| `check_s45.py` | 专门检查 Setup 4/5 的采样密度实现对不对 |
| `oos_check.py` | 分布外检查：测试集是否真的没在训练时见过 |

### 补充分析（写论文用）

| 文件 | 分析内容 |
|---|---|
| `diag_gap.py` | 泛化 gap：train NMSE vs test NMSE 的差距 |
| `diag_bands.py` | 盲区宽度对精度的影响（s3 的连续分析） |
| `diag_depth.py` | 观测点数 / 网络深度的影响 |
| `diag_obs.py` | 不同观测策略（均匀/聚簇/贴边）的影响 |
| `analyze_samples.py` | 逐样本误差排序，找最差/最好的样本 |
| `quali.py` | 挑选有代表性的定性样本 |
| `complexity.py` | 参数量 / FLOPs / 推理耗时（写"效率"一节用） |

## 常用

```bash
# 改完模型先跑这个
python 07_diag/selftest.py

# 确认数据管线没歪
python 07_diag/viz_check.py --cache ../data/cache --setup s1

# 查泛化 gap（判断是过拟合还是欠拟合）
python 07_diag/diag_gap.py --cache ../data/cache --tag s1_seed0

# 参数量 / 速度
python 07_diag/complexity.py
```

## diag_gap.py 的判别逻辑

```
train NMSE 低 + test NMSE 高  → 过拟合，或测试集有分布差异
train NMSE 高 + test NMSE 高  → 欠拟合 / 训练不充分 / 学习率不对
train ≈ test 但都高           → 数据管线或标签有问题（最危险）
```

第三种情况最难发现 —— 这时应该去 `01_data/dataset.py` 里核对
`tx_coords` 索引和 `split` 划分，而不是调模型。
