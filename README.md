# MFFGCN 论文复现

> Zhang et al., *MFFGCN: Multimodal Feature Fusion Graph Convolution Network for
> Radio Map Estimation With Uneven Spatial Sampling*, IEEE TMC 2025.

在 RadioMapSeer 上复现"无线电地图估计"：给定城市建筑物、发射机位置和少量稀疏实测点，
预测完整的 256×256 信号强度图。

**状态：已结项（2026-11-13）**。结论、偏差根因与遗留项见
[`reports/22-项目结项-2026-11-13.md`](reports/22-项目结项-2026-11-13.md)。

**先看 [`代码导读.md`](./代码导读.md)** —— 里面有架构图、阅读路线、公式对照和踩坑记录。

---

## 怎么打开

双击 `MFFGCN.code-workspace`，VSCode 里会有两个根文件夹：

| 文件夹 | 位置 | 定位 |
|---|---|---|
| **① 本地阅读版** | `D:\Projects\MFFGCN` | 按功能分层 + **详细中文批注**，用来看懂代码 |
| **② 服务器执行版** | `zjut148:/public2/czz/ZhangYu/project/mffgcn` | 扁平目录，**训练/评估实际在这里跑** |

⚠ 两份代码**逻辑等价但目录不同**。改代码、跑实验请改服务器那份；本地这份是为了阅读。

---

## 目录结构

```
D:\Projects\MFFGCN\
├── 代码导读.md            ★ 从这里开始看
├── README.md
├── MFFGCN.code-workspace
│
├── src/
│   ├── 01_data/           数据层：prepare_data.py, dataset.py
│   ├── 02_models/         模型层：models.py, losses.py        ★ 核心
│   ├── 03_train/          训练层：train.py, baseline_train.py, rme_gan_train.py, sweep.py
│   ├── 04_eval/           评估层：evaluate.py, compare_all.py, agg_seeds.py …
│   ├── 05_baselines/      基线层：baselines.py, kriging_knn.py
│   ├── 06_figures/        出图层：make_figs.py, make_table.py, plot_*.py
│   ├── 07_diag/           诊断层：diag_*.py, selftest.py, complexity.py …
│   └── 08_scripts/        运维层：调度 / 批量实验 / 监控 / 修复 / 同步 / 报告
│
├── data/                  数据集（本地不存，全部在服务器）
├── runs/                  训练输出：runs/<tag>/{train_log.csv,best.pt,last.pt,test_metrics.json}
├── results/               评估汇总 json + png
├── figs/                  论文插图
└── reports/               各类报告（md / html / docx）
```

每个 `src/0X_*/` 下都有一份 `README.md` 解释该层职责和文件清单。

---

## 核心代码（批注版）

| 文件 | 行数 | 内容 | 批注密度 |
|---|---:|---|---|
| `02_models/models.py` | 417 | 双编码器 U-Net + AFM 门控 + 动态图 + MRConv | ★★★ 逐行 |
| `01_data/dataset.py` | 294 | 5 种采样设定 + 观测点抽取 + 城市划分 | ★★★ 逐行 |
| `03_train/train.py` | 223 | 训练循环、AMP 四步、checkpoint 策略 | ★★★ 逐段 |
| `02_models/losses.py` | 133 | 组合损失、手写 MS-SSIM、NMSE 累加器 | ★★ |
| `04_eval/evaluate.py` | 182 | 分区评估（全图/盲区/已观测）+ 5 联图 | ★★ |
| `01_data/prepare_data.py` | 133 | PNG → npy 缓存、Tx 坐标反推 | ★★ |

批注包含：**张量形状、为什么这么设计、论文哪条公式、踩过什么坑**。
代码逻辑与服务器版本完全一致（参数量 24,545,602 / 7,763,041 已逐项核对）。

---

## 结果速览

**主结果（Setup 1，测试集 NMSE，越小越好）**

| 方法 | NMSE | 说明 |
|---|---:|---|
| **MFFGCN（5 种子）** | **0.01811 ± 0.00072** | 本复现 |
| U-Net 单编码器（4 种子） | 0.02015 ± 0.00010 | 早期融合对照 |
| Kriging 插值 | 0.0899 | 传统方法 |
| KNN 插值 | 0.0908 | 传统方法 |

MFFGCN 比 U-Net 领先约 **10%**（论文报告 17.6%，方向一致）。

**消融（Setup 1）**：完整 0.0181 / 换 sigmoid 门控 0.018077 / 换 spatial 门控 0.018289 /
去残差 0.0392 / 静态图 0.0337 / 无 GCN 0.0184 —— 残差和动态图是贡献最大的两个设计。

详见 `reports/` 下的系列文档。三份 HTML 各有分工：

| 文件 | 内容 |
|---|---|
| `07-与论文趋势一致性对比.html` | 结论级：哪些趋势对得上、哪些被反证 |
| `08-实验图表汇总与分析.html` | 一图一段，7 张核心图 + 论文结论对照 |
| `09-论文实验图逐张对照.html` | **论文 Fig.6–16 共 11 张，左边论文原图、右边我们的复现**，并排看差距 |

---

## 完成度

| 部分 | 状态 |
|---|---|
| 核心模型 + 5 种 Setup 实现 | ✅ |
| 消融实验（表 V 九行） | ✅ 全部跑完 |
| 传统基线（Kriging / KNN） | ✅ |
| 深度基线（U-Net / Pix2pix / RME-GAN / AE / DeepAE） | ✅ 全部跑完 |
| 多种子统计 | ✅ |
| 论文 Fig.6–16 逐图对照 | ✅ 13 张全部产出 |
| 鲁棒性（噪声 / 距离 / 稀疏度） | 🟡 Fig.14 排序反转，机理已定位，未消除 |

原"约 37% / 25%"的进度估计已完成，剩余分歧作为遗留项记录在结项报告中。

---

## 环境

**服务器**
```bash
ssh zjut148
source /public2/czz/ZhangYu/activate_zhangyu.sh
cd /public2/czz/ZhangYu/project/mffgcn
# tmux 里的 python 是 2.7，必须用全路径：
/public2/czz/ZhangYu/miniconda3/envs/zhangyu/bin/python3.10
```

数据：`data/cache`（3.5 GB 缓存）、`data/raw/RadioMapSeer`（3.4 GB 原始包），不用重新下载。

**集群** `ivsn:/public/home/zy_221126030197/work/mffgcn/`（Slurm，node1-6 A100 / node7 V100×8）

服务器与集群之间**没有 NFS、没有互信**，代码靠 scp 手动同步，实验数据不共享。

---

## 常用任务（Ctrl+Shift+P → Tasks: Run Task）

| 任务 | 作用 |
|---|---|
| ⓪ 拉取服务器代码快照 | 拉到 `src/_from_server/`，供与本地批注版 diff |
| ② 查看训练进度 | 双端状态（服务器 + 集群） |
| ③ 三个 Setup 并行训练 | 50 epoch × 3 设置 |
| ④ 评估 + 生成对比表 | 测试集指标与论文对齐 |
| ⑥ 重新生成趋势对比报告 | 更新 `reports/07-*.html` |

手动跑（服务器）：
```bash
python src/train.py --cache data/cache --setup s1 --epochs 50 --decay-epoch 40 \
    --bs 32 --workers 16 --gpu 0 --tag s1_seed0

python src/evaluate.py --ckpt runs/s1_seed0/best.pt --cache data/cache \
    --setup s1 --out eval_out --name s1
```
