# 05_baselines — 基线层

论文对比表里的非 MFFGCN 方法。分两类：

## 文件

| 文件 | 内容 |
|---|---|
| `baselines.py` | 深度基线网络定义：RadioUNet / Pix2pix / RME-GAN 的生成器与判别器 |
| `kriging_knn.py` | **传统插值基线**：克里金（Kriging）、K 近邻（KNN） |

## baselines.py

复用 `02_models/models.py` 里的 `ConvBNReLU` / `EncStage` / `DecStage`，
所以它的网络结构和主模型是同一套积木（保证对比公平，参数量可查）。

配合 `03_train/baseline_train.py` 训练。

## kriging_knn.py

不训练，纯数学插值。这是论文里"深度学习方法到底赢在哪"的对照组。

当前结果（三个 Setup）：

| 方法 | s1 | s2 | s3 |
|---|---:|---:|---:|
| KNN | 0.0908 | 0.1049 | 0.1139 |
| Kriging | 0.0899 | 0.1030 | 0.1116 |
| MFFGCN | **0.0181** | — | — |

深度学习版本领先传统插值约 **5 倍**，这是论文最主要的说服力来源。

原始结果文件在 `interp/` 目录：`interp_knn_s{1,2,3}.json`、`interp_kriging_s{1,2,3}.json`。
（服务器上这些文件生成在 `src/` 根目录，脚本默认从当前工作目录读写，本地归档到了这里。）

## 注意

- Kriging 用的是经验变差函数，参数是**手调**的，不是学出来的。如果调得更细，
  基线会更强，对比更公平 —— 这是复现工作里可以改进的点。
- 两个方法都只读 `data/cache`，不依赖任何 checkpoint。
