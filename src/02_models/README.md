# 02_models — 模型层

**这是全项目的核心。** 只放网络结构和损失函数，不涉及数据、训练循环。

## 文件

| 文件 | 行数 | 内容 |
|---|---:|---|
| `models.py` | 417 | U-Net + AFM + 图卷积，论文方法本体 |
| `losses.py` | 133 | 组合损失、MS-SSIM、NMSE/RMSE 累加器 |

## models.py 结构

```
第 0 部分  基础积木
    ConvBNReLU    Conv+BN+ReLU，复用最多
    EncStage      双 3x3 卷积（编码器一层）
    DecStage      上采样 + skip 拼接（解码器一层）

第 1 部分  AFM 自适应特征融合（论文式 4-5）
    7 种模式：softmax / sigmoid / spatial / concat / add / cross / se(=channel)
              se 是论文表 V 的 "Channel Attention" 行（Squeeze-and-Excitation）

第 2 部分  DualEncoderUNet（第一阶段）
    双编码器 → 每层 AFM 融合 → 单解码器 → sigmoid

第 3 部分  GraphRefineNet（第二阶段）
    MRConv         最大相对图卷积（论文式 9）
    build_knn      动态建图（论文式 6）
    Graph2Image    16x16 → 256x256

第 4 部分  MFFGCN 总装
    fine = coarse + residual
    SingleEncoderUNet  早期融合对照
```

## 参数量（默认配置，实测）

```
MFFGCN        24,545,602  ≈ 24.5 M
MFFGCN(se)    21,445,794  ≈ 21.4 M   ← Table V 的 Channel Attention 变体
SingleUNet     7,763,041  ≈  7.8 M
```

> 注意 `se` 模式比默认的 softmax 门控**少约 3.1 M 参数**：SE 用全局平均池化 +
> 两层 1x1 卷积替代了 AFM 里那层 3x3(c→c) 门控卷积。所以表 V 里
> "Channel Attention ↔ AFM" 的对比不是严格同参数量对比，写论文时要加脚注。

## 张量形状

```
obs (B,1,256,256)  env (B,2,256,256)
  → DualEncoderUNet
      skips[0] (B,32,256,256)
      skips[1] (B,64,128,128)
      skips[2] (B,128,64,64)
      skips[3] (B,256,32,32)
      bottleneck (B,512,16,16)
  → coarse (B,1,256,256)
  → GraphRefineNet
      feat (B,64,256,256)
      z    (B,256节点,256维)      ← 两个 256 是巧合，无关
  → residual (B,1,256,256)
  → fine = coarse + residual
```

## 三个容易搞混的点

1. **残差分支不加 sigmoid** —— 修正量可正可负。只有最终 `fine` 通过
   `coarse` 隐含在 [0,1] 附近。
2. **AFM 的 softmax 是沿通道的** —— C=512 时会把环境分支压死，实验上
   sigmoid 更好。详见 `AFM` 的文档字符串。
3. **`z = z + layer(z, knn)` 是残差** —— 图卷积层外面包了个残差连接。

## 完整解析

每个类、每个关键行都有中文批注，直接看源码。配合根目录 `代码导读.md`
的"五个关键设计决策"一节。
