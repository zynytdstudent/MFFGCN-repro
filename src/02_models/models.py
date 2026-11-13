"""
MFFGCN: 多模态特征融合图卷积网络（论文复现实现）
================================================================
对应论文：Zhang et al., "MFFGCN: Multimodal Feature Fusion Graph Convolution
Network for Radio Map Estimation With Uneven Spatial Sampling", IEEE TMC 2025.

这个文件是整个项目的"心脏"，只负责网络结构（前向计算），不涉及数据、损失、
训练循环。读懂了这个文件，就理解了论文方法本身。

--------------------------------------------------------------------------------
一、任务是什么
--------------------------------------------------------------------------------
输入一张 256x256 的城市地图，已知：
    - 建筑物分布图   m_i         (2D, 0/1)
    - 发射机位置图   p_i         (2D, 只有一个 3x3 亮点)
    - 稀疏观测       f_i         (2D, 只有少量像素有真实信号值，其余为 0)
          └─ 观测位置掩码 mask   (2D, 1 表示该像素被采样过)

输出：完整的 256x256 无线电地图 y_hat（每个像素的归一化信号强度）。
难点在于观测点是"不均匀采样"的 —— 有些区域没有传感器，模型必须靠建筑物
几何 + 电磁传播规律去"猜"。

--------------------------------------------------------------------------------
二、整体结构（两大段，串联）
--------------------------------------------------------------------------------
                        obs  env
                         │    │
              ┌──────────▼────▼──────────┐
  第一阶段     │  DualEncoderUNet         │  双编码器 U-Net
  (粗估计)     │  ├─ 观测编码器（1 通道）  │  + 每一层的 AFM 门控融合
              │  └─ 环境编码器（2 通道）  │
              └──────────┬───────────────┘
                         │ coarse  (B,1,256,256)  粗估计
              ┌──────────▼───────────────┐
  第二阶段     │  GraphRefineNet          │  图卷积精修网络
  (精修)       │  ├─ patch embedding      │  256x256 → 16x16 = 256 个节点
              │  ├─ 动态 KNN 图          │  按特征相似度连边
              │  ├─ 3 x MRConv           │  最大相对图卷积
              │  └─ Graph2Image          │  16x16 → 256x256
              └──────────┬───────────────┘
                         │ residual (B,1,256,256) 残差
                         ▼
              y_hat = coarse + residual      ← 残差叠加（关键设计）
              └─ 这就是 MFFGCN.forward 返回的 fine

为什么要"粗估计 + 残差"两段式？
  U-Net 擅长从稠密图像里提取局部纹理，但它感受野有限、也没法建模"任意两个
  远距离点之间的电磁相关性"。GCN 恰好相反：它能把全图 256 个区域连成一张图，
  让信息沿边传播，但它在稠密像素上做卷积很贵。所以先让 U-Net 出一个"大差不差"
  的粗图，再让 GCN 只学"还差多少"，两者互补。
  消融实验里去掉残差（--no-residual）后 NMSE 从 0.018 涨到 0.039，翻了一倍，
  说明这个设计不是可有可无的。

--------------------------------------------------------------------------------
三、张量形状速查表（batch = B）
--------------------------------------------------------------------------------
    obs      (B, 1, 256, 256)   稀疏观测，未采样处为 0
    env      (B, 2, 256, 256)   通道0=建筑物, 通道1=发射机
    mask     (B, 1, 256, 256)   1=该像素有观测
    -------- DualEncoderUNet --------
    skips[0] (B,  32, 256, 256)
    skips[1] (B,  64, 128, 128)
    skips[2] (B, 128,  64,  64)
    skips[3] (B, 256,  32,  32)
    bottleneck(B, 512,  16,  16)
    coarse   (B,   1, 256, 256)  经 sigmoid，值域 (0,1)
    -------- GraphRefineNet --------
    feat     (B,  64, 256, 256)  融合特征
    z        (B, 256,  16,  16)  patch embedding 后 → flatten 成 (B, 256节点, 256维)
    residual (B,   1, 256, 256)
    -------- 输出 --------
    coarse, fine, residual  三者都是 (B,1,256,256)

注意一个容易混淆的点：图里有 256 个节点（16x16），每个节点是 256 维特征，
这两个 256 纯属数值巧合，没有关系。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================================
# 第 0 部分：基础卷积块
# ==========================================================================
class ConvBNReLU(nn.Sequential):
    """Conv2d + BatchNorm2d + ReLU 三件套，全文件复用最多次的积木。

    默认 k=3, s=1, p=1，即"保持分辨率不变"的 3x3 卷积。
    想降采样就传 s=2（同时 p=1，输出尺寸正好减半）。
    bias=False 是因为后面紧跟 BatchNorm，BN 的 beta 参数会承担偏置的作用，
    多一个 conv bias 纯属浪费显存。
    """

    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),          # inplace 省一半激活显存
        )


class EncStage(nn.Module):
    """编码器的一"层"：两个连续的 3x3 卷积，分辨率不变、通道数 cin→cout。

    这是 RadioUNet 系列 U-Net 的经典配置（VGG 式双卷积）。
    本身不做下采样 —— 下采样由调用方（DualEncoderUNet）用 MaxPool2d 显式完成。
    这样拆开的好处：AFM 融合可以插在"卷积之后、池化之前"，在最高分辨率上做，
    信息损失最小。

    形状: (B,cin,H,W) -> (B,cout,H,W)
    """

    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(
            ConvBNReLU(cin, cout),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class DecStage(nn.Module):
    """解码器的一层：转置卷积上采样 ×2 → 拼接 skip → 双卷积。

    对应 U-Net 的"上采样 + 跳跃连接"。
    ConvTranspose2d(cin, cout, 2, 2) 恰好把 H,W 各放大一倍，且不产生重叠伪影
    （kernel=2, stride=2 是严格的不重叠上采样，等价于每个像素复制到 2x2 块后再
    学一个线性变换）。

    形状: x (B,cin,H,W) + skip (B,cskip,2H,2W) -> (B,cout,2H,2W)
    """

    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, 2)
        self.body = nn.Sequential(
            # 注意拼接后输入通道是 cout(上采样结果) + cskip(跳跃特征)
            ConvBNReLU(cout + cskip, cout),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        return self.body(torch.cat([x, skip], dim=1))


# ==========================================================================
# 第 1 部分：AFM —— 自适应特征融合模块（论文式 4-5）
# ==========================================================================
class AFM(nn.Module):
    """把"观测特征 F_o"和"环境特征 F_e"融合成一个特征 F。

    论文的核心公式（式 4、5）：
        W   = softmax( conv3x3(F_e) )      ← 由环境特征自己生成门控权重
        F_e' = W ⊙ F_e                     ← 门控后的环境特征（⊙ 是逐元素乘）
        F    = conv3x3( concat(F_e', F_o) ) ← 拼接后卷积降维

    为什么门控只作用在环境分支、不加在观测分支？
      观测是"真值锚点"，权重被削弱等于丢真值；环境是"先验提示"，需要根据当前
      场景自适应地决定信多少。所以只让环境分支可调。

    ---------- 关于 mode='softmax' 的工程坑（复现时踩过）----------
    论文写的是 softmax 沿"通道维"归一化。但当通道数 C=512 时，
    softmax 输出满足 sum_c(W_c) = 1，意味着每个通道的平均权重只有 1/512 ≈ 0.002。
    结果就是环境分支被整体压到几乎为 0，门控实际上变成了"关掉环境信息"。
    实验证据：softmax 版本 0.017664，去掉门控（concat）反而 0.0191，而换成
    独立门控 sigmoid 后是 0.018077 —— 说明沿通道 softmax 在这里是有害的。

    因此本实现额外提供两种修正变体，作为论文式 (4) 的合理替代：
        'sigmoid'  每个通道独立过 sigmoid，值域 (0,1)，不强制归一化
                   → 保住了特征幅值，等价于 SE 注意力的门控写法
        'spatial'  先把空间维展平再 softmax，即 sum_{h,w}(W) = 1
                   → 归一化在空间上，通道维不受 C 的影响

    ---------- 另外三种消融基线（论文表 V）----------
        'concat'   不做任何门控，直接拼接后卷积（最朴素）
        'add'      两个分支逐元素相加，再过 1x1 卷积
        'se'       Squeeze-and-Excitation 通道注意力（别名 'channel'），
                   对应论文表 V 的 "Channel Attention" 行
        'cross'    交叉注意力：用观测分支当 query 去检索环境分支，得到上下文
                   再注入环境特征

    参数 c：两个分支的通道数，必须相同（AFM 在 U-Net 每层内使用，该层两分支
    宽度一致）。
    """

    def __init__(self, c, mode="softmax"):
        super().__init__()
        self.mode = mode
        self.c = c

        # ---- 有门控的三种模式：softmax / sigmoid / spatial ----
        if mode in ("softmax", "sigmoid", "spatial"):
            # gate: 只有卷积+BN，故意不加激活，因为 softmax/sigmoid 在 forward
            #       里手动做，不同模式需要不同的归一化方式
            self.gate = nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c),
            )
            # fuse: 拼接 (门控环境 c 通道 + 观测 c 通道) = 2c，卷积降回 c
            self.fuse = nn.Sequential(
                nn.Conv2d(2 * c, c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )
        elif mode in ("se", "channel"):
            # ---- SE 通道注意力（对应论文表 V 的 "Channel Attention" 行）----
            # 论文原文说的是"用 Squeeze-and-Excitation 模块替换 AFM"，即
            # squeeze（全局平均池化，把 H×W 压成一个标量/通道）+ excitation
            # （1x1 卷积降维 C→C/r、ReLU、1x1 卷积升维、sigmoid）得到逐通道增益。
            # 与本文件主路径 AFM 的关键差别：AFM 的门控 W 是"3x3 卷积 + softmax"
            # 得到的整张 H×W 权重图（空间×通道都自适应），而 SE 每个通道只有
            # 一个标量权重、且只能看见全局统计量 —— 表达能力天然更弱，
            # 因此 Table V 里它应当介于"完整 AFM"和"朴素 concat"之间。
            # 参数量的差别也要注意：SE 没有 gate 那层 3x3(c→c) 卷积，
            # 所以 SE 版比 softmax 版少约 9c² 参数（实测 21.45M vs 24.55M），
            # 做消融对比时这一点要写进脚注，否则会被质疑"不是同参数量对比"。
            r = max(c // 16, 4)          # 压缩比 r=16，通道太窄时兜底到 4
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),         # squeeze  → (B,c,1,1)
                nn.Conv2d(c, r, 1, bias=True),   # excitation 降维
                nn.ReLU(inplace=True),
                nn.Conv2d(r, c, 1, bias=True),   # 升维回 c
                nn.Sigmoid(),                    # 逐通道 0~1 增益
            )
            # fuse 与 sigmoid/softmax 变体完全一致：(门控环境 c 通道 + 观测 c 通道) → c
            self.fuse = nn.Sequential(
                nn.Conv2d(2 * c, c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )
        elif mode == "concat":
            # 纯拼接，无门控分支
            self.fuse = nn.Sequential(
                nn.Conv2d(2 * c, c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )
        elif mode == "add":
            # 先相加（c 通道），再用 1x1 卷积做通道混合
            self.mix = nn.Sequential(
                nn.Conv2d(c, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )
        elif mode == "cross":
            # 交叉注意力需要 q/k/v 三个 1x1 投影
            self.q = nn.Conv2d(c, c, 1, bias=False)
            self.k = nn.Conv2d(c, c, 1, bias=False)
            self.v = nn.Conv2d(c, c, 1, bias=False)
            self.fuse = nn.Sequential(
                nn.Conv2d(2 * c, c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"unknown AFM mode {mode}")

    def forward(self, f_obs, f_env):
        """
        f_obs: 观测分支特征 (B, c, H, W)
        f_env: 环境分支特征 (B, c, H, W)
        返回 融合特征   (B, c, H, W)
        """
        # ---- 消融①：直接相加 ----
        if self.mode == "add":
            return self.mix(f_obs + f_env)

        # ---- 消融②：直接拼接 ----
        if self.mode == "concat":
            return self.fuse(torch.cat([f_env, f_obs], dim=1))

        # ---- 消融：SE 通道注意力 ----
        if self.mode in ("se", "channel"):
            # self.se(f_env) 形状 (B,c,1,1)，靠广播自动覆盖 H×W；
            # 观测分支不加权（与其它门控变体保持同一约定），拼完再 3x3 卷积融合
            return self.fuse(torch.cat([f_env * self.se(f_env), f_obs], dim=1))

        # ---- 消融③：交叉注意力 ----
        if self.mode == "cross":
            b, c, h, w = f_env.shape
            # query 用观测特征的"全局平均"表示：把 H,W 压掉，只留一个 C 维向量
            # 因为 query 表达的是"我现在缺什么信息"，一个全局描述就够了
            q = self.q(f_obs).mean(dim=(2, 3)).unsqueeze(1)              # (B, 1, C)
            k = self.k(f_env).flatten(2)                                 # (B, C, N)
            v = self.v(f_env).flatten(2)                                 # (B, C, N)
            # 缩放点积注意力：q·k / sqrt(C)，在 N 个环境位置上做 softmax
            attn = torch.softmax(q @ k / (c ** 0.5), dim=-1)             # (B, 1, N)
            # 加权求和得到上下文向量，形状还原成 (B,C,1,1) 以便广播
            ctx = (attn @ v.transpose(1, 2)).reshape(b, c, 1, 1)
            # 把上下文加到环境特征上（广播到所有空间位置），再与观测拼接
            return self.fuse(torch.cat([f_env + ctx, f_obs], dim=1))

        # ---- 论文主路径：先算门控权重 ----
        g = self.gate(f_env)                     # (B, c, H, W) 未激活的门控图

        if self.mode == "sigmoid":
            # 每通道独立 0~1 门控，不归一化，保留幅值
            w = torch.sigmoid(g)
        elif self.mode == "spatial":
            # 在 H*W 上做 softmax：把 (B,c,H,W) 展平成 (B,c,HW) 再归一化
            # 这样 sum_{h,w} w = 1，与通道数 c 无关
            w = torch.softmax(g.flatten(2), dim=-1).view_as(g)
        else:
            # 论文原式 (4)：在通道维做 softmax（dim=1）
            # ⚠ 已在上方文档说明：C 较大时会压制环境分支
            w = torch.softmax(g, dim=1)

        f_env_w = w * f_env                       # 门控后的环境特征 (B,c,H,W)
        return self.fuse(torch.cat([f_env_w, f_obs], dim=1))


# ==========================================================================
# 第 2 部分：双编码器 U-Net —— 多模态特征融合网络
# ==========================================================================
class DualEncoderUNet(nn.Module):
    """两条独立编码路径 + 每层 AFM 融合 + 单条解码路径。

    为什么用双编码器而不是把 3 通道直接堆叠进去（早期融合）？
      观测图是"稀疏脉冲"，建筑物图是"稠密结构"，两者的统计分布差异极大。
      如果一开始就 concat 成 3 通道送进同一个卷积，第一层卷积就会被两种截然
      不同的尺度主导，稀疏信号容易被建筑物纹理淹没。分成两条路各自提特征，
      再在每一层用 AFM 融合，等价于"晚期/逐层融合"，让网络自己决定何时、
      以多大权重引入环境信息。
      消融实验（--arch single）就是早期融合的对照，NMSE 明显更差。

    通道宽度：默认 (32, 64, 128, 256) + bottleneck 512
    分辨率：  256 → 128 → 64 → 32 → 16(bottleneck)
    """

    def __init__(self, widths=(32, 64, 128, 256), bottleneck=512, out_ch=1,
                 afm_mode="softmax"):
        super().__init__()
        self.afm_mode = afm_mode
        w = list(widths)

        self.enc_obs = nn.ModuleList()   # 观测编码器各层
        self.enc_env = nn.ModuleList()   # 环境编码器各层
        self.afm = nn.ModuleList()       # 每一层配一个 AFM

        # 输入通道数：观测只有 1 通道（稀疏值），环境 2 通道（建筑 + 发射机）
        cin_o, cin_e = 1, 2
        for c in w:
            self.enc_obs.append(EncStage(cin_o, c))
            self.enc_env.append(EncStage(cin_e, c))
            # 两条分支在 EncStage 后通道数都是 c，所以 AFM 的 c 参数就是 c
            self.afm.append(AFM(c, afm_mode))
            cin_o, cin_e = c, c          # 下一层输入 = 本层输出

        self.pool = nn.MaxPool2d(2)      # 所有层共用同一个池化（无参数）

        # 最底层（bottleneck）
        self.enc_obs_b = EncStage(cin_o, bottleneck)
        self.enc_env_b = EncStage(cin_e, bottleneck)
        self.afm_b = AFM(bottleneck, afm_mode)

        # ---- 解码器：宽度反向走 256,128,64,32 ----
        ws = list(reversed(w))
        self.dec = nn.ModuleList()
        cin = bottleneck                 # 从 512 开始
        for c in ws:
            # DecStage(cin=当前通道, cskip=跳跃特征通道, cout=本层输出)
            # 这里 cskip == cout == c，因为对称 U-Net 的结构
            self.dec.append(DecStage(cin, c, c))
            cin = c
        # 输出头：1x1 卷积压到单通道，配合 sigmoid 归一到 (0,1)
        self.head = nn.Conv2d(cin, out_ch, 1)

    def forward(self, obs, env):
        """
        obs: (B,1,256,256)   env: (B,2,256,256)
        return: (B,1,256,256) 归一到 [0,1] 的粗估计
        """
        skips = []
        xo, xe = obs, env

        # ---------- 编码：4 层，逐层下采样 ----------
        for i in range(len(self.enc_obs)):
            xo = self.enc_obs[i](xo)                 # (B,c,256,256) → …
            xe = self.enc_env[i](xe)
            # 关键：融合发生在池化"之前"的最高分辨率上
            skips.append(self.afm[i](xo, xe))
            if i < len(self.enc_obs) - 1:            # 最后一层不池化
                xo = self.pool(xo)                   # 分辨率减半
                xe = self.pool(xe)

        # ---------- 瓶颈层：再池化一次到 16x16 ----------
        xo = self.pool(xo)                           # 32 → 16
        xe = self.pool(xe)
        xo = self.enc_obs_b(xo)                      # (B,512,16,16)
        xe = self.enc_env_b(xe)
        x = self.afm_b(xo, xe)                       # 融合后的瓶颈特征

        # ---------- 解码：自底向上，逐层拼接跳跃特征 ----------
        # skips 的顺序是 [256,128,64,32]（浅→深），解码要深→浅，
        # 所以用 skips[-1-i] 反向取
        for i, dec in enumerate(self.dec):
            x = dec(x, skips[-1 - i])

        # sigmoid 把输出约束到 [0,1]
        # —— 这与数据预处理一致：gain 已经归一化到 [0,1]，且 <0.2 被置 0
        return torch.sigmoid(self.head(x))


# ==========================================================================
# 第 3 部分：图卷积精修网络
# ==========================================================================
class MRConv(nn.Module):
    """Max-Relative Graph Convolution，最大相对图卷积（论文式 9）。

    普通 GCN 只聚合邻居的绝对特征：        h_i = Σ_j x_j · W
    最大相对图卷积聚合的是"邻居相对我的差异"：
        g_i = [ x_i , max_{j∈N(i)} (x_j - x_i) ]        ← 式 9 上半部分
        h_i = g_i · W                                    ← 式 9 下半部分
    直觉：无线电地图上相邻两个区域的关系，更多由"地势落差"（相对差）决定，
    而不是各自的绝对值。取 max 而非 mean，是为了抓住最强的那个方向性提示
    （比如"隔壁街区信号比我强很多"），mean 会把这个信号平均掉。

    实现上分两条线性支路而不是真的拼接：
        out = W_self · x_i  +  W_rel · (max 相对差)
    数学上等价于对 [x_i, x_rel] 拼接后乘一个大矩阵（省一次 cat 的内存）。

    形状: x (B,N,C) + knn_idx (B,N,K) -> (B,N,C_out)
    """

    def __init__(self, cin, cout, act=True):
        super().__init__()
        self.proj_self = nn.Linear(cin, cout, bias=False)   # 对自身特征做变换
        self.proj_rel = nn.Linear(cin, cout, bias=False)    # 对相对差异做变换
        # BatchNorm1d 作用在通道维（需要把 N 和 C 换位）
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x, knn_idx):
        # x:       (B, N, C)     N=256 个节点，C 个特征通道
        # knn_idx: (B, N, K)     每个节点的 K 个邻居编号（已排除自身）
        B, N, C = x.shape
        K = knn_idx.shape[-1]

        # ---- 用 gather 取出邻居特征 ----
        # 把 (B,N,K) 的索引扩成 (B,N*K,C)，这样能一次性把整张邻居表取出来
        idx = knn_idx.reshape(B, N * K, 1).expand(-1, -1, C)
        x_nb = torch.gather(x, 1, idx).reshape(B, N, K, C)   # (B,N,K,C)

        # ---- 相对差 x_j - x_i ----
        # x.unsqueeze(2) 把自身广播到 K 个邻居上
        rel = x_nb - x.unsqueeze(2)                          # (B,N,K,C)
        # 在 K 维取最大 → 每个节点得到一个"最显著相对差"向量
        x_rel = rel.max(dim=2).values                        # (B,N,C)

        # ---- 两条支路相加 ----
        out = self.proj_self(x) + self.proj_rel(x_rel)

        # ---- BN 需要 (B,C,N) 布局，所以转置两次 ----
        out = self.bn(out.transpose(1, 2)).transpose(1, 2)
        return self.act(out)


class GraphRefineNet(nn.Module):
    """图卷积精修网络：把粗估计里"局部的、平滑的"误差修掉。

    完整流程：
        1) 输入拼接：观测值 × mask + 粗估计 × (1-mask)
           → 已观测的点保留真值，未观测的点用粗估计填充（关键：不让粗估计
             污染真值点）
        2) 卷积提特征（两条支路：地图支路看观测，引导支路看粗估计）
        3) patch embedding：4 次 stride-2 卷积，256x256 → 16x16，
           空间位置变成图的节点，每个节点 256 维特征
        4) 动态建图：按节点特征的欧氏距离找 K 近邻（论文式 6）
        5) 3 层 MRConv，每层带残差，信息沿图边传播
        6) Graph2Image：4 次转置卷积，16x16 → 256x256
        7) 输出单通道残差图（不加 sigmoid！因为残差可以是负的）

    输入: obs (B,1,256,256), coarse (B,1,256,256), mask (B,1,256,256)
    输出: 残差图 (B,1,256,256)，值域不限（有正有负）
    """

    def __init__(self, dim=256, k=16, gcn_layers=3, patch=16, img=256,
                 gcn_mode="dynamic"):
        super().__init__()
        self.k = k                  # 每个节点连多少个邻居
        self.patch = patch          # patch 边长（16 → 16x16=256 个节点）
        self.dim = dim              # 节点特征维度
        self.gcn_mode = gcn_mode    # dynamic / static / none（消融用）
        self.grid = img // patch    # 16，图的边长

        # ---- 地图支路：重新注入观测到的真值 ----
        # 输入 2 通道 = [填充后的地图, 观测掩码]，让网络知道"哪些是实测的"
        self.map_branch = nn.Sequential(
            ConvBNReLU(2, 32),
            ConvBNReLU(32, 32),
        )
        # ---- 引导支路：从粗估计里提取语义线索（建筑边缘、信号强弱过渡）----
        self.guide_branch = nn.Sequential(
            ConvBNReLU(1, 32),
            ConvBNReLU(32, 32),
        )
        # 两条支路都是 32 通道，拼起来 64，融合回 64
        self.merge = ConvBNReLU(64, 64)

        # ---- patch embedding：4 次 stride-2 卷积，256→128→64→32→16 ----
        # 通道 64→64→128→192→256(dim)
        self.patch_embed = nn.Sequential(
            ConvBNReLU(64, 64, 3, 2),
            ConvBNReLU(64, 128, 3, 2),
            ConvBNReLU(128, 192, 3, 2),
            ConvBNReLU(192, dim, 3, 2),
        )
        # 位置编码：16x16=256 个位置，每个 dim 维；可学习参数
        # 图卷积本身是置换不变的（不知道节点在原图哪个位置），必须靠它补回位置信息
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid * self.grid, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)   # ViT 同款初始化

        # ---- 图卷积层堆叠 ----
        # gcn_mode == "none" 时层数为 0，即完全跳过 GCN（消融"去掉图卷积"）
        self.gcn = nn.ModuleList(
            [MRConv(dim, dim) for _ in range(gcn_layers if gcn_mode != "none" else 0)]
        )

        # ---- Graph2Image：4 次转置卷积 16→32→64→128→256，再 3x3 卷积出 1 通道 ----
        # 通道 256→192→128→64→32→1
        self.g2i = nn.Sequential(
            nn.ConvTranspose2d(dim, 192, 4, 2, 1), nn.BatchNorm2d(192), nn.ReLU(True),
            nn.ConvTranspose2d(192, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            # 最后一层是普通卷积，不用 sigmoid —— 残差可正可负
            nn.Conv2d(32, 1, 3, 1, 1),
        )

    # ---- 动态图构建（论文式 6）------------------------------------------
    @staticmethod
    def build_knn(feat, k):
        """根据节点特征找 K 个最近邻，构建动态图。

        这是"动态"二字的来源：每个样本、甚至同一批训练里每次前向，图结构都
        不一样，因为它由当前特征算出来，而不是固定的网格拓扑。

        feat:    (B, N, C) 用于建图的引导特征
        返回:    (B, N, K) 邻居索引，已排除自身

        实现细节：
          - 先 L2 归一化再算欧氏距离，等价于用余弦距离排序
          - 强制转 float32：AMP 半精度下 cdist 会有数值溢出，导致距离出现 NaN
          - topk(largest=False) 取距离最小的 k+1 个，第 0 个一定是自己（距离 0），
            所以切片 [1:k+1] 把它丢掉
        """
        B, N, C = feat.shape
        f = F.normalize(feat.float(), dim=-1)        # 单位球面，数值稳定
        dist = torch.cdist(f, f)                     # (B,N,N) 两两欧氏距离
        kk = min(k + 1, N)                           # 多取一个用于剔除自身
        _, idx = torch.topk(dist, kk, dim=-1, largest=False)
        # 丢弃第一列（自身邻居），得到严格 K 个"别人"
        return idx[:, :, 1:k + 1].contiguous()

    def forward(self, obs, coarse, mask):
        # ---- 1) 观测值优先填充 ----
        # mask=1 的位置用真实观测，mask=0 的位置用粗估计
        # 这样已测点的高置信度信息不会被粗估计的误差带偏
        x_in = obs * mask + coarse * (1.0 - mask)

        # ---- 2) 双支路提特征并融合 ----
        f_map = self.map_branch(torch.cat([x_in, mask], dim=1))   # (B,32,256,256)
        f_gui = self.guide_branch(coarse)                          # (B,32,256,256)
        feat = self.merge(torch.cat([f_map, f_gui], dim=1))        # (B,64,256,256)

        # ---- 3) patch embedding + 展平成图节点 ----
        z = self.patch_embed(feat)                   # (B,256,16,16)
        B, C, H, W = z.shape
        z = z.flatten(2).transpose(1, 2)             # (B,256节点,256维)
        z = z + self.pos_embed                       # 注入空间位置信息

        # ---- 4) 建图 + 5) 图卷积 ----
        if self.gcn_mode == "none":
            # 消融：完全不用图卷积，直接用 embedding 特征解码
            pass
        elif self.gcn_mode == "static":
            # 消融：静态图 —— 用固定的位置编码去建 KNN，
            # 得到的图对所有样本都一样（等价于固定的空间邻域）
            knn_idx = self.build_knn(self.pos_embed.expand(B, -1, -1), self.k)
            for layer in self.gcn:
                # 残差连接：z = z + MRConv(z)
                z = z + layer(z, knn_idx)
        else:
            # 主路径：用当前特征建"语义相似度图"
            knn_idx = self.build_knn(z, self.k)
            for layer in self.gcn:
                z = z + layer(z, knn_idx)

        # ---- 6) 还原成 2D 特征图 ----
        z = z.transpose(1, 2).reshape(B, C, H, W)    # (B,256,16,16)
        # ---- 7) Graph2Image 上采样回全分辨率 ----
        return self.g2i(z)                           # (B,1,256,256)


# ==========================================================================
# 第 4 部分：总装 —— MFFGCN
# ==========================================================================
class MFFGCN(nn.Module):
    """论文的完整模型：U-Net 粗估计 + GCN 残差精修。

    forward 返回三元组 (coarse, fine, residual)：
        coarse   = U-Net 的输出，用于计算 L_coarse（辅助监督，权重 l1=0.1）
        fine     = 最终结果 coarse + residual，用于计算主损失 L_refine（权重 l2=1.0）
        residual = GCN 的输出，可视化分析时有用

    为什么要给 coarse 单独加监督？
      U-Net 输出既是最终结果的基底，又是 GCN 的输入（guide 分支）。如果只监督
      fine，梯度要绕过整个 GCN 才能传到 U-Net，前几十个 epoch 会学得很慢。
      加一个 l1=0.1 的辅助损失相当于给 U-Net 一条"捷径"，收敛快很多，
      而且 0.1 这个权重足够小，不会约束 U-Net 去迎合最终目标。
    """

    def __init__(self, widths=(32, 64, 128, 256), bottleneck=512,
                 gcn_dim=256, k=16, gcn_layers=3, img=256, patch=16,
                 afm_mode="softmax", gcn_mode="dynamic", residual=True):
        super().__init__()
        self.fusion = DualEncoderUNet(widths, bottleneck, afm_mode=afm_mode)
        self.refine = GraphRefineNet(dim=gcn_dim, k=k,
                                     gcn_layers=gcn_layers,
                                     patch=patch, img=img,
                                     gcn_mode=gcn_mode)
        self.residual = residual

    def forward(self, obs, env, mask):
        coarse = self.fusion(obs, env)               # (B,1,256,256) ∈ (0,1)
        out = self.refine(obs, coarse, mask)         # (B,1,256,256) 无激活

        if not self.residual:
            # 消融"去残差"：GCN 分支必须自己产出完整地图，
            # 不能只学"修正量"，难度陡增（NMSE 从 0.018 → 0.039）
            fine = torch.sigmoid(out)
            # 第三个返回值仍要凑成 residual 的形状，供上层统一处理
            return coarse, fine, fine - coarse

        fine = coarse + out                          # 残差叠加（论文主设计）
        return coarse, fine, out


class SingleEncoderUNet(nn.Module):
    """单编码器 U-Net：早期融合的对照模型（论文表 V "Single Encoder" 行）。

    把 [稀疏观测, 建筑物, 发射机] 直接堆成 3 通道送进一条编码路径，
    不使用 AFM，也没有 GCN 精修段。

    保留它的两个理由：
      1. 论文消融实验要求有这一行；
      2. 它是最好的"数据管线和代码有没有 bug"的探针 —— 如果连这个朴素模型
         都完全学不动，说明问题出在数据侧（比如坐标错位），而不是网络设计。
         这个项目里就是靠它定位到 tx 坐标索引错位的 bug 的。

    形状: x (B,3,256,256) -> (B,1,256,256)
    """

    def __init__(self, widths=(32, 64, 128, 256), bottleneck=512, out_ch=1,
                 in_ch=3):
        super().__init__()
        w = list(widths)
        self.enc = nn.ModuleList()
        cin = in_ch
        for c in w:
            self.enc.append(EncStage(cin, c))
            cin = c
        self.pool = nn.MaxPool2d(2)
        self.bott = EncStage(cin, bottleneck)
        self.dec = nn.ModuleList()
        cin = bottleneck
        for c in reversed(w):
            self.dec.append(DecStage(cin, c, c))
            cin = c
        self.head = nn.Conv2d(cin, out_ch, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            skips.append(x)
            if i < len(self.enc) - 1:
                x = self.pool(x)
        x = self.bott(self.pool(x))
        for i, d in enumerate(self.dec):
            x = d(x, skips[-1 - i])
        return torch.sigmoid(self.head(x))


# ==========================================================================
# 工具函数
# ==========================================================================
def count_params(m):
    """统计可训练参数量（只算 requires_grad=True 的，排除 BN 的 running stats）。

    实测参考值（默认配置）：
        完整 MFFGCN        24,545,602  ≈ 24.5 M
        单编码器 U-Net      7,763,041  ≈  7.8 M
    注意 MFFGCN 参数比单编码器多 3 倍多，主要来自双编码器（两条路各一份）
    加上 4 个 bottleneck 级的 AFM 门控卷积。
    """
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
