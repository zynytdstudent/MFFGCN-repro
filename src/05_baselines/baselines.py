"""论文表 III 的深度基线模型。

只放基线，不动 models.py（MFFGCN 本体和消融分支保持原样，避免影响正在跑
的训练）。

包含：
    AE          对称自编码器，7 层编解码，隐向量长度 64
    DeepAE      更深的自编码器，13 层编解码，隐向量长度 64
    RadioUNet   两个 U-Net 串联（U + U = W，故又称 WNet），两阶段训练
    Pix2pixGen  Pix2pix+ 的生成器（U-Net + 跳跃连接）
    PatchGAN    两个 GAN 共用的判别器

所有模型统一签名：forward(obs, env) -> (H, W) 归一化无线电地图，输入
obs(1) + env(2) 共 3 通道。这样训练和评估循环可以共用。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ConvBNReLU, DecStage, EncStage, count_params


# --------------------------------------------------------------------------
# 自编码器
# --------------------------------------------------------------------------
class AEBase(nn.Module):
    """对称卷积自编码器。

    widths  每一层的输出通道数，长度 = 编码器层数
    n_down  其中前 n_down 层做 stride-2 下采样（256 -> 256 / 2**n_down）
    code    隐向量维度

    解码器与编码器层数相同、顺序相反：前 n_down 层用转置卷积上采样，
    其余用普通 3x3 卷积，最后 1x1 卷积回到单通道。
    """

    def __init__(self, widths, n_down=4, code=64, in_ch=3, skip="none"):
        super().__init__()
        self.widths = list(widths)
        self.n_down = int(n_down)
        self.spatial = max(1, 256 // (2 ** self.n_down))
        self.code = code
        self.skip = skip          # none | enc | obs（见 forward 注释）
        n = len(self.widths)

        self.enc = nn.ModuleList()
        cin = in_ch
        for i, c in enumerate(self.widths):
            stride = 2 if i < self.n_down else 1
            self.enc.append(ConvBNReLU(cin, c, 3, stride, 1))
            cin = c
        self.flat_dim = self.widths[-1] * self.spatial * self.spatial
        self.to_code = nn.Linear(self.flat_dim, code)
        self.from_code = nn.Linear(code, self.flat_dim)

        self.dec = nn.ModuleList()
        cin = self.widths[-1]
        for i in range(n - 1, -1, -1):
            cout = self.widths[i - 1] if i > 0 else max(16, self.widths[0] // 2)
            if len(self.dec) < self.n_down:
                self.dec.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(cin, cout, 4, 2, 1),
                        nn.BatchNorm2d(cout),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.dec.append(ConvBNReLU(cin, cout, 3, 1, 1))
            cin = cout
        self.head = nn.Conv2d(cin, 1, 1)

    def encode(self, x):
        for e in self.enc:
            x = e(x)
        return self.to_code(x.flatten(1))

    def forward(self, obs, env):
        """skip 选项来自论文对 AE 的定义：

        "Autoencoder consists of a symmetric encoder and decoder **and a skip
        connection between them**"（results/paper_text.txt 第 772 行）。

        我们最初的实现没有任何 skip，结果 AE 的 NMSE 卡在 0.080、且对观测点数
        完全不敏感（Fig.7 里 AE/DeepAE 两条线是平的，论文却是 0.0625→0.0184）。
        这里给出三种可选口径，用 --ae-skip 选择：

          none : 原实现，纯瓶颈重建
          enc  : 编码器末层特征 + 瓶颈重建（"encoder 到 decoder 的 skip"字面解）
          obs  : 输出残差加回观测图 y = clamp(ŷ + obs)（补全式 AE 的常见做法，
                 观测点越多越准，正好能解释论文那两条陡降曲线）
        """
        x = torch.cat([obs, env], dim=1)
        for e in self.enc:
            x = e(x)
        f = x
        z = self.to_code(f.flatten(1))
        x = self.from_code(z).view(-1, self.widths[-1], self.spatial,
                                   self.spatial)
        if self.skip == "enc":
            x = x + f
        for d in self.dec:
            x = d(x)
        if x.shape[-1] != 256:
            x = F.interpolate(x, size=(256, 256), mode="bilinear",
                              align_corners=False)
        y = torch.sigmoid(self.head(x))
        if self.skip == "obs":
            y = (y + obs).clamp(0.0, 1.0)
        return y


class AE(AEBase):
    """7 层编解码，4 次下采样（256 -> 16），隐向量 64。"""

    def __init__(self, code=64, skip="none"):
        super().__init__(widths=(32, 48, 64, 96, 128, 192, 256), n_down=4,
                         code=code, skip=skip)


class DeepAE(AEBase):
    """13 层编解码，5 次下采样（256 -> 8），隐向量 64。"""

    def __init__(self, code=64, skip="none"):
        super().__init__(
            widths=(32, 40, 48, 64, 80, 96, 112,
                    160, 192, 224, 256, 288, 320), n_down=5, code=code,
            skip=skip)


# --------------------------------------------------------------------------
# RadioUNet（WNet）：两个 U-Net 串联，分两阶段训练
# --------------------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256), bottleneck=512, in_ch=3,
                 out_ch=1):
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


class RadioUNet(nn.Module):
    """第一阶段 U-Net 输出粗图，第二阶段把粗图一起当作输入再细化。

    两阶段独立训练：stage 1 训 50 epoch 后冻结，stage 2 再训 50 epoch。
    forward 里用 self.stage 控制训哪一段。
    """

    def __init__(self, widths=(32, 64, 128, 256), bottleneck=512):
        super().__init__()
        self.net1 = UNet(widths, bottleneck, in_ch=3, out_ch=1)
        self.net2 = UNet(widths, bottleneck, in_ch=4, out_ch=1)
        self.stage = 1

    def forward(self, obs, env):
        x1 = torch.cat([obs, env], dim=1)
        if self.stage == 1:
            y = self.net1(x1)
            return y, y
        with torch.no_grad():
            hint = self.net1(x1)
        y = self.net2(torch.cat([x1, hint], dim=1))
        return hint, y


# --------------------------------------------------------------------------
# cGAN：Pix2pix+ / RME-GAN
# --------------------------------------------------------------------------
class Pix2pixGen(nn.Module):
    """U-Net 生成器，带跳跃连接（Pix2pix 标准结构）。"""

    def __init__(self, widths=(32, 64, 128, 256), bottleneck=512, in_ch=3):
        super().__init__()
        self.unet = UNet(widths, bottleneck, in_ch=in_ch, out_ch=1)

    def forward(self, obs, env):
        return self.unet(torch.cat([obs, env], dim=1))


class PatchGAN(nn.Module):
    """70x70 PatchGAN 判别器，输入 [条件, 地图]。"""

    def __init__(self, in_ch=4, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 4, base * 8, 4, 1, 1), nn.BatchNorm2d(base * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, cond, img):
        return self.net(torch.cat([cond, img], dim=1))


def build_baseline(kind, ae_skip="none", ae_ndown=0):
    """ae_ndown>0 时覆盖 AEBase 的下采样次数（只对 ae/deepae 有效）。

    DeepAE 默认 n_down=5 -> skip 落在 8x8 网格上；改成 4 就是 16x16，
    用来验证 "DeepAE 曲线是平的" 到底是不是 skip 分辨率不够。
    """
    k = kind.lower()
    if k == "ae":
        return AE(skip=ae_skip)
    if k == "deepae" and ae_ndown:
        return AEBase(widths=(32, 40, 48, 64, 80, 96, 112,
                              160, 192, 224, 256, 288, 320),
                      n_down=int(ae_ndown), code=64, skip=ae_skip)
    if k == "deepae":
        return DeepAE(skip=ae_skip)
    if k == "radiounet":
        return RadioUNet()
    if k in ("pix2pix", "unet"):
        # unet：pix2pix 的生成器单独做纯监督 L1 训练（消融用，2026-11-13）
        return Pix2pixGen()
    raise ValueError(f"unknown baseline {kind}")


if __name__ == "__main__":
    for name in ("ae", "deepae", "radiounet", "pix2pix", "unet"):
        m = build_baseline(name)
        print(f"{name:10s} {count_params(m)/1e6:8.2f} M")
