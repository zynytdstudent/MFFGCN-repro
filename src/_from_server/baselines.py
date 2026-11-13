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

    def __init__(self, widths, n_down=4, code=64, in_ch=3):
        super().__init__()
        self.widths = list(widths)
        self.n_down = int(n_down)
        self.spatial = max(1, 256 // (2 ** self.n_down))
        self.code = code
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
        x = torch.cat([obs, env], dim=1)
        z = self.encode(x)
        x = self.from_code(z).view(-1, self.widths[-1], self.spatial,
                                   self.spatial)
        for d in self.dec:
            x = d(x)
        if x.shape[-1] != 256:
            x = F.interpolate(x, size=(256, 256), mode="bilinear",
                              align_corners=False)
        return torch.sigmoid(self.head(x))


class AE(AEBase):
    """7 层编解码，4 次下采样（256 -> 16），隐向量 64。"""

    def __init__(self, code=64):
        super().__init__(widths=(32, 48, 64, 96, 128, 192, 256), n_down=4,
                         code=code)


class DeepAE(AEBase):
    """13 层编解码，5 次下采样（256 -> 8），隐向量 64。"""

    def __init__(self, code=64):
        super().__init__(
            widths=(32, 40, 48, 64, 80, 96, 112,
                    160, 192, 224, 256, 288, 320), n_down=5, code=code)


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


def build_baseline(kind):
    k = kind.lower()
    if k == "ae":
        return AE()
    if k == "deepae":
        return DeepAE()
    if k == "radiounet":
        return RadioUNet()
    if k == "pix2pix":
        return Pix2pixGen()
    raise ValueError(f"unknown baseline {kind}")


if __name__ == "__main__":
    for name in ("ae", "deepae", "radiounet", "pix2pix"):
        m = build_baseline(name)
        print(f"{name:10s} {count_params(m)/1e6:8.2f} M")
