"""
损失函数（论文式 10-15）与评估指标
================================================================
本文件两部分职责：
    1) 定义训练用的组合损失 MFFGCNLoss（含自实现的 MS-SSIM）
    2) 定义评估用的 NMSE / RMSE 及其正确累加方式

--------------------------------------------------------------------------------
一、总损失
--------------------------------------------------------------------------------
    L = λ1 · MSE(y_coarse, y)                 ← 辅助监督，λ1 = 0.1
      + λ2 · MSE(y_fine,   y)                 ← 主监督，  λ2 = 1.0
      + λ3 · (1 - MS-SSIM(y_fine, y))         ← 结构相似度，λ3 = 1e-4

为什么既有 MSE 又有 MS-SSIM？
    MSE 逐像素算误差，对"整体偏亮/偏暗"敏感，但会奖励"糊成一片平均值"的
    解（因为模糊能把极端误差摊平）。MS-SSIM 看的是多尺度的结构相似性，
    能惩罚"边缘被磨平、纹理丢失"，逼模型保留建筑物的绕射阴影这类高频结构。
    两者互补，所以论文把 MS-SSIM 作为正则项，权重取到 1e-4 这么小
    —— 它的量级是 [0,2]（1 - 相似度），乘 1e-4 后只是轻微牵引，不抢夺主导权。

为什么 coarse 也要监督？
    见 models.py 里 MFFGCN 的说明：U-Net 的输出既是最终结果的底，也是 GCN 的
    输入，单独给它 0.1 的权重相当于开了一条梯度捷径，收敛更快。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================================
# 第一部分：MS-SSIM（多尺度结构相似度）
# ==========================================================================
# 没有依赖 pytorch-msssim / skimage，纯手写，避免集群上装包失败。
# 实现遵循 Wang et al. 2003 的原始定义，5 个尺度，权重取自论文。
# --------------------------------------------------------------------------
def _gaussian_window(size=11, sigma=1.5, channels=1, device=None, dtype=None):
    """生成 11x11 的可分离高斯核，作为 SSIM 的滑动窗口。

    先算一维核 g，再用外积 g[:,None] * g[None,:] 得到二维核 —— 高斯核可分离，
    这样构造是 O(n) 而不是 O(n²)。
    expand 到 (channels, 1, size, size) 是为了配 groups=channels 的深度卷积，
    每个通道独立做滤波（单通道图时就是普通卷积）。

    sigma=1.5、size=11 是 SSIM 的经典默认值，不要随便改，
    否则算出的数值和论文/其他实现不可比。
    """
    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()                       # 归一化，保证滤波后亮度不变
    w = g[:, None] * g[None, :]           # 外积 → 二维高斯核
    return w.expand(channels, 1, size, size).contiguous()


def _ssim_components(x, y, window):
    """在单一尺度上算 SSIM 的"亮度项"和"对比度-结构项"。

    SSIM(x,y) = [ l(x,y)^α · c(x,y)^β · s(x,y)^γ ]
    工程实现里把 c 和 s 合并成一个 cs 项（因为两者都依赖协方差），
    于是：  SSIM = lum · cs

    公式（逐像素滑动窗口内统计）：
        μx, μy      窗口内均值（用高斯核卷积得到）
        σx², σy²    窗口内方差
        σxy         窗口内协方差
        lum = (2μxμy + C1) / (μx² + μy² + C1)
        cs  = (2σxy + C2) / (σx² + σy² + C2)
        C1 = (0.01·L)², C2 = (0.03·L)²，L 是动态范围，输入归一化到 [0,1] 故 L=1

    实现技巧：方差用 E[x²] - E[x]² 算，即 conv(x*x) - μ²，
    这样只需 3 次卷积（x、y、x²、y²、xy 共 5 次，但可共用 μ），
    不必用 unfold 拼窗口，显存友好。
    """
    c = x.shape[1]
    pad = window.shape[-1] // 2           # 11 → 5，保持输出尺寸不变
    mu_x = F.conv2d(x, window, padding=pad, groups=c)
    mu_y = F.conv2d(y, window, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    # 方差 = E[x²] - E[x]²
    sig_x2 = F.conv2d(x * x, window, padding=pad, groups=c) - mu_x2
    sig_y2 = F.conv2d(y * y, window, padding=pad, groups=c) - mu_y2
    # 协方差 = E[xy] - E[x]E[y]
    sig_xy = F.conv2d(x * y, window, padding=pad, groups=c) - mu_xy

    c1, c2 = 0.01 ** 2, 0.03 ** 2         # 防止分母为 0 的稳定常数
    lum = (2 * mu_x * mu_y + c1) / (mu_x2 + mu_y2 + c1)
    cs = (2 * sig_xy + c2) / (sig_x2 + sig_y2 + c2)
    return lum, cs


def ms_ssim(x, y, levels=5, weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
    """多尺度 SSIM。x, y 形状 (B,1,H,W)，值域 [0,1]，返回标量均值 ∈ [0,1]。

    为什么要多尺度？
      单尺度 SSIM 只在一个窗口大小上看结构。无线电地图的关键结构有粗有细：
      大片阴影是粗结构，建筑物边缘的绕射条纹是细结构。多尺度做法是
      反复 2x 平均池化下采样，在 5 个分辨率上分别算，再按权重加权几何平均。

    标准公式（Wang 2003）：
        MS-SSIM = [ lum_M ]^{w_M} · Π_{i=1..M} [ cs_i ]^{w_i}
    即"亮度只在最粗尺度算一次，对比度-结构在每个尺度都算"。

    代码里最后一层存的是 (lum·cs)，前 4 层只存 cs，然后全部连乘取幂：
        cs0^w0 · cs1^w1 · cs2^w2 · cs3^w3 · (lum4·cs4)^w4
      = cs0^w0 · cs1^w1 · cs2^w2 · cs3^w3 · cs4^w4 · lum4^w4
    与标准公式完全等价（指数可分配到乘积上），只是少写一个分支。

    256 输入下 5 层后的分辨率：256→128→64→32→16，11x11 窗口仍然放得下。
    """
    window = _gaussian_window(11, 1.5, x.shape[1], x.device, x.dtype)
    vals = []
    for i in range(levels):
        lum, cs = _ssim_components(x, y, window)
        if i < levels - 1:
            # 非最粗尺度：只保留对比度-结构项，然后下采样继续
            # 对 (B,C,H,W) 的 B、C、H、W 全部求均值 → 每个 batch 元素一个标量
            vals.append((cs.mean(dim=(1, 2, 3)), False))
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
        else:
            # 最粗尺度：亮度项在这里才用上
            vals.append((lum.mean(dim=(1, 2, 3)) * cs.mean(dim=(1, 2, 3)), True))

    out = torch.ones_like(vals[0][0])     # 从全 1 开始连乘
    for i, (v, is_last) in enumerate(vals):
        w = weights[i]
        # clamp + 1e-8：负数或 0 取非整数次幂会得到 NaN，
        # 而 SSIM 在极端情况下确实可能算出接近 0 甚至略负的值
        term = ((v + 1e-8).clamp(min=1e-8) ** w)
        out = out * term
        if is_last:
            pass                          # 保留分支以便将来做尺度加权调试
    return out.mean()


class MFFGCNLoss(nn.Module):
    """MFFGCN 的组合损失。

    λ1 / λ2 / λ3 的默认值 0.1 / 1.0 / 1e-4 与论文一致，
    改动前建议先跑 --l1/--l2/--l3 的敏感性对比，不要凭感觉调。
    """

    def __init__(self, l1=0.1, l2=1.0, l3=1e-4, use_msssim=True):
        super().__init__()
        self.l1, self.l2, self.l3 = l1, l2, l3
        self.use_msssim = use_msssim

    def forward(self, coarse, fine, target):
        """
        coarse: U-Net 粗估计 (B,1,256,256)，已过 sigmoid
        fine  : 残差叠加后的最终结果 (B,1,256,256)
                ⚠ 注意 fine 本身没有经过 clamp/sigmoid，值可能略微越界，
                  所以算 MS-SSIM 前要先 clamp 到 [0,1]（下面第 3 项做了这件事）
        target: 真值 (B,1,256,256) ∈ [0,1]
        """
        l_c = F.mse_loss(coarse, target)        # 粗估计 MSE
        l_r = F.mse_loss(fine, target)          # 精修结果 MSE（主损失）

        total = self.l1 * l_c + self.l2 * l_r

        l_ms = torch.zeros((), device=target.device)
        if self.use_msssim and self.l3 > 0:
            # 1 - 相似度：越相似损失越小，取值范围 [0, 2]
            l_ms = 1.0 - ms_ssim(fine.clamp(0, 1), target.clamp(0, 1))
            total = total + self.l3 * l_ms

        # 返回的 parts 全部 detach，只用于打日志，不参与反传
        return total, {
            "loss": total.detach(),
            "mse_coarse": l_c.detach(),
            "mse_fine": l_r.detach(),
            "ms_ssim_loss": l_ms.detach() if torch.is_tensor(l_ms) else l_ms,
        }


# ==========================================================================
# 第二部分：评估指标
# ==========================================================================
# 论文报告的主指标是 NMSE（归一化均方误差），越小越好。
#   NMSE = ‖y_hat − y‖² / ‖y‖²
# 分母用真值的能量做归一化，好处是不同城市/不同发射机功率的地图可以直接平均。
#
# ⚠ 一个必须注意的口径问题：
#   下面 nmse()/rmse() 是"逐样本算比值再平均"（macro 平均），
#   而 MetricAccumulator 是"全数据集累加分子分母再相除"（micro 平均）。
#   两者数值会有细小的差别。项目里所有正式数字一律用 MetricAccumulator，
#   因为它可以跨 batch 精确累加，不受 batch 大小影响。
# --------------------------------------------------------------------------
@torch.no_grad()
def nmse(pred, target, eps=1e-8):
    """逐样本 NMSE 再取平均（macro）。仅用于快速查看，正式指标用累加器。"""
    b = pred.shape[0]
    num = (pred - target).flatten(1).pow(2).sum(1)      # 每个样本的分子
    den = target.flatten(1).pow(2).sum(1) + eps         # 每个样本的分母
    return (num / den).mean().item()


@torch.no_grad()
def rmse(pred, target):
    """逐样本 RMSE 再取平均。"""
    b = pred.shape[0]
    return (pred - target).flatten(1).pow(2).mean(1).sqrt().mean().item()


class MetricAccumulator:
    """在整个数据集上精确累加 NMSE / RMSE（micro 口径）。

    为什么不逐 batch 求平均就完事？
      因为最后一个小 batch 可能不满（比如 8080 样本、batch=32，
      最后一个 batch 只有 16 条），直接平均会让它被赋予和其他满 batch 相同的
      权重，产生偏差。累加分子和分母再一次性相除，则完全与切分方式无关。

    维护 5 个累加量：
        se_fine    精修结果与真值的平方误差总和
        se_coarse  粗估计与真值的平方误差总和
        sq_t       真值的平方和（NMSE 的分母）
        n_elem     累计元素个数（RMSE 的分母）
        n_img      累计图片张数
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.se_fine = 0.0
        self.se_coarse = 0.0
        self.sq_t = 0.0
        self.n_elem = 0
        self.n_img = 0

    def update(self, coarse, fine, target):
        # .item() 把 GPU 标量搬回 CPU 的 Python float，避免长期持有计算图
        self.se_fine += (fine - target).pow(2).sum().item()
        self.se_coarse += (coarse - target).pow(2).sum().item()
        self.sq_t += target.pow(2).sum().item()
        self.n_elem += target.numel()
        self.n_img += target.shape[0]

    def compute(self):
        # max(..., 1e-12) 防止某张图全黑导致分母为 0
        return {
            "nmse": self.se_fine / max(self.sq_t, 1e-12),
            "rmse": (self.se_fine / max(self.n_elem, 1)) ** 0.5,
            "nmse_coarse": self.se_coarse / max(self.sq_t, 1e-12),
            "rmse_coarse": (self.se_coarse / max(self.n_elem, 1)) ** 0.5,
            "n_images": self.n_img,
        }
