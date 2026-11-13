"""Loss functions (eq. 10-15) and evaluation metrics."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# MS-SSIM (self-contained, no extra dependency)
# --------------------------------------------------------------------------
def _gaussian_window(size=11, sigma=1.5, channels=1, device=None, dtype=None):
    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = g[:, None] * g[None, :]
    return w.expand(channels, 1, size, size).contiguous()


def _ssim_components(x, y, window):
    c = x.shape[1]
    pad = window.shape[-1] // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=c)
    mu_y = F.conv2d(y, window, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sig_x2 = F.conv2d(x * x, window, padding=pad, groups=c) - mu_x2
    sig_y2 = F.conv2d(y * y, window, padding=pad, groups=c) - mu_y2
    sig_xy = F.conv2d(x * y, window, padding=pad, groups=c) - mu_xy

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    lum = (2 * mu_x * mu_y + c1) / (mu_x2 + mu_y2 + c1)
    cs = (2 * sig_xy + c2) / (sig_x2 + sig_y2 + c2)
    return lum, cs


def ms_ssim(x, y, levels=5, weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
    """
    Multi-scale SSIM.  x, y in [0, 1] with shape (B, 1, H, W).
    Returns mean MS-SSIM in [0, 1].
    """
    window = _gaussian_window(11, 1.5, x.shape[1], x.device, x.dtype)
    vals = []
    for i in range(levels):
        lum, cs = _ssim_components(x, y, window)
        if i < levels - 1:
            # only the contrast-structure term is used at coarse scales
            vals.append((cs.mean(dim=(1, 2, 3)), False))
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
        else:
            vals.append((lum.mean(dim=(1, 2, 3)) * cs.mean(dim=(1, 2, 3)), True))

    out = torch.ones_like(vals[0][0])
    for i, (v, is_last) in enumerate(vals):
        w = weights[i]
        term = ((v + 1e-8).clamp(min=1e-8) ** w)
        out = out * term
        if is_last:
            pass
    return out.mean()


class MFFGCNLoss(nn.Module):
    """L = l1 * L_coarse + l2 * L_refine + l3 * (1 - MS-SSIM)"""

    def __init__(self, l1=0.1, l2=1.0, l3=1e-4, use_msssim=True):
        super().__init__()
        self.l1, self.l2, self.l3 = l1, l2, l3
        self.use_msssim = use_msssim

    def forward(self, coarse, fine, target):
        l_c = F.mse_loss(coarse, target)
        l_r = F.mse_loss(fine, target)
        total = self.l1 * l_c + self.l2 * l_r
        l_ms = torch.zeros((), device=target.device)
        if self.use_msssim and self.l3 > 0:
            l_ms = 1.0 - ms_ssim(fine.clamp(0, 1), target.clamp(0, 1))
            total = total + self.l3 * l_ms
        return total, {
            "loss": total.detach(),
            "mse_coarse": l_c.detach(),
            "mse_fine": l_r.detach(),
            "ms_ssim_loss": l_ms.detach() if torch.is_tensor(l_ms) else l_ms,
        }


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
@torch.no_grad()
def nmse(pred, target, eps=1e-8):
    """Normalised MSE, averaged over the batch: ||y-y'||^2 / ||y||^2."""
    b = pred.shape[0]
    num = (pred - target).flatten(1).pow(2).sum(1)
    den = target.flatten(1).pow(2).sum(1) + eps
    return (num / den).mean().item()


@torch.no_grad()
def rmse(pred, target):
    b = pred.shape[0]
    return (pred - target).flatten(1).pow(2).mean(1).sqrt().mean().item()


class MetricAccumulator:
    """Accumulate NMSE / RMSE exactly over a dataset (sum of elements)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.se_fine = 0.0
        self.se_coarse = 0.0
        self.sq_t = 0.0
        self.n_elem = 0
        self.n_img = 0

    def update(self, coarse, fine, target):
        self.se_fine += (fine - target).pow(2).sum().item()
        self.se_coarse += (coarse - target).pow(2).sum().item()
        self.sq_t += target.pow(2).sum().item()
        self.n_elem += target.numel()
        self.n_img += target.shape[0]

    def compute(self):
        return {
            "nmse": self.se_fine / max(self.sq_t, 1e-12),
            "rmse": (self.se_fine / max(self.n_elem, 1)) ** 0.5,
            "nmse_coarse": self.se_coarse / max(self.sq_t, 1e-12),
            "rmse_coarse": (self.se_coarse / max(self.n_elem, 1)) ** 0.5,
            "n_images": self.n_img,
        }
