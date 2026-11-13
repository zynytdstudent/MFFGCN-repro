# -*- coding: utf-8 -*-
"""诊断：DeepAE 的 skip 为什么没把曲线救回来——是「skip 的分辨率」还是「深度」？

背景
----
2026-11-12 给 AE 加了 `skip=enc`（编码器末层特征绕过 64 维 code，直接加回解码器
入口），AE 的 Fig.7 曲线从 0.0840→0.0778 的平线变成 0.0173→0.0126，问题解决。
但同样加了 skip 的 DeepAE（13 层、`n_down=5`）不但没降下来，反而更差：

    DeepAE skip=enc : 0.0545 -> 0.0502 (1.08x)   test 0.0521
    AE     skip=enc : 0.0173 -> 0.0126 (1.37x)   test 0.0147

两者的 skip 唯一的差别是分辨率：AE 的 `n_down=4`，编码器末层特征在 16x16
（256 个格子）；DeepAE 的 `n_down=5`，只有 8x8（**64 个格子**）。而 300 个观测点
要落在这些格子里才能被 skip 送过去，64 格装不下 300 个点（平均每格 4.7 个）。

本脚本不训练，只做推理，用**同一份 AE 权重**把 skip 人为粗化到 8x8，
看误差会不会跳到 DeepAE 的量级：

    enc        : 原样 16x16 skip（= 训练时的口径）
    enc8       : skip 先 avg_pool 到 8x8 再上采样回来（故意丢掉 3/4 的空间信息）
    none       : 关掉 skip

如果 enc8 掉到 ~0.05 且曲线变平，就说明 DeepAE 的瓶颈在 skip 分辨率，
而不是「13 层太深训不动」。

用法（src/ 下）：
    python ae_skip_res.py --gpu 1 --out ../results/ae_skip_res.json
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (".", "..", os.path.join(_HERE, ".."), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import RadioMapSeer          # noqa: E402
from baselines import AE                  # noqa: E402

COUNTS = [50, 100, 150, 200, 250, 300]


class AEProbe(AE):
    """AE + 可选的 skip 粗化。coarse=N 时把 skip 压到 NxN 再升回来。"""

    def __init__(self, mode="enc", coarse=8):
        super().__init__(skip="none" if mode == "none" else "enc")
        self.mode = mode
        self.coarse = coarse

    def forward(self, obs, env):
        x = torch.cat([obs, env], dim=1)
        for e in self.enc:
            x = e(x)
        f = x
        if self.mode == "enc8":
            # 故意把空间信息压到 8x8（DeepAE 的 skip 就是这么大），再升回 16x16
            f = F.interpolate(
                F.avg_pool2d(f, f.shape[-1] // self.coarse),
                size=(self.spatial, self.spatial), mode="nearest")
        z = self.to_code(f.flatten(1))
        x = self.from_code(z).view(-1, self.widths[-1], self.spatial,
                                   self.spatial)
        if self.mode != "none":
            x = x + f
        for d in self.dec:
            x = d(x)
        if x.shape[-1] != 256:
            x = F.interpolate(x, size=(256, 256), mode="bilinear",
                              align_corners=False)
        return torch.sigmoid(self.head(x))


@torch.no_grad()
def evaluate(m, loader, dev):
    se = sq = 0.0
    for b in loader:
        obs = b["obs"].to(dev)
        env = b["env"].to(dev)
        t = b["target"].to(dev).float()
        y = m(obs, env).float().clamp(0, 1)
        se += float((y - t).pow(2).sum())
        sq += float(t.pow(2).sum())
    return se / max(sq, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bl_ae_s1_skipenc")
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--setup", default="s1")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--out", default="../results/ae_skip_res.json")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(os.path.join("runs", a.tag, "best.pt"),
                    map_location="cpu", weights_only=False)
    out = {"tag": a.tag, "setup": a.setup, "counts": COUNTS, "curves": {}}

    for mode in ("enc", "enc8", "none"):
        m = AEProbe(mode).to(dev)
        m.load_state_dict(ck["model"])
        m.eval()
        nm = []
        for k in COUNTS:
            ds = RadioMapSeer(a.cache, "test", a.setup, n_min=k, n_max=k,
                              seed=2000, cache_gain=True, augment=False,
                              obs_on_signal=False)
            ds.set_epoch(0)
            dl = DataLoader(ds, batch_size=a.bs, shuffle=False,
                            num_workers=a.workers, pin_memory=True)
            v = evaluate(m, dl, dev)
            nm.append(v)
            print("[%-5s] n=%3d NMSE=%.5f" % (mode, k, v), flush=True)
        out["curves"][mode] = nm
        del m
        torch.cuda.empty_cache()

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("[saved]", a.out)


if __name__ == "__main__":
    main()
