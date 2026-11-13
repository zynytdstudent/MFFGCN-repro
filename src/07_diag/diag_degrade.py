# -*- coding: utf-8 -*-
"""诊断 Fig.15 的 degrade 实现：五档环境图到底长什么样（2026-11-13）。

怀疑点
----
supp_eval.degrade() 是 `adaptive_avg_pool2d(res) -> nearest 放大回 256 -> >0.5 二值化`。
对 0/1 的二值建筑图，"平均池化 + 阈值 0.5" 等价于**多数表决**：只要 4x4 块里
超过一半是建筑，整块就还原成建筑。这几乎是无损的形状近似 —— 模型看到的建筑
轮廓和原图基本一样，误差自然涨不上去。

而且实测 NMSE 是
    256: 0.00533   192: 0.01785   128: 0.01802   85: 0.02572   64: 0.02831
192 -> 128 几乎不动，256 -> 192 却跳 3.35 倍，形状都不像论文的单调曲线。
怀疑非整数倍（256/192 = 1.333）的 nearest 放大产生了像素错位。

本脚本把同一张图的五档退化结果并排画出来，同时打印每档相对原图的
建筑像素差异比例，用来看"退化到底改了多少"。

用法（项目根目录）
    python diag_degrade.py                 # 默认一张
    python diag_degrade.py --n 4           # 四张拼一起
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMG = 256
LEVELS = [256, 192, 128, 85, 64]


def degrade(bld, res, binarize=True, mode="nearest"):
    """与 supp_eval.degrade 完全一致的实现，另外多一个 mode 参数做对照。"""
    if res >= IMG:
        return bld.clone()
    small = F.adaptive_avg_pool2d(bld, (res, res))
    up = F.interpolate(small, size=(IMG, IMG), mode=mode,
                       align_corners=False if mode != "nearest" else None)
    return (up > 0.5).float() if binarize else up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--out", default="../figs/diag_degrade.png")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dataset import RadioMapSeer

    ds = RadioMapSeer(a.cache, "test", "s3", seed=2024)
    # 等间隔取 n 张，和 supp_eval 的 --span 一致
    idx = np.linspace(0, len(ds) - 1, a.n).astype(int)

    fig, axes = plt.subplots(a.n * 3, len(LEVELS),
                             figsize=(2.0 * len(LEVELS), 2.0 * a.n * 3),
                             squeeze=False)
    for j, i in enumerate(idx):
        b = ds[int(i)]
        env = b["env"][None, 0:1]                      # 建筑物通道
        for k, res in enumerate(LEVELS):
            cur = degrade(env, res)
            diff = float((cur - env).abs().mean())

            # 三个视角：原尺度最近邻、差分、以及"平均池化不二值化"的灰度版
            for row, (img, ttl) in enumerate((
                    (cur[0, 0].numpy(), "res=%d  diff=%.3f" % (res, diff)),
                    ((cur - env).abs()[0, 0].numpy(), "|delta|"),
                    (F.interpolate(F.adaptive_avg_pool2d(env, (res, res)),
                                   size=(IMG, IMG), mode="nearest"
                                   )[0, 0].numpy(), "gray (no thr)"))):
                ax = axes[j * 3 + row][k]
                ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(ttl, fontsize=8)
                if k == 0:
                    ax.set_ylabel({0: "sample %d" % j, 1: "delta", 2: "gray"}[row],
                                  fontsize=8)
            print("  res=%3d  mean|delta| = %.4f  (%.1f%% 像素被改)"
                  % (res, diff, 100 * float((cur != env).float().mean())))

    fig.tight_layout()
    fig.savefig(a.out, dpi=120, bbox_inches="tight")
    print("[ok]", a.out)


if __name__ == "__main__":
    main()
