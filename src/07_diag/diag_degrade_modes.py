# -*- coding: utf-8 -*-
"""Fig.15 口径消歧：四种降采样/上采样组合，哪个能复现论文的劣化幅度（2026-11-13）。

背景
----
论文正文（paper_text.txt:1123）只有一句：
    "we downsample and upsample the original environment map first, and then
     processed the new map back to a binary mask through the threshold"
并强调 "the resolution scaling causes a significant change in the **shape of
the building edge**"。

我们现在的实现是 avgpool -> nearest -> 阈值 0.5，实测只改动 1.2%(192) ~ 4.2%(64)
的像素（见 diag_degrade.py）。多数表决把边缘糊掉了却没让建筑消失，所以
25% 档只涨 5.3 倍，论文是 16 倍。

nearest 上采样会把粗网格"整齐复制"，几何边界依然是准的；bilinear/bicubic 会
让边缘落在半像素上，阈值化之后产生半像素错位——这才是论文说的 "shape of the
building edge" 变了。本脚本把四种组合都跑一遍，看哪种能对上论文的倍数。

用法（项目根目录，GPU 可选）
    python diag_degrade_modes.py --gpu 1 --n 200
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import RadioMapSeer          # noqa: E402
from supp_eval import Acc, forward, load_model   # noqa: E402

IMG = 256
LEVELS = [192, 128, 85, 64]


def degrade(bld, res, mode):
    """四种口径。

    nearest   : 现状。avgpool -> nearest 放大 -> >0.5
    bilinear  : avgpool -> bilinear 放大 -> >0.5（边缘落在半像素，二值化后错位）
    bicubic   : avgpool -> bicubic 放大 -> >0.5（带过冲，边缘更"脏"）
    decimate  : 每 k 个像素取一个（不做平均）-> bilinear 放大 -> >0.5
    """
    if res >= IMG:
        return bld
    if mode == "decimate":
        k = max(1, int(round(IMG / res)))
        small = bld[:, :, ::k, ::k][:, :, :res, :res]
        up = F.interpolate(small, size=(IMG, IMG), mode="bilinear",
                           align_corners=False)
        return (up > 0.5).float()
    small = F.adaptive_avg_pool2d(bld, (res, res))
    if mode == "nearest":
        up = F.interpolate(small, size=(IMG, IMG), mode="nearest")
    else:
        up = F.interpolate(small, size=(IMG, IMG), mode=mode,
                           align_corners=False)
    return (up > 0.5).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--setup", default="s3")
    ap.add_argument("--models", default="MFFGCN,RadioUNet")
    ap.add_argument("--out", default="../results/diag_degrade_modes.json")
    a = ap.parse_args()

    dev = torch.device("cuda:%d" % a.gpu if torch.cuda.is_available() else "cpu")
    wanted = [m.strip() for m in a.models.split(",") if m.strip()]
    from supp_eval import MODELS
    models = []
    for name, tpl, kind in MODELS:
        if name not in wanted:
            continue
        tag = tpl.format(s=a.setup)
        m, k = load_model(tag, kind, dev)
        m.eval()
        models.append((name, m, k))
        print("[load]", name, tag, flush=True)

    ds = RadioMapSeer("../data/cache", "test", a.setup, seed=2024)
    idx = np.linspace(0, len(ds) - 1, a.n).astype(int)   # --span 等间隔

    modes = ["nearest", "bilinear", "bicubic", "decimate"]
    acc = {(m, mo, lv): Acc() for m, _, _ in models
           for mo in modes for lv in [256] + LEVELS}

    for j, i in enumerate(idx):
        b = ds[int(i)]
        obs = b["obs"][None].to(dev)
        env0 = b["env"][None].to(dev)
        mask = b["mask"][None].to(dev)
        tgt = b["target"][None].to(dev)
        for mo in modes:
            for lv in [256] + LEVELS:
                env = env0.clone()
                env[:, 0:1] = degrade(env0[:, 0:1], lv, mo)
                for name, m, kind in models:
                    acc[(name, mo, lv)].add(forward(m, kind, obs, env, mask), tgt)
        if (j + 1) % 40 == 0:
            print("  %d/%d" % (j + 1, len(idx)), flush=True)

    out = {"n": a.n, "setup": a.setup, "levels": LEVELS, "modes": modes,
           "models": {}}
    print("\n%-10s %-9s %8s %8s %8s %8s %8s" %
          ("model", "mode", "256", "192", "128", "85", "64"))
    for name, _, _ in models:
        out["models"][name] = {}
        for mo in modes:
            base = acc[(name, mo, 256)].out()["nmse"]
            row = [acc[(name, mo, lv)].out()["nmse"] for lv in LEVELS]
            out["models"][name][mo] = {"base": base, "levels": row}
            print("%-10s %-9s %8.5f %8.5f %8.5f %8.5f %8.5f  (x%.1f .. x%.1f)"
                  % (name, mo, base, *row, row[0] / base, row[-1] / base))
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\n[ok] wrote", a.out)
    print("论文参照: 25%(64) 档约 x16, 75%(192) 档约 x6（相对干净基线）")


if __name__ == "__main__":
    main()
