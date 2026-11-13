# -*- coding: utf-8 -*-
"""论文 Fig.7 的正确做法：固定观测点数，扫"每张图有多少个观测样本"。

为什么重写
----------
原来的 paper_fig_7.py 把横轴理解成了**训练集城市数**（fs_*_c50 … c300，
即用前 50/100/…/300 个城市训练），一共 42 个训练任务。这是错的。

论文原文（results/paper_text.txt 第 855 行）：
    "In Setup 1, we also compare the effect of different numbers of
     observation samples on different models, as shown in Fig. 7. The errors
     of all the models decrease with the increase of the number of samples.
     The decrease of the errors of KNN and Kriging is more obvious, which is
     due to the fact that the interpolation algorithms are more sensitive to
     the number of samples."

两个决定性证据：
  1) 原文写的是 "observation samples"，横轴 50–300 正好落在观测预算
     [10, 300] 的区间里；
  2) KNN / Kriging 没有训练过程，只有"观测点多少"才对它们有意义。
     如果横轴是训练集大小，这两条曲线根本不该存在。

所以正确做法是：**权重不动**，只改测试时每张图的观测點数 n，
在 n ∈ {50,100,150,200,250,300} 上各评一遍。7 个模型 × 6 个点 = 42 次
纯推理，比原来 42 次训练便宜两个数量级。

配对性：dataset 的观测点随机种子由 (seed, 样本序号 i, epoch, txid) 决定，
与模型无关，所以同一 n 下 7 个模型看到的是**完全相同的观测点位置**。

用法（在 src/ 下）：
    python fig7_obs_sweep.py --gpu 0                 # 全量测试集
    python fig7_obs_sweep.py --gpu 0 --limit 3000    # 快速版

    # 修复版权重落地后，只重扫 Pix2pix+ / RME-GAN 两条线并合并：
    python fig7_obs_sweep.py --gpu 0 --fix-tags --merge \
        --models "Pix2pix+,RME-GAN"
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (".", "..", os.path.join(_HERE, "..")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import RadioMapSeer          # noqa: E402
from evaluate import build_from_ckpt      # noqa: E402
from baselines import build_baseline       # noqa: E402

# (显示名, runs 目录, 类型)
# 注：U-Net 用 txf_single_s1（arch=single，nmse 0.00583）；
# 早先写的 ab_single_s1 其实是"单编码器消融"的另一支且 best.pt 已丢。
MODELS = [
    ("AE",        "bl_ae_s1",        "ae"),
    ("DeepAE",    "bl_deepae_s1",    "deepae"),
    ("U-Net",     "txf_single_s1",   "single"),
    ("RadioUNet", "bl_radiounet_s1", "radiounet"),
    ("Pix2pix+",  "bl_pix2pix_s1",   "pix2pix"),
    ("RME-GAN",   "bl_rmegan_s1",    "pix2pix"),
    ("MFFGCN",    "txf_mffgcn_s1",   "mffgcn"),
]
COUNTS = [50, 100, 150, 200, 250, 300]

# 2026-11-09 修差距：Pix2pix+ / RME-GAN 用错的 lr 已重训，产物在 *_fix。
# 默认仍读旧 tag（保持可复现），加 --fix-tags 才切到重训版。
FIX_TAGS = {
    "bl_pix2pix_s1": "bl_pix2pix_s1_fix",
    "bl_rmegan_s1": "bl_rmegan_s1_fix",
}


def load_model(tag, kind, dev):
    ck = torch.load(os.path.join("runs", tag, "best.pt"),
                    map_location=dev, weights_only=False)
    if kind == "mffgcn":
        m, arch = build_from_ckpt(ck, dev)
        return m, "mffgcn"
    if kind == "single":
        m, arch = build_from_ckpt(ck, dev)
        return m, "single"
    m = build_baseline(kind, ae_skip=(ck.get("args") or {}).get("ae_skip",
                                                               "none")).to(dev)
    m.load_state_dict(ck["model"])
    if hasattr(m, "stage"):
        m.stage = 2          # RadioUNet 用第二级的输出
    return m, kind


@torch.no_grad()
def evaluate(m, arch, loader, dev, limit=0):
    se = sq = 0.0
    n = 0
    for i, b in enumerate(loader):
        obs = b["obs"].to(dev)
        env = b["env"].to(dev)
        mask = b["mask"].to(dev)
        t = b["target"].to(dev).float()
        if arch == "single":
            y = m(torch.cat([obs, env], dim=1))
        elif arch == "mffgcn":
            _, y, _ = m(obs, env, mask)
        else:
            out = m(obs, env)
            y = out[1] if isinstance(out, tuple) else out
        y = y.float().clamp(0, 1)
        se += float((y - t).pow(2).sum())
        sq += float(t.pow(2).sum())
        n += t.numel()
        if limit and n >= limit * 256 * 256:
            break
    return se / max(sq, 1e-12), (se / max(n, 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--setup", default="s1")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="../results/fig7_obs_sweep.json")
    ap.add_argument("--models", default="", help="逗号分隔的显示名子集，默认全部")
    ap.add_argument("--merge", action="store_true", help="合并进已有 --out，不覆盖")
    ap.add_argument("--fix-tags", action="store_true",
                    help="Pix2pix+/RME-GAN 改用重训版 bl_*_s1_fix 权重")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res = {"setup": a.setup, "counts": COUNTS, "curves": {}}
    if a.merge and os.path.exists(a.out):
        with open(a.out) as fh:
            old = json.load(fh)
        res["counts"] = old.get("counts", COUNTS)
        res["curves"] = old.get("curves", {})
    want = [s.strip() for s in a.models.split(",") if s.strip()]
    models = list(MODELS)
    if a.fix_tags:
        models = [(n, FIX_TAGS.get(t, t), k) for n, t, k in models]
    for name, tag, kind in models:
        if want and name not in want:
            continue
        if not os.path.exists(os.path.join("runs", tag, "best.pt")):
            print(f"[skip] {name} ({tag}) 无权重", flush=True)
            continue
        m, arch = load_model(tag, kind, dev)
        m.eval()
        nm, rm = [], []
        for k in COUNTS:
            # n_min == n_max == k ⇒ 每张图恰好 k 个观测点
            ds = RadioMapSeer(a.cache, "test", a.setup, n_min=k, n_max=k,
                              seed=2000, cache_gain=True, augment=False,
                              obs_on_signal=False)
            ds.set_epoch(0)
            dl = DataLoader(ds, batch_size=a.bs, shuffle=False,
                            num_workers=a.workers, pin_memory=True)
            nmse, rmse = evaluate(m, arch, dl, dev, a.limit)
            nm.append(nmse)
            rm.append(rmse)
            print(f"[{name:9s}] n={k:3d}  NMSE={nmse:.5f}  RMSE={rmse:.5f}",
                  flush=True)
        res["curves"][name] = {"nmse": nm, "rmse": rm}
        del m
        torch.cuda.empty_cache()

    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("[saved]", a.out)


if __name__ == "__main__":
    main()
