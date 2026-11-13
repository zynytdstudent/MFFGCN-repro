# -*- coding: utf-8 -*-
"""口径探针：同一张 checkpoint，用不同"平均方式"算 NMSE / RMSE，看差多少。

为什么需要这个
--------------
train.py 里的 MetricAccumulator 是 **micro** 口径：
        NMSE = Σ_all (ŷ-y)² / Σ_all y²
论文（以及 RadioUNet 官方评测脚本）用的是 **macro** 口径：
        NMSE = mean_i [ Σ(ŷ_i-y_i)² / Σ y_i² ]
两者在"每张图能量差别很大"时会明显不同：micro 被高能量图主导，
macro 让每张图权重相同（含那些信号极弱、比值爆炸的图）。

如果论文报的是 macro 而我们报 micro，那"数值差一截"就有一半是口径而不是模型。

同时输出：
  - 每张图的 Σy² 分布（极端值会把 macro 拉偏）
  - 只在"有信号像素"上算的 NMSE（另一种常见口径）
  - 观测点总数的分布（确认 [10,300] 的预算按预期生效）

用法（在 src/ 下）：
    python 07_diag/probe_metric.py --tag txf_mffgcn_s1 --setup s1 --gpu 0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

# 本地是分目录存放，服务器是扁平 src/，两边都要能跑
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (".", "..", os.path.join(_HERE, ".."),
           os.path.join(_HERE, "..", "03_train"),
           os.path.join(_HERE, "..", "04_eval"),
           os.path.join(_HERE, "..", "05_baselines")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import build_loaders            # noqa: E402
from evaluate import build_from_ckpt        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="txf_mffgcn_s1")
    ap.add_argument("--setup", default="s1")
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--limit", type=int, default=0, help="0=全量测试集")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(os.path.join("runs", a.tag, "best.pt"),
                    map_location=dev, weights_only=False)
    model, arch = build_from_ckpt(ck, dev)
    model.eval()
    print(f"[load] {a.tag} arch={arch} best_ep={ck.get('epoch')}", flush=True)

    _, _, te = build_loaders(a.cache, a.setup, batch_size=a.bs,
                             workers=a.workers, seed=0, augment=False,
                             obs_on_signal=False)
    te.dataset.set_epoch(0)

    se_i, sq_i, nz_i, nzse_i, seall_i, nobs = [], [], [], [], [], []
    se_tot = sq_tot = 0.0
    n_elem = 0
    n_img = 0
    with torch.no_grad():
        for i, b in enumerate(te):
            obs = b["obs"].to(dev)
            env = b["env"].to(dev)
            mask = b["mask"].to(dev)
            tgt = b["target"].to(dev)
            if arch == "single":
                y = model(torch.cat([obs, env], dim=1))
            else:
                _, y, _ = model(obs, env, mask)
            y = y.float().clamp(0, 1)
            t = tgt.float()

            se = (y - t).pow(2).flatten(1).sum(1)      # 每张图平方误差
            sq = t.flatten(1).pow(2).sum(1)            # 每张图真值能量
            nz = (t.flatten(1) > 1e-6).sum(1)          # 有信号像素数
            nzse = (y - t).pow(2).flatten(1)[t.flatten(1) > 1e-6].sum() \
                if False else (y - t).pow(2) * (t > 1e-6).float()
            nzse = nzse.flatten(1).sum(1)

            se_i.append(se.cpu().numpy())
            sq_i.append(sq.cpu().numpy())
            nz_i.append(nz.cpu().numpy())
            nzse_i.append(nzse.cpu().numpy())
            seall_i.append(se.cpu().numpy())
            nobs.append(mask.flatten(1).sum(1).cpu().numpy())

            se_tot += float(se.sum())
            sq_tot += float(sq.sum())
            n_elem += t.numel()
            n_img += t.shape[0]
            if a.limit and n_img >= a.limit:
                break

    se_i = np.concatenate(se_i).astype(np.float64)
    sq_i = np.concatenate(sq_i).astype(np.float64)
    nz_i = np.concatenate(nz_i).astype(np.float64)
    nzse_i = np.concatenate(nzse_i).astype(np.float64)
    nobs = np.concatenate(nobs)

    per_img = se_i / np.maximum(sq_i, 1e-12)
    micro_nmse = se_tot / max(sq_tot, 1e-12)
    macro_nmse = float(per_img.mean())
    micro_rmse = (se_tot / n_elem) ** 0.5
    macro_rmse = float(np.sqrt(se_i / (256.0 * 256.0)).mean())

    # 只在有信号像素上：分子分母都限制在 t>0 区域
    sig_nmse = float((nzse_i / np.maximum(sq_i, 1e-12)).mean())

    out = {
        "tag": a.tag, "setup": a.setup, "n_images": int(len(se_i)),
        "nmse_micro": micro_nmse,
        "nmse_macro": macro_nmse,
        "nmse_macro_on_signal": sig_nmse,
        "rmse_micro": micro_rmse,
        "rmse_macro_perimage": macro_rmse,
        "E_t2": float(sq_i.mean() / (256.0 * 256.0)),
        "E_t2_median": float(np.median(sq_i) / (256.0 * 256.0)),
        "per_img_nmse_p50": float(np.percentile(per_img, 50)),
        "per_img_nmse_p90": float(np.percentile(per_img, 90)),
        "per_img_nmse_p99": float(np.percentile(per_img, 99)),
        "per_img_nmse_max": float(per_img.max()),
        "sq_p01": float(np.percentile(sq_i, 1)),
        "sq_p99": float(np.percentile(sq_i, 99)),
        "signal_ratio_mean": float((nz_i / (256.0 * 256.0)).mean()),
        "n_obs_mean": float(nobs.mean()),
        "n_obs_min": int(nobs.min()), "n_obs_max": int(nobs.max()),
        "ratio_macro_over_micro": macro_nmse / max(micro_nmse, 1e-12),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    os.makedirs("../results", exist_ok=True)
    path = f"../results/probe_metric_{a.tag}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("[saved]", path)


if __name__ == "__main__":
    main()
