# -*- coding: utf-8 -*-
"""混合精度影响探针：同一权重，fp16 autocast vs fp32，测试集指标差多少。

动机
----
txf_mffgcn_s1 的逐 epoch 验证曲线里出现过孤立尖峰：
    ep10 0.0963 / ep12 0.0320 / ep21 0.0594 / ep33 0.0270
相邻 epoch 却只有 0.008。训练损失本身是平滑下降的，说明不是权重崩了，
更像是**验证时 fp16 溢出**：autocast 下某个 batch 输出 inf，
clamp(0,1) 把 inf 变成 1.0，单张图的误差瞬间爆表。

如果这个猜测成立，那么：
  1) best.pt 的选择被噪声污染（选模型依据是 val NMSE）；
  2) 最终 test 指标也可能偏高。

本脚本对同一 checkpoint 分别用 amp=True / amp=False 跑完整测试集，
把差异直接量出来。

用法（在 src/ 下）：
    python probe_amp.py --tag txf_mffgcn_s1 --setup s1 --gpu 0
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (".", "..", os.path.join(_HERE, "..")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import build_loaders            # noqa: E402
from evaluate import build_from_ckpt        # noqa: E402


@torch.no_grad()
def run(model, arch, loader, dev, amp, limit=0):
    from torch.amp import autocast
    se = sq = 0.0
    n = 0
    n_inf = 0
    out_max = 0.0
    for i, b in enumerate(loader):
        obs = b["obs"].to(dev)
        env = b["env"].to(dev)
        mask = b["mask"].to(dev)
        t = b["target"].to(dev).float()
        with autocast("cuda", enabled=amp):
            if arch == "single":
                y = model(torch.cat([obs, env], dim=1))
            else:
                _, y, _ = model(obs, env, mask)
        # 统计溢出（fp16 下 inf/nan 会被 clamp 成 1.0，指标被污染）
        n_inf += int((~torch.isfinite(y)).sum())
        y = y.float()
        out_max = max(out_max, float(y.max()))
        y = y.clamp(0, 1)
        se += float((y - t).pow(2).sum())
        sq += float(t.pow(2).sum())
        n += t.numel()
        if limit and (i + 1) * loader.batch_size >= limit:
            break
    return {"nmse": se / max(sq, 1e-12), "rmse": (se / max(n, 1)) ** 0.5,
            "nonfinite_elems": n_inf, "raw_out_max": out_max}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="txf_mffgcn_s1")
    ap.add_argument("--setup", default="s1")
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join("runs", a.tag, "best.pt"),
                    map_location=dev, weights_only=False)
    model, arch = build_from_ckpt(ck, dev)
    model.eval()
    _, _, te = build_loaders(a.cache, a.setup, batch_size=a.bs,
                             workers=a.workers, seed=0, augment=False,
                             obs_on_signal=False)
    te.dataset.set_epoch(0)

    out = {"tag": a.tag, "setup": a.setup, "best_epoch": ck.get("epoch")}
    for amp in (True, False):
        r = run(model, arch, te, dev, amp, a.limit)
        key = "fp16" if amp else "fp32"
        out[key] = r
        print(f"[{key}] NMSE={r['nmse']:.6f} RMSE={r['rmse']:.6f} "
              f"nonfinite={r['nonfinite_elems']} raw_max={r['raw_out_max']:.3f}",
              flush=True)
    d = (out["fp16"]["nmse"] - out["fp32"]["nmse"]) / max(out["fp32"]["nmse"], 1e-12)
    out["fp16_penalty_pct"] = 100.0 * d
    print(f"[diff] fp16 比 fp32 的 NMSE 偏差 {100*d:+.2f}%")
    path = f"../results/probe_amp_{a.tag}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("[saved]", path)


if __name__ == "__main__":
    main()
