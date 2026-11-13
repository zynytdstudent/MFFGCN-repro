"""Fig.16 脚本口径自检（修正版）：miss=0 必须复现主模型 s1 的 ~0.0045。

上一版漏了 model.eval()，BatchNorm 走 batch 统计，数字全错（0.0149 是假的）。
这一版加回 eval()，并且同时打印"逐样本 NMSE 的分位数"——
用来判断 fig16.json 里 0.08 那个量级到底是口径错还是真退化。

用法：python probe_fig16_base.py [--n 60] [--gpu 1]
"""
import argparse
import importlib

import numpy as np
import torch

from dataset import RadioMapSeer

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="1")
ap.add_argument("--n", type=int, default=60)
a = ap.parse_args()

dev = torch.device("cuda:%s" % a.gpu)
pf = importlib.import_module("paper_fig_16")
ds = RadioMapSeer("../data/cache", "test", "s1", seed=2024)

model, kind = pf.load_model("txf_mffgcn_s1", "mffgcn", dev)
model.eval()

print("样本数 n=%d" % a.n)
print("%-8s %10s %10s %10s %10s" % ("miss", "NMSE", "RMSE", "中位NMSE", "P90NMSE"))
for miss in (0, 1, 2, 4):
    per = []
    se = sq = 0.0
    cnt = 0
    for i in range(min(a.n, len(ds))):
        b = ds[i]
        obs = b["obs"][None].to(dev)
        env = b["env"][None].to(dev)
        mask = b["mask"][None].to(dev)
        tgt = b["target"][None].to(dev)
        if miss:
            env = env.clone()
            env[:, 0:1] = pf.drop_buildings(env[:, 0:1], miss).to(dev)
        with torch.no_grad():
            _, y, _ = model(obs, env, mask)
        y = y.clamp(0, 1)
        e = float(((y - tgt) ** 2).sum())
        t = float((tgt ** 2).sum())
        se += e
        sq += t
        cnt += tgt.numel()
        per.append(e / max(t, 1e-12))
    per = np.asarray(per)
    print("%-8d %10.6f %10.6f %10.6f %10.6f"
          % (miss, se / max(sq, 1e-12), (se / max(cnt, 1)) ** 0.5,
             float(np.median(per)), float(np.percentile(per, 90))))
