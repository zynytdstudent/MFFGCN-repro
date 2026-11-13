# -*- coding: utf-8 -*-
"""量化每个模型对"建筑图空间结构"的依赖度，解释 Fig.14 的排序为什么反了。

动机
----
论文 Fig.14（Setup 3，环境图脉冲噪声）说 MFFGCN 抗噪最好，理由是双编码器。
我们的复现是**完全镜像**的（1.0% 档劣化倍数）：

    论文  MFFGCN 2.67x  <  RME-GAN 7.30x  <  RadioUNet 10.46x
    我们  RME-GAN 2.93x <  RadioUNet 6.23x <  MFFGCN 13.80x

三个模型的抗噪排序恰好整个倒过来，而且数值近乎一一对应。这种"整体翻转"
不像偶然，更像"谁更依赖建筑图"这一个潜变量在排序。于是这里直接把那个
潜变量测出来：

    条件 A  clean env           —— 正常
    条件 B  shuffled env        —— 保持建筑像素比例不变，但把像素位置整体
                                   打乱（用同一个随机置换）。建筑图的**统计量
                                   完全不变、空间结构全丢**。

B 掉多少，就等于该模型"用了多少建筑图的结构信息"。依赖越深，脉冲噪声
（局部伪造结构）造成的伤害越大。

为什么用整体置换而不是直接零化：零化会同时改变建筑比例（输入分布外推），
置换只破坏结构，是更干净的对照组。

用法（在 src/ 下）：
    python env_dependence.py --n 200 --gpu 1
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (".", "..", os.path.join(_HERE, ".."), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import RadioMapSeer          # noqa: E402
from supp_eval import MODELS, load_model, forward, sample_idx   # noqa: E402


def permute_spatial(x, rng):
    """对每张图做固定的空间置换：建筑像素数不变，位置整体打乱。"""
    b = x.shape[0]
    n = x.shape[-2] * x.shape[-1]
    out = x.clone()
    for i in range(b):
        p = torch.from_numpy(rng.permutation(n))
        out[i, 0] = x[i, 0].reshape(-1)[p].reshape(x.shape[-2], x.shape[-1])
    return out


@torch.no_grad()
def run(args, dev):
    ds = RadioMapSeer("../data/cache", "test", args.setup, seed=2024)
    idx = sample_idx(ds, args.n, args.span)
    models = []
    for name, tpl, kind in MODELS:
        m, k = load_model(tpl.format(s=args.setup), kind, dev)
        m.eval()
        models.append((name, m, k))
        print("[load] %-10s %s" % (name, tpl.format(s=args.setup)), flush=True)

    conds = ["clean", "shuffled"]
    se = {(n, c): 0.0 for n, _, _ in MODELS for c in conds}
    sq = 0.0
    for k, i in enumerate(idx):
        b = ds[i]
        obs = b["obs"][None].to(dev)
        env0 = b["env"][None].to(dev)
        mask = b["mask"][None].to(dev)
        tgt = b["target"][None].to(dev)
        sq += float(tgt.pow(2).sum())

        rng = np.random.default_rng([2024 + i * 7919, 77])
        envs = {"clean": env0, "shuffled": permute_spatial(env0, rng)}
        for c in conds:
            for name, m, kind in models:
                y = forward(m, kind, obs, envs[c], mask).float().clamp(0, 1)
                se[(name, c)] += float((y - tgt).pow(2).sum())
        if (k + 1) % 50 == 0:
            print("  %d/%d" % (k + 1, len(idx)), flush=True)

    out = {"setup": args.setup, "n": len(idx), "cond": conds, "models": {}}
    for name in [n for n, _, _ in MODELS]:
        d = {c: {"nmse": se[(name, c)] / max(sq, 1e-12)} for c in conds}
        d["ratio"] = d["shuffled"]["nmse"] / max(d["clean"]["nmse"], 1e-12)
        out["models"][name] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="s3")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--span", action="store_true", default=True)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--out", default="../results/env_dependence.json")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = run(a, dev)

    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("\n[env dependence] 打乱建筑图结构后误差涨多少")
    print("%-11s %10s %10s %8s" % ("model", "clean", "shuffled", "ratio"))
    for n, d in res["models"].items():
        print("%-11s %10.5f %10.5f %7.2fx"
              % (n, d["clean"]["nmse"], d["shuffled"]["nmse"], d["ratio"]))
    print("[saved]", a.out)


if __name__ == "__main__":
    main()
