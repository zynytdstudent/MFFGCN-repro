"""探针：拆开"脉冲噪声"，看 MFFGCN 到底怕哪一种翻转。

Fig.14 里论文说 MFFGCN 最抗噪，我们实测它最敏感（1% 噪声下劣化 8.5 倍，
RadioUNet 只有 4.0 倍）。差 3 倍，种子解释不了，所以要看机理。

1% 的像素翻转其实是两件事叠在一起：
    add    空地 -> 建筑 ：凭空多出几百栋"假建筑"
    remove 建筑 -> 空地 ：真实遮挡少掉几个点
两者对模型的影响完全不同。假建筑会在图结构里新增遮挡边，可能正是
双分支 AFM + 动态图最怕的东西。这里把两者分开测。

用法（在 src/ 下）：
    python probe_noise_mode.py --n 100 --gpu 0
"""
import argparse
import json
import os

import numpy as np
import torch

from dataset import RadioMapSeer
from evaluate import build_from_ckpt
from supp_eval import load_model, forward   # 复用已验证的口径（含 radiounet stage=2）

MODELS = [("MFFGCN", "txf_mffgcn_s3", "mffgcn"),
          ("RadioUNet", "bl_radiounet_s3", "radiounet"),
          ("RME-GAN", "bl_rmegan_s3", "pix2pix")]


class Acc:
    def __init__(self):
        self.se = 0.0
        self.n = 0.0

    def add(self, pred, tgt):
        # 必须和 evaluate.py 一样先裁剪到 [0,1] —— forward 返回的是未裁剪的输出。
        # 漏掉这一步 clean NMSE 会虚高 3 倍（踩过）。
        pred = pred.clamp(0, 1)
        self.se += float(((pred - tgt) ** 2).sum())
        self.n += float((tgt ** 2).sum())

    def out(self):
        return self.se / max(self.n, 1e-12), self.n


def flip(bld, p, rng, mode):
    """mode: add(只 0->1) / remove(只 1->0) / both(对称翻转)"""
    if p <= 0:
        return bld
    x = bld.clone()
    flat = x.view(-1)
    n_pix = int(round(p * x.numel()))
    if mode == "both":
        idx = torch.from_numpy(rng.choice(x.numel(), size=n_pix, replace=False))
        flat[idx] = 1.0 - flat[idx]
    else:
        want = 1.0 if mode == "add" else 0.0
        cand = torch.nonzero(flat != want, as_tuple=False).view(-1).cpu().numpy()
        if len(cand) == 0:
            return bld
        idx = torch.from_numpy(rng.choice(cand, size=min(n_pix, len(cand)),
                                          replace=False))
        flat[idx] = want
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--p", type=float, default=0.01)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    dev = torch.device("cuda")
    models = []
    for nm, t, k in MODELS:
        m, kk = load_model(t, k, dev)
        m.eval()          # 必须！否则 BN 走 batch 统计，bs=1 时会严重失真
        models.append((nm, m, kk))

    ds = RadioMapSeer("../data/cache", "test", "s3", seed=2024)
    modes = ["clean", "both", "add", "remove"]
    acc = {(nm, md): Acc() for nm, _, _ in MODELS for md in modes}

    for i in range(a.n):
        b = ds[i]
        obs = b["obs"][None].to(dev)
        env0 = b["env"][None].to(dev)
        mask = b["mask"][None].to(dev)
        tgt = b["target"][None].to(dev)
        for md in modes:
            env = env0.clone()
            if md != "clean":
                rng = np.random.default_rng([2024 + i * 7919, 31])
                env[:, 0:1] = flip(env0[:, 0:1], a.p, rng, md)
            for nm, m, kind in models:
                acc[(nm, md)].add(forward(m, kind, obs, env, mask), tgt)
        if (i + 1) % 25 == 0:
            print("  %d/%d" % (i + 1, a.n), flush=True)

    out = {"n": a.n, "p": a.p, "setup": "s3", "models": {}}
    print("\n%-10s %10s %10s %10s %10s" % ("model", "clean", "both", "add", "remove"))
    for nm, _, _ in MODELS:
        row = {}
        for md in modes:
            v, _ = acc[(nm, md)].out()
            row[md] = v
        out["models"][nm] = row
        c = row["clean"]
        print("%-10s %10.5f %10.5f %10.5f %10.5f   (相对 clean: both ×%.1f  add ×%.1f  remove ×%.1f)"
              % (nm, c, row["both"], row["add"], row["remove"],
                 row["both"] / c, row["add"] / c, row["remove"] / c))

    with open("../results/probe_noise_mode.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("saved ../results/probe_noise_mode.json")


if __name__ == "__main__":
    main()
