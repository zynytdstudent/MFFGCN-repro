"""同一份权重，分别用两种观测点放置策略评估，量化口径差异。

背景：旧的消融 run 训练/评估时都用 --obs-on-signal（观测点只落在有信号的
像素上），主线用全图均匀撒点。如果两者指标差得很多，消融表就得重跑；如果
差得很少，旧结果还能用。

用法：python oos_check.py --tag calib_mffgcn --n 400
"""
import argparse

import torch

from dataset import RadioMapSeer
from evaluate import build_from_ckpt


def run(ckpt, cache, setup, oos, dev, n):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    model, arch = build_from_ckpt(ck, dev)
    model.eval()
    ds = RadioMapSeer(cache, "test", setup, seed=2000, obs_on_signal=oos)
    ds.rows = ds.rows[:n]
    se = sq = 0.0
    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            obs = b["obs"][None].to(dev)
            env = b["env"][None].to(dev)
            mask = b["mask"][None].to(dev)
            tgt = b["target"][None].to(dev).float()
            if arch == "single":
                p = model(torch.cat([obs, env], dim=1)).clamp(0, 1)
            else:
                _, p, _ = model(obs, env, mask)
                p = p.clamp(0, 1)
            se += float((p - tgt).pow(2).sum())
            sq += float(tgt.pow(2).sum())
    return (se / sq) ** 0.5, (se / tgt.numel() / len(ds)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="calib_mffgcn")
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--setup", default="s1")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()

    dev = torch.device("cuda:" + a.gpu)
    ck = "runs/%s/best.pt" % a.tag
    print("%-22s %6s %9s %9s" % ("obs policy", "n", "NMSE", "RMSE"))
    for oos, name in ((False, "均匀撒点"), (True, "只放有信号处")):
        nm, rm = run(ck, a.cache, a.setup, oos, dev, a.n)
        print("%-22s %6d %9.5f %9.5f" % (name, a.n, nm, rm), flush=True)


if __name__ == "__main__":
    main()
