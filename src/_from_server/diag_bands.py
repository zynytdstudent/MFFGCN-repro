"""按地图编号分段统计 RMSE，定位测试段误差跳变的位置。

用法：
  python diag_bands.py --ckpt runs/calib_single/best.pt --start 500 --end 700 --step 25
"""
import argparse

import numpy as np
import torch

from dataset import RadioMapSeer
from models import MFFGCN, SingleEncoderUNet


def build(ck):
    a = ck.get("args", {})
    w = tuple(int(x) for x in str(a.get("widths", "32,64,128,256")).split(","))
    bn = int(a.get("bottleneck", 512))
    if a.get("arch", "single") == "single":
        m = SingleEncoderUNet(widths=w, bottleneck=bn)
    else:
        m = MFFGCN(widths=w, bottleneck=bn, k=int(a.get("k", 16)),
                   afm_mode=a.get("afm_mode", "softmax"),
                   gcn_mode=a.get("gcn_mode", "dynamic"),
                   residual=not a.get("no_residual", False))
    m.load_state_dict(ck["model"])
    m.eval()
    return m, a.get("arch", "single")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--setup", default="s1")
    ap.add_argument("--start", type=int, default=500)
    ap.add_argument("--end", type=int, default=701)
    ap.add_argument("--step", type=int, default=25)
    ap.add_argument("--per-map", type=int, default=4, help="每张地图取几个 Tx")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()

    dev = torch.device("cuda:" + a.gpu if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model, arch = build(ck)
    model.to(dev)

    # 用 "test" 之外的通道拿到全部地图：split 参数传一个不存在的名字会取全集
    ds = RadioMapSeer(a.cache, "all", a.setup, seed=a.seed,
                      obs_on_signal=False, augment=False)
    # rows 里每 80 条对应一张地图，按 city 排序
    rows = ds.rows

    print("%-14s %6s %10s %10s" % ("地图区间", "样本数", "RMSE", "NMSE"))
    with torch.no_grad():
        for lo in range(a.start, min(a.end, 701), a.step):
            hi = min(lo + a.step, min(a.end, 701))
            se = 0.0
            sq = 0.0
            n = 0
            for i, (idx, city, txid) in enumerate(rows):
                if not (lo <= city < hi):
                    continue
                if txid % (80 // a.per_map) != 0:
                    continue
                b = ds[i]
                obs = b["obs"][None].to(dev)
                env = b["env"][None].to(dev)
                mask = b["mask"][None].to(dev)
                tgt = b["target"][None].to(dev)
                if arch == "single":
                    p = model(torch.cat([obs, env], dim=1))
                else:
                    _, p, _ = model(obs, env, mask)
                se += float((p - tgt).pow(2).sum())
                sq += float(tgt.pow(2).sum())
                n += 1
            if n:
                print("%-14s %6d %10.4f %10.4f" %
                      ("%d-%d" % (lo, hi - 1), n, (se / (n * 65536)) ** 0.5, se / max(sq, 1e-9)))


if __name__ == "__main__":
    main()
