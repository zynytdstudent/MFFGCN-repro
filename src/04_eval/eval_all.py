"""用修复后的 dataset.py 重新评估多个已有 checkpoint，给出真实测试指标。

修复内容：tx_coords 按全局行号索引，此前 val/test 用了 split 内位置，
导致评估输入的发射机位置全错（训练集恰好无影响）。

用法：
  python eval_all.py --tags calib_single calib_mffgcn calib_sigmoid bs15_s1
"""
import argparse
import json
import os

import torch

from dataset import RadioMapSeer
from evaluate import build_from_ckpt
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--split", default="test")
    ap.add_argument("--setup", default="s1")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()

    dev = torch.device("cuda:" + a.gpu if torch.cuda.is_available() else "cpu")
    print("%-22s %5s %9s %9s %9s %9s" %
          ("run", "ep", "NMSE", "RMSE", "cNMSE", "cRMSE"))
    rows = []
    for tag in a.tags:
        ckpt = os.path.join(a.runs, tag, "best.pt")
        if not os.path.exists(ckpt):
            print("%-22s  缺失" % tag)
            continue
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        model, arch = build_from_ckpt(ck, dev)
        model.eval()
        args = ck.get("args", {})
        oos = bool(args.get("obs_on_signal", False))
        ds = RadioMapSeer(a.cache, a.split, args.get("setup", a.setup),
                          seed=int(args.get("seed", 0)) + 2000
                          if a.split == "test" else 2024,
                          obs_on_signal=oos)
        ld = DataLoader(ds, batch_size=a.batch, shuffle=False,
                        num_workers=a.workers, pin_memory=True)
        se = sq = sc = ne = 0.0
        with torch.no_grad():
            for b in ld:
                obs = b["obs"].to(dev)
                env = b["env"].to(dev)
                mask = b["mask"].to(dev)
                tgt = b["target"].to(dev).float()
                if arch == "single":
                    p = model(torch.cat([obs, env], dim=1))
                    c = p
                else:
                    c, p, _ = model(obs, env, mask)
                p = p.float().clamp(0, 1)
                c = c.float().clamp(0, 1)
                se += float((p - tgt).pow(2).sum())
                sc += float((c - tgt).pow(2).sum())
                sq += float(tgt.pow(2).sum())
                ne += tgt.numel()
        nm, rm = se / sq, (se / ne) ** 0.5
        cn, cr = sc / sq, (sc / ne) ** 0.5
        print("%-22s %5s %9.4f %9.4f %9.4f %9.4f" %
              (tag, ck.get("epoch"), nm, rm, cn, cr))
        rows.append(dict(tag=tag, epoch=ck.get("epoch"), nmse=nm, rmse=rm,
                         coarse_nmse=cn, coarse_rmse=cr,
                         obs_on_signal=oos, setup=args.get("setup")))
    out = "eval_all_%s.json" % a.split
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print("[ok] wrote", out)


if __name__ == "__main__":
    main()
