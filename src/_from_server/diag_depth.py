"""盲区深度剖面：把误差按「离最近观测点的距离」分档统计。

论文的说法是「误差集中在阴影/建筑边缘，而不是盲区内部」；本脚本把盲区
像素按到最近可见像素的欧氏距离切成若干档，逐档报 RMSE / NMSE / 像素占比，
于是这句论断可以被直接验证，而不是靠肉眼看误差图。

用法：
  python diag_depth.py --ckpt runs/txf_mffgcn_s1/best.pt --setup s1 \
      --out ../results/depth_s1.json

  --split test --max-samples 2000 可缩短跑时（默认全测试集）。
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader

from dataset import RadioMapSeer
from evaluate import build_from_ckpt

BANDS = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 10 ** 9)]


def band_label(lo, hi):
    return "%d+px" % lo if hi >= 10 ** 9 else "%d-%dpx" % (lo, hi)


def new_acc():
    # [se, sq, n]
    return {band_label(lo, hi): [0.0, 0.0, 0] for lo, hi in BANDS}


def add(acc, dist, err2, tgt2, sel):
    """sel: 参与统计的像素（bool ndarray）。"""
    for lo, hi in BANDS:
        m = sel & (dist >= lo) & (dist < hi)
        n = int(m.sum())
        if not n:
            continue
        slot = acc[band_label(lo, hi)]
        slot[0] += float(err2[m].sum())
        slot[1] += float(tgt2[m].sum())
        slot[2] += n


def pack(acc):
    out = {}
    total_n = sum(v[2] for v in acc.values()) or 1
    for k, (se, sq, n) in acc.items():
        if not n:
            continue
        out[k] = {
            "rmse": (se / n) ** 0.5,
            "nmse": se / max(sq, 1e-12),
            "pixels": n,
            "share": n / total_n,
        }
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--setup", default="s1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model, arch = build_from_ckpt(ck, dev)
    model.eval()

    ds = RadioMapSeer(a.cache, a.split, a.setup, seed=2024, obs_on_signal=False)
    if a.max_samples:
        ds = torch.utils.data.Subset(ds, range(min(a.max_samples, len(ds))))
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                    num_workers=a.workers, pin_memory=True)

    acc_all = new_acc()       # 全图
    acc_blind = new_acc()     # 只统计盲区（未观测）像素
    se_tot = sq_tot = 0.0
    n_pix = 0
    n_img = 0

    for bi, batch in enumerate(dl):
        obs = batch["obs"].to(dev)
        env = batch["env"].to(dev)
        mask = batch["mask"].to(dev)
        target = batch["target"].to(dev)
        if arch == "single":
            pred = model(torch.cat([obs, env], dim=1)).clamp(0, 1)
        else:
            _, pred, _ = model(obs, env, mask)
            pred = pred.clamp(0, 1)

        err2_b = (pred - target).pow(2).squeeze(1).double().cpu().numpy()
        tgt2_b = target.pow(2).squeeze(1).double().cpu().numpy()
        mask_b = (mask.squeeze(1) > 0.5).cpu().numpy()

        se_tot += float(err2_b.sum())
        sq_tot += float(tgt2_b.sum())
        n_pix += err2_b.size
        n_img += err2_b.shape[0]

        for j in range(err2_b.shape[0]):
            observed = mask_b[j]
            # distance_transform_edt 给出「到最近的 0 元素」的距离，
            # 传 ~observed 即：每个像素到最近观测像素的欧氏距离。
            dist = distance_transform_edt(~observed)
            blind = ~observed
            add(acc_all, dist, err2_b[j], tgt2_b[j], np.ones_like(blind))
            add(acc_blind, dist, err2_b[j], tgt2_b[j], blind)

        if (bi + 1) % 20 == 0:
            print("  ... %d batches / %d imgs" % (bi + 1, n_img), flush=True)

    res = {
        "ckpt": a.ckpt,
        "name": a.name or os.path.basename(os.path.dirname(a.ckpt)),
        "setup": a.setup,
        "arch": arch,
        "best_epoch": ck.get("epoch"),
        "n_images": n_img,
        "overall": {"rmse": (se_tot / max(n_pix, 1)) ** 0.5,
                    "nmse": se_tot / max(sq_tot, 1e-12)},
        "all_pixels": pack(acc_all),
        "blind_pixels": pack(acc_blind),
    }

    print("\n%s  setup=%s  n=%d  overall RMSE=%.5f" %
          (res["name"], a.setup, n_img, res["overall"]["rmse"]))
    hdr = "%-10s %10s %10s %8s" % ("距离档", "盲区内RMSE", "盲区NMSE", "像素占比")
    print(hdr)
    print("-" * len(hdr))
    for k, v in res["blind_pixels"].items():
        print("%-10s %10.5f %10.4f %7.1f%%" %
              (k, v["rmse"], v["nmse"], 100 * v["share"]))

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        print("\n[ok] -> %s" % a.out)


if __name__ == "__main__":
    main()
