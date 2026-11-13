"""批量重评已有 checkpoint：全局 / 盲区 / 观测点三个区域 + 盲区距离分带。

背景：2026-11-04 之前的所有评估都用了错误的发射机位置（dataset.py 用 split
内局部下标去查全局坐标表），训练不受影响、权重有效，但测试指标全部失真。
本脚本用修复后的数据集重新评估，并且一次产出两份数据：

  1. 三个区域的 NMSE/RMSE —— 其中 constr（盲区）那一栏就是论文表 III 的盲区指标
  2. 盲区内部按"到盲区边界的距离"分带的误差曲线 —— 对应论文诊断"误差集中在
     盲区边缘还是盲区深处"的图

观察点采样必须与训练时一致：数据集 seed = 训练 seed + 2000。

用法：
    python reeval.py --tags ab_nogcn_s2 ab_static_s2 --gpu 0
    python reeval.py --tags ALL --gpu 0 --out reeval_all.json
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader

from dataset import RadioMapSeer
from evaluate import build_from_ckpt

BINS = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 1e9]
BAND_LABELS = ["0-1", "1-2", "2-4", "4-8", "8-16", "16-32", "32+"]


def _blank_regions():
    return {k: [0.0, 0.0, 0] for k in ("all", "constr", "obs")}


def eval_run(ckpt_path, cache, dev, batch=64, workers=4, split="test",
             max_samples=None):
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model, arch = build_from_ckpt(ck, dev)
    model.eval()
    args = ck.get("args") or {}
    setup = args.get("setup", "s1")
    seed = int(args.get("seed", 0))
    oos = bool(args.get("obs_on_signal", False))

    ds = RadioMapSeer(cache, split, setup, seed=seed + 2000, obs_on_signal=oos)
    if max_samples:
        ds.rows = ds.rows[:max_samples]
    ld = DataLoader(ds, batch_size=batch, shuffle=False,
                    num_workers=workers, pin_memory=True)

    fine = _blank_regions()
    coarse = _blank_regions()
    band = {k: [0.0, 0.0, 0] for k in BAND_LABELS}
    nb = len(BAND_LABELS)

    with torch.no_grad():
        for b in ld:
            obs = b["obs"].to(dev)
            env = b["env"].to(dev)
            mask = b["mask"].to(dev)
            tgt = b["target"].to(dev).float()
            allowed = b["allowed"]

            if arch == "single":
                p = model(torch.cat([obs, env], dim=1)).clamp(0, 1)
                c = p
            else:
                c, p, _ = model(obs, env, mask)
                c = c.clamp(0, 1)
                p = p.clamp(0, 1)

            regs = {
                "all": torch.ones_like(mask),
                "constr": (1.0 - allowed.to(dev)).clamp(0, 1),
                "obs": mask,
            }
            for k, reg in regs.items():
                npx = float(reg.sum())
                if npx == 0:
                    continue
                for store, pred in ((fine, p), (coarse, c)):
                    diff = (pred - tgt) * reg
                    store[k][0] += float(diff.pow(2).sum())
                    store[k][1] += float((tgt * reg).pow(2).sum())
                    store[k][2] += int(npx)

            # ---- distance-band profile inside the blind region -----------
            if regs["constr"].sum() > 0:
                a_all = allowed.cpu().numpy()[:, 0] > 0.5
                err = ((p - tgt) ** 2).cpu().numpy()[:, 0]
                g2 = (tgt ** 2).cpu().numpy()[:, 0]
                for i in range(a_all.shape[0]):
                    d = distance_transform_edt(~a_all[i])
                    sel = ~a_all[i]
                    if not sel.any():
                        continue
                    dd = d[sel]
                    ee = err[i][sel]
                    gg = g2[i][sel]
                    idx = np.digitize(dd, BINS[1:-1])
                    for j in range(nb):
                        m = idx == j
                        if m.any():
                            band[BAND_LABELS[j]][0] += float(ee[m].sum())
                            band[BAND_LABELS[j]][1] += float(gg[m].sum())
                            band[BAND_LABELS[j]][2] += int(m.sum())

    def pack(store):
        out = {}
        for k, (se, sq, npx) in store.items():
            if npx == 0:
                continue
            out[f"{k}_nmse"] = se / max(sq, 1e-12)
            out[f"{k}_rmse"] = (se / npx) ** 0.5
        return out

    bands = {}
    for k, (se, sq, npx) in band.items():
        if npx:
            bands[k] = {"rmse": (se / npx) ** 0.5,
                        "nmse": se / max(sq, 1e-12),
                        "px": npx}

    return {
        "tag": os.path.basename(os.path.dirname(ckpt_path)),
        "setup": setup,
        "best_epoch": ck.get("epoch"),
        "arch": arch,
        "afm_mode": args.get("afm_mode"),
        "gcn_mode": args.get("gcn_mode"),
        "no_residual": args.get("no_residual", False),
        "n_samples": fine["all"][2] // (256 * 256) if fine["all"][2] else 0,
        "fine": pack(fine),
        "coarse": pack(coarse),
        "bands": bands,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True,
                    help="run names under --runs, or ALL for every run with best.pt")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--ckpt", default="best.pt",
                    help="best.pt or last.pt；旧 run 的 best 是按错误的验证集"
                         "选出来的，可能需要和最后一轮对比")
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", default="reeval.json")
    ap.add_argument("--append", action="store_true",
                    help="merge into existing --out instead of overwriting")
    a = ap.parse_args()

    dev = torch.device("cuda:" + a.gpu if torch.cuda.is_available() else "cpu")

    tags = a.tags
    if tags == ["ALL"]:
        tags = sorted(os.path.basename(os.path.dirname(p))
                      for p in glob.glob(os.path.join(a.runs, "*", a.ckpt)))

    results = []
    if a.append and os.path.exists(a.out):
        with open(a.out) as fh:
            results = json.load(fh)
        done = {r["tag"] for r in results}
    else:
        done = set()

    print("%-22s %5s %9s %9s %9s" %
          ("tag", "ep", "NMSE", "RMSE", "constrRMSE"), flush=True)
    for tag in tags:
        if tag in done:
            print("%-22s  已完成，跳过" % tag, flush=True)
            continue
        ckpt = os.path.join(a.runs, tag, a.ckpt)
        if not os.path.exists(ckpt):
            print("%-22s  缺失" % tag, flush=True)
            continue
        r = eval_run(ckpt, a.cache, dev, batch=a.batch, workers=a.workers,
                     split=a.split, max_samples=a.max_samples)
        results.append(r)
        f = r["fine"]
        print("%-22s %5s %9.5f %9.5f %9.5f" %
              (tag, r["best_epoch"], f.get("all_nmse", float("nan")),
               f.get("all_rmse", float("nan")),
               f.get("constr_rmse", float("nan"))), flush=True)
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=1)

    print("[ok] wrote", a.out, f"({len(results)} runs)")


if __name__ == "__main__":
    main()
