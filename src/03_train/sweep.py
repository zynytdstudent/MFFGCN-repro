"""sweep.py - 用修复后的 dataset.py 批量重评所有已有 checkpoint。

背景：dataset.py 曾用 split 内位置索引 tx_coords（全局行号），
导致 val/test 的发射机位置错误。checkpoint 本身是好的，
只需用修好的代码重评一次即可重建消融表，不必重训。

注意：因为 val 指标当时是错的，不少 run 的 best.pt 落在第 1 个 epoch
（几乎没训练）。所以默认同时评 best.pt 和 last.pt：last.pt 是 50 epoch
训练结束的模型，论文表格应以它为准；best.pt 仅作参考。

特性：
  * 固定评估随机种子（默认 2024），保证各模型的测试观测完全一致
  * 断点续跑（已写入 eval_sweep_<split>.jsonl 的记录自动跳过）
  * --manifest-only 只打印 checkpoint 元信息，不跑评估

用法：
  python sweep.py --manifest-only
  python sweep.py --gpu 0 --batch 32
  python sweep.py --gpu 0 --tags calib_single bs15_s1 --ckpt both
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import RadioMapSeer
from evaluate import build_from_ckpt

# 训练早期的调试 / 诊断目录，不属于论文表格
SKIP_PREFIX = ("exp", "fix", "s1_", "s2_", "s3_", "ctrl_", "D1_", "D2_",
               "smoke", "sig_", "txf_", "pair_")
DEFAULT_INCLUDE = ("ab_", "calib_", "FINAL_", "afm_")


def discover(runs, include, tags=None):
    out = []
    for d in sorted(os.listdir(runs)):
        if tags:
            if d in tags and os.path.isfile(os.path.join(runs, d, "best.pt")):
                out.append(d)
            continue
        if not os.path.isdir(os.path.join(runs, d)):
            continue
        if not os.path.isfile(os.path.join(runs, d, "best.pt")):
            continue
        if d.startswith(SKIP_PREFIX):
            continue
        if d.startswith(tuple(include)) or d == "bs15_s1":
            out.append(d)
    return out


def load_meta(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args") or {}
    meta = dict(
        epoch=ck.get("epoch"),
        arch=a.get("arch", "mffgcn"),
        setup=a.get("setup", "s1"),
        afm=a.get("afm_mode", "softmax"),
        gcn=a.get("gcn_mode", "dynamic"),
        k=a.get("k", 16),
        residual=not a.get("no_residual", False),
        obs_on_signal=bool(a.get("obs_on_signal", False)),
        seed=a.get("seed", 0),
    )
    return ck, meta


@torch.no_grad()
def evaluate(ck, meta, dev, cache, split, eval_seed, batch, workers):
    model, arch = build_from_ckpt(ck, dev)
    model.eval()
    n_par = sum(p.numel() for p in model.parameters())
    ds = RadioMapSeer(cache, split, meta["setup"], seed=eval_seed,
                      obs_on_signal=meta["obs_on_signal"])
    ld = DataLoader(ds, batch_size=batch, shuffle=False,
                    num_workers=workers, pin_memory=True)
    se = sq = sc = 0.0
    n = 0
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
        n += tgt.numel()
    del model
    return dict(nmse=se / sq, rmse=(se / n) ** 0.5,
                coarse_nmse=sc / sq, coarse_rmse=(sc / n) ** 0.5,
                n_samples=len(ds), params_m=round(n_par / 1e6, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--split", default="test")
    ap.add_argument("--eval-seed", type=int, default=2024)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tags", nargs="+", default=None)
    ap.add_argument("--include", nargs="+", default=list(DEFAULT_INCLUDE))
    ap.add_argument("--ckpt", default="both", choices=["best", "last", "both"])
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tags = discover(a.runs, a.include, a.tags)
    which = ["best", "last"] if a.ckpt == "both" else [a.ckpt]
    jobs = []
    for t in tags:
        for w in which:
            if os.path.isfile(os.path.join(a.runs, t, w + ".pt")):
                jobs.append((t, w))

    out_path = a.out or ("eval_sweep_%s.jsonl" % a.split)
    done = set()
    best_rec = {}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done.add((r["tag"], r.get("which", "best")))
                best_rec[(r["tag"], r.get("which", "best"))] = r

    if a.manifest_only:
        print("%-20s %-7s %-4s %-8s %-8s %5s %5s %6s  %s" %
              ("tag", "arch", "setup", "afm", "gcn", "ep", "seed", "obsSig",
               "present"))
        for t in tags:
            _, m = load_meta(os.path.join(a.runs, t, "best.pt"))
            pres = "".join(w[0] for w in ("best", "last")
                           if os.path.isfile(os.path.join(a.runs, t, w + ".pt")))
            print("%-20s %-7s %-4s %-8s %-8s %5s %5s %6d  %s" %
                  (t, m["arch"], m["setup"], m["afm"], m["gcn"],
                   m["epoch"], m["seed"], m["obs_on_signal"], pres))
        print("runs:", len(tags), " jobs:", len(jobs),
              " already:", len(done))
        return

    dev = torch.device("cuda:" + a.gpu if torch.cuda.is_available() else "cpu")
    print("[info] device=%s split=%s eval_seed=%d batch=%d jobs=%d"
          % (dev, a.split, a.eval_seed, a.batch, len(jobs)), flush=True)
    print("%-18s %-4s %4s %8s %8s %9s %6s %5s" %
          ("tag", "ck", "ep", "NMSE", "RMSE", "cNMSE", "Mpar", "sec"),
          flush=True)
    t_all = time.time()
    for t, w in jobs:
        if (t, w) in done:
            r = best_rec[(t, w)]
            print("%-18s %-4s %4s %8.4f %8.4f %9.4f %6s %5s 跳过" %
                  (t, w, r.get("epoch"), r["nmse"], r["rmse"],
                   r.get("coarse_nmse", 0), r.get("params_m", "-"), "-"),
                  flush=True)
            continue
        t0 = time.time()
        try:
            ck, meta = load_meta(os.path.join(a.runs, t, w + ".pt"))
            res = evaluate(ck, meta, dev, a.cache, a.split,
                           a.eval_seed, a.batch, a.workers)
        except Exception as exc:
            print("%-18s %-4s 失败: %s" % (t, w, exc), flush=True)
            continue
        dt = time.time() - t0
        with open(out_path, "a") as fh:
            fh.write(json.dumps(dict(tag=t, which=w, split=a.split,
                                     eval_seed=a.eval_seed,
                                     elapsed_s=round(dt, 1),
                                     **meta, **res),
                                ensure_ascii=False) + "\n")
        print("%-18s %-4s %4s %8.4f %8.4f %9.4f %6.2f %5.0f" %
              (t, w, meta["epoch"], res["nmse"], res["rmse"],
               res["coarse_nmse"], res["params_m"], dt), flush=True)
        del ck
        torch.cuda.empty_cache()
    print("[ok] %.1f 分钟 -> %s" % ((time.time() - t_all) / 60, out_path))


if __name__ == "__main__":
    main()
