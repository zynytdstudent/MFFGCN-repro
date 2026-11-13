"""观测置零探针：测每个模型到底有多依赖观测输入。

对同一个 checkpoint、同一批测试样本，跑三种条件：
  normal     原始输入（应与已存的 test_metrics 接近）
  obs_zero   观测值全部清零，但保留"哪里有传感器"的掩码
  mask_zero  观测值与掩码同时清零（模型完全不知道观测信息）

依赖度 = (该条件 NMSE / normal NMSE - 1) * 100%
依赖度低说明模型主要靠环境先验做预测，退化幅度指标会失真。
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

SRC = "/public2/czz/ZhangYu/project/mffgcn/src"
sys.path.insert(0, SRC)

from dataset import RadioMapSeer
from models import MFFGCN, SingleEncoderUNet
from baselines import DeepAE, RadioUNet, Pix2pixGen


def build(ck, tag, device):
    a = ck.get("args") or {}
    arch = a.get("arch")
    widths = tuple(int(x) for x in str(a.get("widths", "32,64,128,256")).split(","))
    bn = int(a.get("bottleneck", 512))
    if arch == "single":
        m = SingleEncoderUNet(widths=widths, bottleneck=bn)
    elif arch == "mffgcn":
        m = MFFGCN(widths=widths, bottleneck=bn,
                   afm_mode=a.get("afm_mode", "softmax"),
                   gcn_mode=a.get("gcn_mode", "dynamic"),
                   k=int(a.get("k", 16)),
                   residual=not a.get("no_residual", False))
    elif arch == "radiounet":
        m = RadioUNet()
    elif arch == "deepae":
        m = DeepAE()
    elif arch == "pix2pix":
        m = Pix2pixGen()
    elif arch is None:
        m = Pix2pixGen()
    else:
        raise ValueError("unknown arch: %s" % arch)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), (arch or "pix2pix")


def forward(model, arch, obs, env, mask):
    if arch == "single":
        y = model(torch.cat([obs, env], dim=1)).clamp(0, 1)
        return y, y
    if arch == "mffgcn":
        c, f, _ = model(obs, env, mask)
        return c.clamp(0, 1), f.clamp(0, 1)
    y = model(obs, env)
    if isinstance(y, (tuple, list)):
        y = y[-1]
    y = y.clamp(0, 1)
    return y, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--max-samples", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="/tmp/zero_obs.json")
    args = ap.parse_args()

    device = torch.device("cpu")
    results = {}

    for tag in args.tags:
        path = os.path.join(SRC, "runs", tag, "best.pt")
        if not os.path.exists(path):
            print("SKIP %s (no best.pt)" % tag, flush=True)
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model, arch = build(ck, tag, device)
        a = ck.get("args") or {}
        setup = a.get("setup", "s1")
        seed = int(a.get("seed", 0))
        oos = bool(a.get("obs_on_signal", False))

        ds = RadioMapSeer(args.cache, "test", setup, seed=seed + 2000, obs_on_signal=oos)
        if args.max_samples:
            ds.rows = ds.rows[:args.max_samples]
        ld = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers)

        row = {"arch": arch, "setup": setup, "n": len(ds)}
        with torch.no_grad():
            for cond in ("normal", "obs_zero", "mask_zero"):
                se = sq = 0.0
                n = 0
                for b in ld:
                    obs = b["obs"].to(device)
                    env = b["env"].to(device)
                    mask = b["mask"].to(device)
                    tgt = b["target"].to(device)
                    if cond in ("obs_zero", "mask_zero"):
                        obs = torch.zeros_like(obs)
                    if cond == "mask_zero":
                        mask = torch.zeros_like(mask)
                    coarse, fine = forward(model, arch, obs, env, mask)
                    p = torch.nan_to_num(fine, nan=0.0, posinf=1.0, neginf=0.0)
                    se += float(((p - tgt) ** 2).sum())
                    sq += float((tgt ** 2).sum())
                    n += p.shape[0]
                row[cond] = {"nmse": se / max(sq, 1e-12),
                             "rmse": (se / max(n * 256 * 256, 1)) ** 0.5}
        base = row["normal"]["nmse"]
        row["drop_obs_pct"] = (row["obs_zero"]["nmse"] / base - 1) * 100 if base else 0
        row["drop_all_pct"] = (row["mask_zero"]["nmse"] / base - 1) * 100 if base else 0
        results[tag] = row
        print("%-20s normal=%.5f  obs0=%.5f (%+.1f%%)  mask0=%.5f (%+.1f%%)"
              % (tag, base, row["obs_zero"]["nmse"], row["drop_obs_pct"],
                 row["mask_zero"]["nmse"], row["drop_all_pct"]), flush=True)
        del model

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()
