"""诊断：模型是否真的在使用稀疏观测？

对同一个已训练 checkpoint，在测试集上比较三种输入的误差：
  real    : 正常观测
  zero    : 把观测全部置 0（模型若依赖观测，误差应显著变差）
  shuffled: 换成别的样本的观测（误差若与 real 接近，说明模型在忽略观测）
另给出 trivial 基线：直接输出全 0。

用法：
  python diag_obs.py --ckpt runs/calib_single/best.pt --n 200 --split test --setup s1
"""
import argparse
import json
import os

import numpy as np
import torch

from dataset import RadioMapSeer, IMG
from models import MFFGCN, SingleEncoderUNet


def build_from_ckpt(ck):
    a = ck.get("args", {})
    widths = tuple(int(x) for x in str(a.get("widths", "32,64,128,256")).split(","))
    bn = int(a.get("bottleneck", 512))
    arch = a.get("arch", "single")
    if arch == "single":
        m = SingleEncoderUNet(widths=widths, bottleneck=bn)
    else:
        m = MFFGCN(widths=widths, bottleneck=bn, k=int(a.get("k", 16)),
                   afm_mode=a.get("afm_mode", "softmax"),
                   gcn_mode=a.get("gcn_mode", "dynamic"),
                   residual=not a.get("no_residual", False))
    m.load_state_dict(ck["model"])
    m.eval()
    return m, arch, a


def rmse(pred, tgt):
    return float(torch.sqrt(torch.mean((pred - tgt) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--split", default="test")
    ap.add_argument("--setup", default="s1")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()

    dev = torch.device("cuda:" + a.gpu if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model, arch, targs = build_from_ckpt(ck)
    model.to(dev)
    print("ckpt=%s arch=%s obs_on_signal=%s best_ep=%s" %
          (a.ckpt, arch, targs.get("obs_on_signal"), ck.get("epoch")))

    ds = RadioMapSeer(a.cache, a.split, a.setup, seed=a.seed,
                      obs_on_signal=False, augment=False)
    n = min(a.n, len(ds))
    idx = np.linspace(0, len(ds) - 1, n).astype(int)

    acc = {"real": [], "zero": [], "shuf": [], "trivial": []}
    preds_real, preds_zero = [], []

    with torch.no_grad():
        for j, i in enumerate(idx):
            b = ds[int(i)]
            obs = b["obs"][None].to(dev)
            env = b["env"][None].to(dev)
            mask = b["mask"][None].to(dev)
            tgt = b["target"][None].to(dev)

            # 打乱：用下一个样本的观测
            ob = ds[int(idx[(j + 1) % n])]
            obs_shuf = ob["obs"][None].to(dev)

            def f(o):
                if arch == "single":
                    return model(torch.cat([o, env], dim=1))
                c, fi, _ = model(o, env, mask)
                return fi

            p_real = f(obs)
            p_zero = f(torch.zeros_like(obs))
            p_shuf = f(obs_shuf)

            acc["real"].append(rmse(p_real, tgt))
            acc["zero"].append(rmse(p_zero, tgt))
            acc["shuf"].append(rmse(p_shuf, tgt))
            acc["trivial"].append(rmse(torch.zeros_like(tgt), tgt))
            preds_real.append(p_real.flatten()[::97].cpu())
            preds_zero.append(p_zero.flatten()[::97].cpu())

    out = {k: float(np.mean(v)) for k, v in acc.items()}
    pr = torch.cat(preds_real).numpy()
    pz = torch.cat(preds_zero).numpy()
    corr = float(np.corrcoef(pr, pz)[0, 1]) if pr.std() > 0 and pz.std() > 0 else float("nan")

    print(json.dumps({"rmse": out,
                      "pred_corr_real_vs_zero": round(corr, 5),
                      "n": n}, indent=2))
    print("\n判读：")
    print("  zero/real 比 > 1.5  -> 模型确实依赖观测")
    print("  zero/real ≈ 1.0     -> 模型基本忽略观测（管线或输入编码有问题）")
    print("  shuf/real 明显 < zero/real 且≈1 -> 观测位置信息未被利用")


if __name__ == "__main__":
    main()
