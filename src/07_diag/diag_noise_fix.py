# -*- coding: utf-8 -*-
"""Fig.14 第三轮诊断：噪声到底伤在哪一步，能不能在推理侧补回来。

前两轮已经把三组结构假设全部否定（GCN 建图 / AFM 激活 / 归一化维度），
spatial 重训后 1% 噪声劣化 ×18.0，比通道维的 ×13.8 更脆。于是换问题：
既然结构不是根因，那 **输入侧** 能不能解释？

本脚本只做一件事——同一份 txf_mffgcn_s3 权重（no-res 也不要，就用主路径），
在推理时对加噪后的 env 图做不同处理，看 1% 档 NMSE 能不能回来：

    clean            干净输入（基线）
    flip_1pct        1% 像素 0<->1 随机翻转（与前两轮口径一致，×13.8 那条）
    flip_med3        flip 后对 env 做 3x3 中值滤波再送网（推理侧去噪）
    erase_1pct       只把 1% 的建筑像素 1->0（缺口）
    add_1pct         只把 1% 的空白像素 0->1（假建筑）

为什么要拆 erase/add：前一轮已证明「伤害来自孤立而非面积」，但没分清是
「多出来的假边缘」致命，还是「缺掉的边缘」致命。这两者对第一层卷积核的
意义完全相反，是选下一步方向的依据。

为什么中值滤波值得试：655 个孤立翻转点在 3x3 中值滤波下会被邻域投票抹掉
（孤立点邻域里同值像素占绝对多数），而真实建筑边缘是成片的、不会被抹掉。
如果 med3 能把 NMSE 从 0.075 拉回 0.02 量级，说明缺口在输入预处理，
不在模型结构——也就不必再改网络。

用法（在 src/ 下）
    python diag_noise_fix.py --n 60 --gpu 0
输出：控制台表格 + ../results/diag_noise_fix.json
"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from dataset import RadioMapSeer
from evaluate import build_from_ckpt
from baselines import build_baseline

TAG = "txf_mffgcn_s3"
CH = 0  # env 的建筑通道


def flips(x, p, rng, mode="both"):
    """按比例翻转建筑通道像素。mode=both 双向；erase 只 1->0；add 只 0->1。"""
    dev = x.device
    x = x.detach().cpu().clone()
    flat = x.view(-1)
    n = int(round(p * flat.numel()))
    if mode == "both":
        idx = torch.from_numpy(rng.choice(flat.numel(), size=n, replace=False))
    elif mode == "erase":
        pool = (flat > 0.5).nonzero().view(-1).numpy()
        idx = torch.from_numpy(rng.choice(pool, size=min(n, len(pool)),
                                         replace=False))
    elif mode == "add":
        pool = (flat < 0.5).nonzero().view(-1).numpy()
        idx = torch.from_numpy(rng.choice(pool, size=min(n, len(pool)),
                                         replace=False))
    else:
        raise ValueError(mode)
    flat[idx] = 1.0 - flat[idx]
    return x.to(dev)


def med3(env):
    """逐通道 3x3 中值滤波（torch 没有现成 median_pool，用 unfold 拼）。"""
    k = 3
    pad = F.pad(env, (1, 1, 1, 1), mode="replicate")
    win = pad.unfold(2, k, 1).unfold(3, k, 1)      # B,C,H,W,3,3
    win = win.contiguous().view(env.shape[0], env.shape[1], env.shape[2],
                                env.shape[3], k * k)
    return win.median(dim=-1).values


def load(tag, kind, device):
    if kind in ("mffgcn", "single"):
        ck = torch.load("runs/%s/best.pt" % tag, map_location="cpu",
                        weights_only=False)
        m, _ = build_from_ckpt(ck, device)
        return m.eval(), kind
    ck = torch.load("runs/%s/best.pt" % tag, map_location="cpu",
                    weights_only=False)
    m = build_baseline(kind,
                       ae_skip=(ck.get("args") or {}).get("ae_skip", "none"))
    m.load_state_dict(ck["model"])
    if kind == "radiounet":
        m.stage = 2
    return m.eval().to(device), kind


def forward(model, kind, obs, env, mask):
    if kind == "single":
        return model(torch.cat([obs, env], dim=1))
    if kind == "mffgcn":
        _, y, _ = model(obs, env, mask)
        return y
    out = model(obs, env)
    return out[-1] if isinstance(out, (tuple, list)) else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="s3")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--kind", default="mffgcn",
                    choices=["mffgcn", "single", "radiounet", "unet"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available()
                       else "cpu")
    model, kind = load(args.tag, args.kind, dev)

    ds = RadioMapSeer("../data/cache", "test", args.setup, seed=2024)
    step = max(1, len(ds) // args.n)
    idx = [i for i in range(0, len(ds), step)][:args.n]

    cases = ["clean", "flip_1pct", "flip_med3", "erase_1pct", "add_1pct"]
    se = {c: 0.0 for c in cases}
    sq = 0.0
    with torch.no_grad():
        for j, i in enumerate(idx):
            b = ds[i]
            obs = b["obs"][None].to(dev)
            env0 = b["env"][None].to(dev)
            mask = b["mask"][None].to(dev)
            tgt = b["target"][None].to(dev)

            envs = {"clean": env0}
            for c, mode in (("flip_1pct", "both"), ("flip_med3", "both"),
                            ("erase_1pct", "erase"), ("add_1pct", "add")):
                g = np.random.default_rng(1000 + i)
                e = env0.clone()
                e[:, CH:CH + 1] = flips(e[:, CH:CH + 1], 0.01, g, mode)
                envs[c] = med3(e) if c == "flip_med3" else e

            for c in cases:
                y = forward(model, kind, obs, envs[c], mask).clamp(0, 1)
                se[c] += float(((y - tgt) ** 2).sum())
            sq += float((tgt ** 2).sum())
            if (j + 1) % 20 == 0:
                print("[%d/%d]" % (j + 1, len(idx)), flush=True)

    base = se["clean"] / sq
    print("\n%-12s %10s %8s" % ("case", "NMSE", "vs clean"))
    out = {"setup": args.setup, "n": len(idx), "tag": args.tag,
           "kind": kind, "nmse": {}}
    for c in cases:
        v = se[c] / sq
        out["nmse"][c] = v
        print("%-12s %10.5f %8.2fx" % (c, v, v / base))
    out_path = args.out or ("../results/diag_noise_fix_%s.json"
                            % args.tag.replace("_s3", ""))
    json.dump(out, open(out_path, "w"), indent=1)
    print("[save]", out_path)


if __name__ == "__main__":
    main()
