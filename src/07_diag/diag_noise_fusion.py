# -*- coding: utf-8 -*-
"""Fig.14 第二轮诊断：哪种融合方式抗冲激噪声（全部用手头已训好的 s3 权重）。

为什么需要这个脚本
------------------
第一轮（diag_noise_graph.py）已经排除了两个嫌疑：
  - GCN 精修网不是放大器：coarse 单独就劣化 x11，dynamic/static/none 完全相同；
  - 手工切换 AFM 激活是**无效实验**：sigmoid/spatial 在干净输入下 NMSE 就是
    0.39/0.14（模型废掉），因为门控激活是和网络联合学习的，推理时换掉等于打坏模型。

于是改用"已经用各自门控训练过"的权重来比 —— 通道消融那批实验正好在 s3 上齐备：
    txf_mffgcn_s3       softmax  论文主路径（当前复现）
    txf_ab_sigmoid_s3   sigmoid  逐通道独立门控
    txf_ab_channel_s3   se       通道注意力（不加权观测支路）
    txf_ab_add_s3       add      直接相加（最弱融合）
    txf_ab_concat_s3    concat   直接拼接
    txf_ab_cross_s3     cross    交叉注意力（query 来自观测支路）
    txf_single_s3       single   早期融合单编码器（消融对照）
    bl_radiounet_s3     radio    RadioUNet（跨架构对照）

关注的是"劣化倍数"而不是绝对值：每个模型自己的 0% 基线不同，只有倍数可比。
论文口径（paper_text.txt:1150）：1% 噪声下 MFFGCN +145%、RadioUNet +160%、
RME-GAN +182%，即 MFFGCN 应为**最抗噪**的那个。

用法（在 src/ 下）
    python diag_noise_fusion.py --n 60 --gpu 0
    python diag_noise_fusion.py --n 60 --gpu 0 --tags txf_mffgcn_s3 txf_ab_cross_s3
输出：控制台表格 + ../results/diag_noise_fusion.json
"""
import argparse
import json
import os

import numpy as np
import torch

from dataset import RadioMapSeer
from evaluate import build_from_ckpt
from baselines import build_baseline

IMG = 256
LEVELS = (0.0, 0.001, 0.01)

DEFAULT_TAGS = [
    ("MFFGCN-softmax", "txf_mffgcn_s3",     "mffgcn"),
    # spatial：论文 eq.(4) 说 W 是 "weight matrix"，暗示在 H×W 空间维归一化；
    # 主路径用的是通道维 softmax(dim=1)。这是唯一有文本依据的实现分叉点，
    # 重训这一个点就是为了验证它是不是 Fig.14「结论相反」的根因。
    ("MFFGCN-spatial", "txf_ab_spatial_s3", "mffgcn"),
    ("MFFGCN-sigmoid", "txf_ab_sigmoid_s3", "mffgcn"),
    ("MFFGCN-se",      "txf_ab_channel_s3", "mffgcn"),
    ("MFFGCN-cross",   "txf_ab_cross_s3",   "mffgcn"),
    ("MFFGCN-add",     "txf_ab_add_s3",     "mffgcn"),
    # txf_ab_concat_s3 的权重还在，但当前 models.py 的 AFM.__init__ 已经不接受
    # "concat"（会 raise ValueError），重建就崩，所以不放进扫描表。
    ("MFFGCN-nogcn",   "txf_ab_nogcn_s3",   "mffgcn"),
    ("MFFGCN-static",  "txf_ab_static_s3",  "mffgcn"),
    ("MFFGCN-nores",   "txf_ab_nores_s3",   "mffgcn"),
    ("SingleEncoder",  "txf_single_s3",     "single"),
    ("RadioUNet",      "bl_radiounet_s3",   "radiounet"),
]


def load(tag, kind, device):
    p = "runs/%s/best.pt" % tag
    if not os.path.exists(p):
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if kind in ("mffgcn", "single"):
        m, arch = build_from_ckpt(ck, device)
        return m.eval(), arch
    m = build_baseline(kind, ae_skip=(ck.get("args") or {}).get("ae_skip", "none"))
    m.load_state_dict(ck["model"])
    if kind == "radiounet":
        m.stage = 2
    return m.eval().to(device), kind


def forward(model, kind, obs, env, mask):
    # single 是早期融合：把 [obs, env] 拼成 3 通道送进单条编码路径，
    # 前向签名是 forward(x)，不能按双输入调用（evaluate.py:141 同一写法）
    if kind == "single":
        return model(torch.cat([obs, env], dim=1))
    if kind == "mffgcn":
        _, y, _ = model(obs, env, mask)
        return y
    out = model(obs, env)
    return out[-1] if isinstance(out, (tuple, list)) else out


class Acc:
    def __init__(self):
        self.se = 0.0
        self.sq = 0.0

    def add(self, y, tgt):
        y = y.clamp(0, 1)
        self.se += float(((y - tgt) ** 2).sum())
        self.sq += float((tgt ** 2).sum())

    @property
    def nmse(self):
        return self.se / max(self.sq, 1e-12)


def noise(bld, p, rng):
    x = bld.clone()
    n = int(round(p * x.numel()))
    flat = x.view(-1)
    idx = torch.from_numpy(rng.choice(x.numel(), size=n, replace=False))
    flat[idx] = 1.0 - flat[idx]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="s3")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tags", nargs="*", default=None,
                    help="只测指定 tag（默认测整张消融表）")
    ap.add_argument("--out", default="../results/diag_noise_fusion.json")
    args = ap.parse_args()

    dev = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    ds = RadioMapSeer("../data/cache", "test", args.setup, seed=2024)
    step = max(1, len(ds) // args.n)
    idx = [i for i in range(0, len(ds), step)][:args.n]

    batch = []
    for i in idx:
        b = ds[i]
        batch.append((b["obs"][None].to(dev), b["env"][None].to(dev),
                      b["mask"][None].to(dev), b["target"][None].to(dev)))
    print("[info] n=%d setup=%s" % (len(idx), args.setup), flush=True)

    specs = DEFAULT_TAGS
    if args.tags:
        specs = [s for s in DEFAULT_TAGS if s[1] in args.tags]

    res = {"setup": args.setup, "n": len(idx), "levels": list(LEVELS), "models": {}}
    with torch.no_grad():
        for disp, tag, kind in specs:
            got = load(tag, kind, dev)
            if got is None:
                print("[skip] %s 权重不存在" % tag, flush=True)
                continue
            model, k = got
            acc = {l: Acc() for l in LEVELS}
            for j, i in enumerate(idx):
                obs, env0, mask, tgt = batch[j]
                for l in LEVELS:
                    if l == 0.0:
                        env = env0
                    else:
                        rng = np.random.default_rng([2024 + i * 7919, 31])
                        env = env0.clone()
                        env[:, 0:1] = noise(env0[:, 0:1], l, rng)
                    acc[l].add(forward(model, k, obs, env, mask), tgt)
            res["models"][disp] = {str(l): acc[l].nmse for l in LEVELS}
            print("%-16s 0%%=%.5f  0.1%%=%.5f  1%%=%.5f  x%.2f"
                  % (disp, acc[0.0].nmse, acc[0.001].nmse, acc[0.01].nmse,
                     acc[0.01].nmse / max(acc[0.0].nmse, 1e-12)), flush=True)
            del model
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)

    print("\n==== 按 1%% 噪声劣化倍数排序（越小越抗噪）====")
    rows = [(k, v["0.0"], v["0.01"], v["0.01"] / max(v["0.0"], 1e-12))
            for k, v in res["models"].items()]
    rows.sort(key=lambda r: r[3])
    print("%-16s %10s %10s %8s" % ("model", "clean", "1%", "x"))
    for k, a, b, r in rows:
        print("%-16s %10.5f %10.5f   x%.2f" % (k, a, b, r))
    print("[save] %s" % args.out)


if __name__ == "__main__":
    main()
