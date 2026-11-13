# -*- coding: utf-8 -*-
"""Fig.14 诊断：MFFGCN 为什么在冲激噪声下比论文脆一个数量级。

实测背景（results/supp_noise_paper.json，--span, Setup 3, n=200）
    1% 冲激噪声：MFFGCN 0.00533 -> 0.07351 (+1279%)
                 RadioUNet 0.00639 -> 0.03983 (+523%)

论文正文（results/paper_text.txt:1150）
    "injecting 1% impulse noise into the building map increases the NMSE of
     MFFGCN by 145%, whereas RadioUNet and RME-GAN surge by 160% and 182%,
     respectively, confirming that MFFGCN is still the most robust"
    即论文里 MFFGCN 最抗噪，我们的实现里它最脆 —— 排序被反转。

为什么断定问题在 MFFGCN 而不是噪声口径
    RadioUNet 是同一份噪声下的控制组：它 0.1% 档劣化 x1.24，按敏感性外推
    与论文 x2.6@1% 基本吻合。同一份噪声下基线正常、只有 MFFGCN 异常，
    所以不是"我们的噪声更重"，而是 MFFGCN 自身的环境通路对稀疏扰动过敏。

本脚本做两组对照，都不需要重训
--------------------------------
A. 图的建法 —— 同一份权重，只换前向时的建图方式
     dynamic  论文主路径：用当前特征建 KNN（默认）
     static   用固定位置编码建 KNN，等价于固定空间邻域
     none     跳过图卷积
   理由：精修网把 256x256 特征切成 16x16 = 256 个节点。655 个孤立噪声像素
   落在不同 patch 上，最多污染 25% 的节点；3 层 k=16 的图卷积再把它扩散到全图。
   若 dynamic 明显比 static 脆，这条 "稀疏扰动 -> 特征漂移 -> 图重连 -> 全图扩散"
   的放大链就成立。

B. 扰动形态 —— 翻转的像素数完全相同
     scattered  随机 655 个孤立像素（论文 "impulse noise" 的字面定义）
     clustered  一个 26x26 连续块（676 px）
   若 scattered 远差于 clustered，说明伤害来自"孤立"而不是"总量"：
   被打散的是 patch/图的拓扑，不是丢失的信息量。

用法（在 src/ 下）
    python diag_noise_graph.py --tag txf_mffgcn_s3 --n 60 --gpu 1
输出：控制台表格 + ../results/diag_noise_graph.json
"""
import argparse
import json
import os

import numpy as np
import torch

from dataset import RadioMapSeer
from models import MFFGCN

IMG = 256
MODES = ("dynamic", "static", "none")
LEVELS = (0.0, 0.001, 0.01)


def build(ck, gcn_mode, device):
    """按 checkpoint 里存的 args 重建结构，只把 gcn_mode 换掉。

    static 只改 forward 里的拓扑构造，参数完全一致，可直接加载。
    none 会**不实例化** gcn 层（GraphRefineNet 里按 gcn_mode 建模块），
    所以 checkpoint 里的 refine.gcn.* 会成为多余键 —— 这时必须 strict=False，
    否则 load_state_dict 直接报错（第一版就崩在这里）。
    被丢掉的只是图卷积那几层，g2i/head 等其余权重照常加载。
    """
    a = ck.get("args") or {}
    widths = tuple(int(x) for x in
                   str(a.get("widths", "32,64,128,256")).split(","))
    m = MFFGCN(widths=widths,
               bottleneck=int(a.get("bottleneck", 512)),
               afm_mode=a.get("afm_mode", "softmax"),
               gcn_mode=gcn_mode,
               k=int(a.get("k", 16)),
               residual=not a.get("no_residual", False))
    if gcn_mode == "none":
        miss = m.load_state_dict(ck["model"], strict=False)
        print("[warn] gcn_mode=none: 丢弃 %d 个图卷积权重"
              % len(miss.unexpected_keys), flush=True)
    else:
        m.load_state_dict(ck["model"])
    return m.eval().to(device)


def noise_scattered(bld, p, rng):
    """随机翻转 p 比例的孤立像素（与 supp_eval.impulse_noise 同一实现）。"""
    x = bld.clone()
    n = int(round(p * x.numel()))
    flat = x.view(-1)
    idx = torch.from_numpy(rng.choice(x.numel(), size=n, replace=False))
    flat[idx] = 1.0 - flat[idx]
    return x


def noise_clustered(bld, n_pix, rng):
    """翻转一个面积约 n_pix 的连续方块，像素总量与 scattered 对齐。"""
    x = bld.clone()
    side = int(round(n_pix ** 0.5))
    r = int(rng.integers(0, IMG - side))
    c = int(rng.integers(0, IMG - side))
    sel = x[..., r:r + side, c:c + side]
    sel.copy_(1.0 - sel)
    return x


class Acc:
    """与 evaluate.py 同口径：NMSE = Σse / Σtgt²，输出 clamp 到 [0,1]。"""

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="txf_mffgcn_s3", help="runs/<tag>/best.pt")
    ap.add_argument("--setup", default="s3", help="数据集 split 用的 setup")
    ap.add_argument("--n", type=int, default=60, help="样本数（等间隔铺满测试集）")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="../results/diag_noise_graph.json")
    args = ap.parse_args()

    dev = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    ck = torch.load("runs/%s/best.pt" % args.tag, map_location="cpu",
                    weights_only=False)

    ds = RadioMapSeer("../data/cache", "test", args.setup, seed=2024)
    # 等间隔铺满整个测试集，避免只取前 n 张造成的 E[t²] 偏差
    step = max(1, len(ds) // args.n)
    idx = [i for i in range(0, len(ds), step)][:args.n]
    print("[info] tag=%s n=%d setup=%s" % (args.tag, len(idx), args.setup),
          flush=True)

    # 先一次性把样本读出来，三种建图方式用完全相同的输入
    batch = []
    for i in idx:
        b = ds[i]
        batch.append((b["obs"][None].to(dev), b["env"][None].to(dev),
                      b["mask"][None].to(dev), b["target"][None].to(dev)))

    res = {"tag": args.tag, "setup": args.setup, "n": len(idx),
           "levels": list(LEVELS), "graph": {}, "shape": {}}

    with torch.no_grad():
        # ---------- A. 建图方式 x 噪声强度 ----------
        # 同时记录 coarse（U-Net 粗估计）与 fine（GCN 精修后）：
        # 若 coarse 已经崩、fine 只是跟着崩，说明问题在双编码器/AFM 那一级，
        # 而不是精修网。这是区分"哪一级放大噪声"最快的一刀。
        for mode in MODES:
            model = build(ck, mode, dev)
            acc = {(l, s): Acc() for l in LEVELS for s in ("fine", "coarse")}
            for k, i in enumerate(idx):
                obs, env0, mask, tgt = batch[k]
                for l in LEVELS:
                    if l == 0.0:
                        env = env0
                    else:
                        rng = np.random.default_rng([2024 + i * 7919, 31])
                        env = env0.clone()
                        env[:, 0:1] = noise_scattered(env0[:, 0:1], l, rng)
                    coarse, y, _ = model(obs, env, mask)
                    acc[(l, "fine")].add(y, tgt)
                    acc[(l, "coarse")].add(coarse, tgt)
            res["graph"][mode] = {
                "fine": {str(l): acc[(l, "fine")].nmse for l in LEVELS},
                "coarse": {str(l): acc[(l, "coarse")].nmse for l in LEVELS},
            }
            print("[graph:%s] fine   " % mode +
                  "  ".join("%.5f@%.1f%%" % (acc[(l, "fine")].nmse, l * 100)
                            for l in LEVELS), flush=True)
            print("[graph:%s] coarse " % mode +
                  "  ".join("%.5f@%.1f%%" % (acc[(l, "coarse")].nmse, l * 100)
                            for l in LEVELS), flush=True)
            del model
            torch.cuda.empty_cache()

        # ---------- C. AFM 门控方式（同一份权重，只换激活函数） ----------
        # 论文 eq.(4) 只写 "a Softmax activation function to produce a weight
        # matrix W with values between 0 and 1"，没说是沿哪个维度归一化。
        # 三种读法都只用 self.gate 这一个卷积，参数完全相同，可以原地切换：
        #   softmax  沿通道维（我们现在的主路径）—— C=256 时对离群值极敏感
        #   sigmoid  逐通道独立 0~1，不归一化
        #   spatial  沿 H*W 归一化，sum=1
        # 若 softmax 明显比另外两种脆，说明门控的归一化维度读错了，
        # 这正好能解释"论文里 MFFGCN 最抗噪、我们最脆"的排序反转。
        model = build(ck, "dynamic", dev)
        res["afm"] = {}
        for am in ("softmax", "sigmoid", "spatial"):
            for mod in model.fusion.afm:
                mod.mode = am
            model.fusion.afm_b.mode = am
            acc = {l: Acc() for l in LEVELS}
            for k, i in enumerate(idx):
                obs, env0, mask, tgt = batch[k]
                for l in LEVELS:
                    if l == 0.0:
                        env = env0
                    else:
                        rng = np.random.default_rng([2024 + i * 7919, 31])
                        env = env0.clone()
                        env[:, 0:1] = noise_scattered(env0[:, 0:1], l, rng)
                    _, y, _ = model(obs, env, mask)
                    acc[l].add(y, tgt)
            res["afm"][am] = {str(l): acc[l].nmse for l in LEVELS}
            print("[afm:%s] " % am +
                  "  ".join("%.5f@%.1f%%" % (acc[l].nmse, l * 100)
                            for l in LEVELS), flush=True)
        del model
        torch.cuda.empty_cache()

        # ---------- B. 扰动形态（1% 档，像素数对齐） ----------
        model = build(ck, "dynamic", dev)
        n_pix = int(round(0.01 * IMG * IMG))
        acc_s = Acc()
        acc_c = Acc()
        for k, i in enumerate(idx):
            obs, env0, mask, tgt = batch[k]
            rng = np.random.default_rng([2024 + i * 7919, 31])
            env_s = env0.clone()
            env_s[:, 0:1] = noise_scattered(env0[:, 0:1], 0.01, rng)
            rng2 = np.random.default_rng([2024 + i * 7919, 31])
            env_c = env0.clone()
            env_c[:, 0:1] = noise_clustered(env0[:, 0:1], n_pix, rng2)
            _, ys, _ = model(obs, env_s, mask)
            _, yc, _ = model(obs, env_c, mask)
            acc_s.add(ys, tgt)
            acc_c.add(yc, tgt)
        res["shape"] = {"scattered": acc_s.nmse, "clustered": acc_c.nmse,
                        "pixels": n_pix}
        print("[shape] scattered=%.5f  clustered=%.5f (each %d px)"
              % (acc_s.nmse, acc_c.nmse, n_pix), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)

    # ---------- 汇总 ----------
    print("\n==== 相对干净输入的劣化倍数 ====")
    print("mode       coarse@0%  coarse@1%   x    fine@0%   fine@1%    x")
    for m in MODES:
        c = res["graph"][m]["coarse"]
        g = res["graph"][m]["fine"]
        print("%-9s  %.5f    %.5f  x%.1f   %.5f   %.5f  x%.1f"
              % (m, c["0.0"], c["0.01"], c["0.01"] / max(c["0.0"], 1e-12),
                 g["0.0"], g["0.01"], g["0.01"] / max(g["0.0"], 1e-12)))
    print("[shape] scattered=%.5f  clustered=%.5f (each %d px)"
          % (res["shape"]["scattered"], res["shape"]["clustered"],
             res["shape"]["pixels"]))
    print("[save] %s" % args.out)


if __name__ == "__main__":
    main()
