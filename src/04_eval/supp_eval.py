# -*- coding: utf-8 -*-
"""补充实验统一入口：缺失建筑 / 环境图降分辨率 / 盲区形状 / 推理时延。

为什么要单独写这一个脚本，而不是复用 paper_fig_16.py：
    1) 旧脚本每调一次 `ds[i]` 都会重抽观测点，多个配置之间不是同一批数据，
       样本数一少指标就被"这次抽到几个观测点"主导（实测同一批 20 个样本能差 4 倍）。
       这里改成**先读一次样本、再在内存里对同一份张量做不同扰动**，配置之间严格配对。
    2) 旧的"抹掉 N 栋建筑"按连通域面积从大到小删，而 RadioMapSeer 里相邻建筑会粘成
       一个大团（最大的 6462 px = 全图 9.9%，典型建筑只有 300–600 px）。
       于是"1 栋缺失"实际抹掉 5.2% 的图，指标从 0.0038 崩到 0.0808 —— 和论文
       Fig.16 曲线形状对不上，不是模型不行，是扰动强度差了 10 倍。
       这里改成**随机挑 N 个连通域**，并同时记录实际抹除面积占比。

四个模式（--mode）：
    missing   缺失建筑 1–4 栋（对应论文 Fig.16，Setup 3）
    degrade   环境图降分辨率，下采样-上采样后**重新二值化**（论文 Fig.15，Setup 3）
    noise     环境图脉冲噪声 0.1%–2%（论文 Fig.14，Setup 3）
    shape     盲区形状 条带/圆形/三角形（对应论文表 VI 零样本泛化，s2 权重）
    latency   推理时延 / 参数量 / FLOPs（论文表 VIII）

论文口径（已核对原文，见 results/paper_text.txt）：
    - Fig.14/15/16 都明确写在 Setup 3 下做，不是 Setup 1
    - Fig.15 原文："downsample and upsample ... and then processed the new map
      back to a binary mask through the threshold" —— 退化后要重新二值化
    - 表 VI 原文："trained on rectangular constrained regions under circles
      and triangles" —— 形状是圆和三角，不是圆和"不规则"

统一约定：
    - 全部走 evaluate.py 的口径：model.eval()、输出 clamp(0,1)、NMSE = Σse/Σtgt²
    - 一次读样本、多配置复用，配置之间配对可比
    - 结果写 ../results/supp_<mode>.json，图写 ../figs/supp_<mode>.png

用法（在 src/ 下）：
    python supp_eval.py --mode missing --n 200 --gpu 1
    python supp_eval.py --mode degrade --n 200 --gpu 1
    python supp_eval.py --mode noise   --n 200 --gpu 1
    python supp_eval.py --mode shape   --n 120 --gpu 1
    python supp_eval.py --mode latency --gpu 1
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from dataset import RadioMapSeer
from evaluate import build_from_ckpt
from baselines import build_baseline

IMG = 256

# (显示名, tag 模板, baselines 的 kind)
MODELS = [
    ("MFFGCN",   "txf_mffgcn_{s}",   "mffgcn"),
    ("RadioUNet", "bl_radiounet_{s}", "radiounet"),
    ("RME-GAN",  "bl_rmegan_{s}",     "pix2pix"),
]

# ----------------------------------------------------------------------
# 取值网格
# ----------------------------------------------------------------------
# 默认网格：范围拉得更宽，用来看趋势。
GRID_DEFAULT = {
    "missing": [0, 1, 2, 3, 4],
    "degrade": [256, 192, 128, 64, 32],
    "noise":   [0.0, 0.001, 0.005, 0.01, 0.02],
    "dist":    [50, 70, 90, 110],      # Fig.9  盲区离图中心的距离（px）
    "side":    [90, 110, 130, 150],    # Fig.10 盲区边长（px）
}

# 论文网格：只取论文图上真实存在的横坐标点，用于逐点叠图比对。
#   Fig.14 噪声强度 0.1% / 0.4% / 0.7% / 1.0%
#   Fig.15 下采样保留率 25% / 33% / 50% / 75%（256×ratio → 64/85/128/192）
#   Fig.16 缺失建筑 1–4 栋
# 每项都额外保留一个"干净参照"（0 / 256），但画图时不落到论文横轴上。
GRID_PAPER = {
    "missing": [0, 1, 2, 3, 4],
    "degrade": [256, 192, 128, 85, 64],
    "noise":   [0.0, 0.001, 0.004, 0.007, 0.010],
    "dist":    [50, 70, 90, 110],
    "side":    [90, 110, 130, 150],
}


def grid_for(mode, paper):
    return list((GRID_PAPER if paper else GRID_DEFAULT)[mode])


# ----------------------------------------------------------------------
# 模型装载 / 前向
# ----------------------------------------------------------------------
def load_model(tag, kind, device):
    ck = torch.load("runs/%s/best.pt" % tag, map_location="cpu",
                    weights_only=False)
    if kind == "mffgcn":
        m, _ = build_from_ckpt(ck, device)
        return m.to(device), kind
    m = build_baseline(kind, ae_skip=(ck.get("args") or {}).get("ae_skip",
                                                               "none"))
    m.load_state_dict(ck["model"])
    if kind == "radiounet":
        m.stage = 2          # 用第二阶段（带 hint）的输出，与论文设置一致
    return m.to(device), kind


@torch.no_grad()
def forward(model, kind, obs, env, mask):
    if kind == "mffgcn":
        _, y, _ = model(obs, env, mask)
        return y
    out = model(obs, env)
    return out[-1] if isinstance(out, (tuple, list)) else out


class Acc:
    """NMSE = Σse / Σtgt²，和 evaluate.py 完全一致。"""

    def __init__(self):
        self.se = 0.0
        self.sq = 0.0
        self.n = 0
        self.per = []

    def add(self, y, tgt):
        y = y.clamp(0, 1)
        e = float(((y - tgt) ** 2).sum())
        t = float((tgt ** 2).sum())
        self.se += e
        self.sq += t
        self.n += tgt.numel()
        self.per.append(e / max(t, 1e-12))

    def out(self):
        p = np.asarray(self.per) if self.per else np.zeros(1)
        return {
            "nmse": self.se / max(self.sq, 1e-12),
            "rmse": (self.se / max(self.n, 1)) ** 0.5,
            # 顺手把分母记下来：NMSE = RMSE²/E[t²]，没有 E[t²] 就无法和
            # 别人（或别的采样范围）的 NMSE 比大小，这是之前踩过的坑
            "et2": self.sq / max(self.n, 1),
            "median_nmse": float(np.median(p)),
            "p90_nmse": float(np.percentile(p, 90)),
            "n": len(self.per),
        }


# ----------------------------------------------------------------------
# 扰动算子
# ----------------------------------------------------------------------
def building_components(bld, rng, min_area=50):
    """把建筑物图拆成连通域，返回 (label 图, 按随机顺序排好的连通域 id 列表)。

    随机顺序只生成**一次**，各级别取前 n 个 —— 这样 1→4 栋是嵌套的
    （第 n 级 = 第 n-1 级再多抹一栋），曲线才单调，不会出现"抹 4 栋反而比 3 栋好"。
    旧脚本每个级别独立挑最大的，既非嵌套又抹掉 5% 的图。
    """
    b = (bld[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
    lab, num = ndimage.label(b)
    if num == 0:
        return lab, [], np.zeros(1)
    areas = ndimage.sum(b, lab, range(1, num + 1))
    cand = [i for i in range(num) if areas[i] >= min_area]
    rng.shuffle(cand)
    return lab, cand, areas


def erase_first(bld, lab, cand, areas, n):
    """抹掉候选序列里的前 n 个连通域，返回 (新图, 累计抹除面积占比)。"""
    if n <= 0 or not cand:
        return bld, 0.0
    out = (lab > 0).astype(np.float32)
    erased = 0.0
    for i in cand[:n]:
        out[lab == (i + 1)] = 0
        erased += float(areas[i])
    t = torch.from_numpy(out)[None, None]
    return t.to(bld.device), erased / out.size



def degrade(bld, res, binarize=True):
    """环境图降分辨率：res×res 平均池化 → 最近邻放大回 256 → 重新二值化。

    论文 Fig.15 的做法就是"下采样-上采样后按阈值转回二值掩码"，
    不做二值化会得到灰度边缘，和论文口径不一致（模型也会因此"看到"额外的软信息）。
    """
    if res >= IMG:
        return bld
    # 用 adaptive_avg_pool2d 而不是 avg_pool2d(kernel=IMG//res)：
    # 256//192 = 1，kernel=1 就是恒等映射，75% 那一档会变成"没退化"（曾踩过）。
    small = F.adaptive_avg_pool2d(bld, (res, res))
    up = F.interpolate(small, size=(IMG, IMG), mode="nearest")
    return (up > 0.5).float() if binarize else up


def impulse_noise(bld, p, rng):
    """环境图脉冲噪声：随机挑 p 比例的像素翻转（建筑↔空地）。"""
    if p <= 0:
        return bld
    x = bld.clone()
    n_pix = int(round(p * x.numel()))
    idx = torch.from_numpy(rng.choice(x.numel(), size=n_pix, replace=False))
    flat = x.view(-1)
    flat[idx] = 1.0 - flat[idx]
    return x


def blind_strip(frac, rng):
    """贴边矩形条带（Setup 2 的原有形状），True=可观测。"""
    allowed = np.ones((IMG, IMG), dtype=bool)
    side = int(rng.integers(4))
    if side in (0, 1):
        h = int(round(IMG * frac))
        if side == 0:
            allowed[:h, :] = False
        else:
            allowed[-h:, :] = False
    else:
        w = int(round(IMG * frac))
        if side == 2:
            allowed[:, :w] = False
        else:
            allowed[:, -w:] = False
    return allowed


def blind_square(side_px, dist_px):
    """正方形盲区：边长 side_px，中心离图中心 dist_px（沿对角方向移动）。

    论文 Fig.9/10 只说"受限区离中心的距离 / 受限区边长"，没写 Fig.9 用的边长。
    取 side=100、沿对角线走：dist 到 110 时偏移 78 px，方窗仍完整落在图内，
    不会因为被裁掉一角而让各档的盲区面积不一致（那样曲线会被面积变化污染）。
    """
    allowed = np.ones((IMG, IMG), dtype=bool)
    off = dist_px / np.sqrt(2.0)
    cy, cx = IMG / 2.0 + off, IMG / 2.0 + off
    y0 = int(round(cy - side_px / 2.0))
    x0 = int(round(cx - side_px / 2.0))
    y0 = max(0, min(IMG - side_px - 1, y0))
    x0 = max(0, min(IMG - side_px - 1, x0))
    allowed[y0:y0 + side_px, x0:x0 + side_px] = False
    return allowed


def _disc(shape, cy, cx, r):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def blind_circle(frac, rng):
    """贴边圆形盲区：圆心落在一条边上，面积凑到 frac（半个圆在画布外）。"""
    side = int(rng.integers(4))
    target = frac * IMG * IMG
    r = np.sqrt(target / np.pi)
    for _ in range(12):                     # 二分校正，保证画布内面积 = target
        if side == 0:
            m = _disc((IMG, IMG), 0, IMG // 2, r)
        elif side == 1:
            m = _disc((IMG, IMG), IMG - 1, IMG // 2, r)
        elif side == 2:
            m = _disc((IMG, IMG), IMG // 2, 0, r)
        else:
            m = _disc((IMG, IMG), IMG // 2, IMG - 1, r)
        cur = m.sum()
        if abs(cur - target) / target < 0.01:
            break
        r *= np.sqrt(target / max(cur, 1))
    return ~m


def _tri_mask(shape, verts):
    """点是否落在三角形内（重心坐标符号法），verts 为 3 个 (y, x)。"""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    pts = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float64)
    (y0, x0), (y1, x1), (y2, x2) = verts

    def side(ay, ax, by, bx):
        return (by - ay) * (pts[:, 1] - ax) - (bx - ax) * (pts[:, 0] - ay)

    d0 = side(y0, x0, y1, x1)
    d1 = side(y1, x1, y2, x2)
    d2 = side(y2, x2, y0, x0)
    neg = (d0 < 0) | (d1 < 0) | (d2 < 0)
    pos = (d0 > 0) | (d1 > 0) | (d2 > 0)
    return (~(neg & pos)).reshape(shape)


def blind_triangle(frac, rng):
    """贴边三角形盲区（论文表 VI 用的形状之一）。

    底边落在一条边上、顶点指向对边中点；面积 = 底×高/2，
    要让面积等于 frac 就得底边 = 2·frac·IMG。
    """
    side = int(rng.integers(4))
    target = frac * IMG * IMG
    base = 2.0 * target / IMG
    c = float(rng.uniform(0.25, 0.75)) * IMG      # 底边中点位置
    lo = max(0.0, min(IMG - 1.0, c - base / 2))
    hi = max(0.0, min(IMG - 1.0, c + base / 2))
    if side == 0:
        m = _tri_mask((IMG, IMG), [(0, lo), (0, hi), (IMG - 1, c)])
    elif side == 1:
        m = _tri_mask((IMG, IMG), [(IMG - 1, lo), (IMG - 1, hi), (0, c)])
    elif side == 2:
        m = _tri_mask((IMG, IMG), [(lo, 0), (hi, 0), (c, IMG - 1)])
    else:
        m = _tri_mask((IMG, IMG), [(lo, IMG - 1), (hi, IMG - 1), (c, 0)])
    return ~m


def blind_irregular(frac, rng):
    """不规则盲区：若干个半径不等的圆叠在一起，贴着同一条边，总面积凑到 frac。"""
    side = int(rng.integers(4))
    target = frac * IMG * IMG
    m = np.zeros((IMG, IMG), dtype=bool)
    n_blob = int(rng.integers(3, 6))
    for _ in range(n_blob):
        r = float(rng.uniform(0.18, 0.42)) * IMG
        t = float(rng.uniform(0.15, 0.85))
        if side == 0:
            cy, cx = 0, t * IMG
        elif side == 1:
            cy, cx = IMG - 1, t * IMG
        elif side == 2:
            cy, cx = t * IMG, 0
        else:
            cy, cx = t * IMG, IMG - 1
        m |= _disc((IMG, IMG), cy, cx, r)
    if m.sum() > target:                    # 超了就按比例缩半径重画一次
        scale = np.sqrt(target / m.sum())
        m[:] = False
        for _ in range(n_blob):
            r = float(rng.uniform(0.18, 0.42)) * IMG * scale
            t = float(rng.uniform(0.15, 0.85))
            if side == 0:
                cy, cx = 0, t * IMG
            elif side == 1:
                cy, cx = IMG - 1, t * IMG
            elif side == 2:
                cy, cx = t * IMG, 0
            else:
                cy, cx = t * IMG, IMG - 1
            m |= _disc((IMG, IMG), cy, cx, r)
    return ~m


def resample_obs(target, allowed, rng, n_min=10, n_max=300):
    """按 dataset 的规则重抽观测点：n_obs ~ U[n_min, n_max]，均匀落在 allowed 内。"""
    cand = np.flatnonzero(allowed.ravel())
    if cand.size == 0:
        return np.zeros((IMG, IMG), np.float32), np.zeros((IMG, IMG), np.float32)
    n_obs = int(rng.integers(n_min, n_max + 1))
    n_obs = min(n_obs, cand.size)
    mask = np.zeros(IMG * IMG, dtype=np.float32)
    mask[rng.choice(cand, size=n_obs, replace=False)] = 1.0
    mask = mask.reshape(IMG, IMG)
    return target * mask, mask


def sample_idx(ds, n, span=False):
    """取参与评估的图索引。

    span=False 是旧行为：直接取前 n 张。**这在 test 集上是有偏的** ——
    index.csv 按城市排序，前 200 张只覆盖约 2.5 个城市，这批图的目标
    能量 E[t^2]=0.0448，而全 8080 张是 0.0688。因为 NMSE = SE/Σt²，
    分母偏小就把所有 NMSE 系统性地抬高了 0.0688/0.0448 = 1.54 倍
    （开平方到 RMSE 是 1.24 倍）。论文的 E[t^2] 由 Fig.9/10 的
    NMSE 与 RMSE 互推是 0.0532，说明它用的是全测试集。

    span=True 时等间隔铺满整个测试集，n 不用加大就能拿到无偏的 E[t^2]。
    """
    if not span or n >= len(ds):
        return list(range(min(n, len(ds))))
    return [int(round(x)) for x in np.linspace(0, len(ds) - 1, n)]


# ----------------------------------------------------------------------
# 四个模式
# ----------------------------------------------------------------------
def run_missing(args, dev, models):
    ds = RadioMapSeer("../data/cache", "test", args.setup, seed=2024)
    levels = grid_for("missing", args.paper_grid)
    acc = {(m, l): Acc() for m, _, _ in MODELS for l in levels}
    erase = {l: [] for l in levels}

    idx = sample_idx(ds, args.n, args.span)
    for k, i in enumerate(idx):
        b = ds[i]
        obs = b["obs"][None].to(dev)
        env0 = b["env"][None].to(dev)
        mask = b["mask"][None].to(dev)
        tgt = b["target"][None].to(dev)
        # 连通域与抹除顺序每个样本只算一次，各级别共用（嵌套）
        rng0 = np.random.default_rng([2024 + i * 7919, 17])
        lab, cand, areas = building_components(env0[:, 0:1], rng0)
        for l in levels:
            env = env0.clone()
            if l:
                new_b, frac = erase_first(env0[:, 0:1], lab, cand, areas, l)
                env[:, 0:1] = new_b
                erase[l].append(frac)
            for name, m, kind in models:
                acc[(name, l)].add(forward(m, kind, obs, env, mask), tgt)
        if (k + 1) % 50 == 0:
            print("  missing %d/%d" % (k + 1, len(idx)), flush=True)

    out = {"mode": "missing", "n": args.n, "setup": args.setup, "span": args.span,
           "erase_frac": {str(l): float(np.mean(v)) if v else 0.0
                          for l, v in erase.items()},
           "models": {name: {str(l): acc[(name, l)].out() for l in levels}
                      for name, _, _ in MODELS}}
    return out


def run_degrade(args, dev, models):
    ds = RadioMapSeer("../data/cache", "test", args.setup, seed=2024)
    # 论文 Fig.15 说 75% 分辨率就足以让误差翻倍，所以 192 这一档一定要有
    levels = grid_for("degrade", args.paper_grid)
    acc = {(m, l): Acc() for m, _, _ in MODELS for l in levels}

    idx = sample_idx(ds, args.n, args.span)
    for k, i in enumerate(idx):
        b = ds[i]
        obs = b["obs"][None].to(dev)
        env0 = b["env"][None].to(dev)
        mask = b["mask"][None].to(dev)
        tgt = b["target"][None].to(dev)
        for l in levels:
            env = env0.clone()
            env[:, 0:1] = degrade(env0[:, 0:1], l)   # 只退化建筑物通道，发射机位置保持精确
            for name, m, kind in models:
                acc[(name, l)].add(forward(m, kind, obs, env, mask), tgt)
        if (k + 1) % 50 == 0:
            print("  degrade %d/%d" % (k + 1, len(idx)), flush=True)

    out = {"mode": "degrade", "n": args.n, "setup": args.setup, "span": args.span,
           "models": {name: {str(l): acc[(name, l)].out() for l in levels}
                      for name, _, _ in MODELS}}
    return out


def run_noise(args, dev, models):
    """脉冲噪声（论文 Fig.14）：在建筑二值图上随机翻转 0.1%–2% 的像素。"""
    ds = RadioMapSeer("../data/cache", "test", args.setup, seed=2024)
    levels = grid_for("noise", args.paper_grid)
    acc = {(m, l): Acc() for m, _, _ in MODELS for l in levels}

    idx = sample_idx(ds, args.n, args.span)
    for k, i in enumerate(idx):
        b = ds[i]
        obs = b["obs"][None].to(dev)
        env0 = b["env"][None].to(dev)
        mask = b["mask"][None].to(dev)
        tgt = b["target"][None].to(dev)
        for l in levels:
            env = env0.clone()
            if l:
                rng = np.random.default_rng([2024 + i * 7919, 31])
                env[:, 0:1] = impulse_noise(env0[:, 0:1], l, rng)
            for name, m, kind in models:
                acc[(name, l)].add(forward(m, kind, obs, env, mask), tgt)
        if (k + 1) % 50 == 0:
            print("  noise %d/%d" % (k + 1, len(idx)), flush=True)

    out = {"mode": "noise", "n": args.n, "setup": args.setup, "span": args.span,
           "models": {name: {str(l): acc[(name, l)].out() for l in levels}
                      for name, _, _ in MODELS}}
    return out


def run_geom(args, dev, models, mode):
    """Fig.9（盲区离中心的距离）/ Fig.10（盲区边长）。

    论文写的是"under Setup 3 深入分析障碍区的影响"，因此用 s3 权重，
    并把受限区换成可参数化的正方形（训练时见到的仍是 Setup 3 的贴边条带）。
    观测点在剩余区域里按 dataset 的同一规则重抽（n_obs ~ U[10,300]），
    所以不同档位之间唯一的变量就是几何本身。
    """
    ds = RadioMapSeer("../data/cache", "test", args.setup, seed=2024)
    levels = grid_for(mode, args.paper_grid)
    SIDE = 100 if mode == "dist" else None
    acc = {(m, l): Acc() for m, _, _ in MODELS for l in levels}

    idx = sample_idx(ds, args.n, args.span)
    for k, i in enumerate(idx):
        b = ds[i]
        tgt_np = b["target"][0].numpy()
        env = b["env"][None].to(dev)
        tgt = b["target"][None].to(dev)
        for l in levels:
            rng = np.random.default_rng([2024 + i * 7919, 41])
            allowed = blind_square(SIDE or l, l if mode == "dist" else 0)
            obs_np, mask_np = resample_obs(tgt_np, allowed, rng)
            obs = torch.from_numpy(obs_np)[None, None].to(dev)
            mask = torch.from_numpy(mask_np)[None, None].to(dev)
            for name, m, kind in models:
                acc[(name, l)].add(forward(m, kind, obs, env, mask), tgt)
        if (k + 1) % 40 == 0:
            print("  %s %d/%d" % (mode, k + 1, len(idx)), flush=True)

    return {"mode": mode, "n": args.n, "setup": args.setup, "span": args.span,
            "side_px": SIDE,
            "models": {name: {str(l): acc[(name, l)].out() for l in levels}
                       for name, _, _ in MODELS}}


def run_shape(args, dev, models):
    """盲区形状零样本泛化（表 VI）。模型用 s2 权重（训练时只见贴边矩形条带）。"""
    ds = RadioMapSeer("../data/cache", "test", args.setup, seed=2024)
    shapes = ["strip", "circle", "triangle", "irregular"]
    frac = 0.25
    acc = {(m, s): Acc() for m, _, _ in MODELS for s in shapes}

    idx = sample_idx(ds, args.n, args.span)
    for k, i in enumerate(idx):
        b = ds[i]
        tgt_np = b["target"][0].numpy()
        env = b["env"][None].to(dev)
        tgt = b["target"][None].to(dev)
        for s in shapes:
            rng = np.random.default_rng([2024 + i * 7919, 29])
            if s == "strip":
                allowed = blind_strip(frac, rng)
            elif s == "circle":
                allowed = blind_circle(frac, rng)
            elif s == "triangle":
                allowed = blind_triangle(frac, rng)
            else:
                allowed = blind_irregular(frac, rng)
            obs_np, mask_np = resample_obs(tgt_np, allowed, rng)
            obs = torch.from_numpy(obs_np)[None, None].to(dev)
            mask = torch.from_numpy(mask_np)[None, None].to(dev)
            for name, m, kind in models:
                acc[(name, s)].add(forward(m, kind, obs, env, mask), tgt)
        if (k + 1) % 40 == 0:
            print("  shape %d/%d" % (k + 1, len(idx)), flush=True)

    out = {"mode": "shape", "n": args.n, "setup": "s2", "span": args.span,
           "blind_frac": frac,
           "models": {name: {s: acc[(name, s)].out() for s in shapes}
                      for name, _, _ in MODELS}}
    return out


def run_latency(args, dev, models):
    """推理时延 / 参数量 / FLOPs。单卡独占时测才有意义，所以会打印当时占用率。"""

    def params(m):
        return sum(p.numel() for p in m.parameters())

    def flops(m, kind, bs):
        obs = torch.rand(bs, 1, IMG, IMG, device=dev)
        env = torch.rand(bs, 2, IMG, IMG, device=dev)
        mask = (torch.rand(bs, 1, IMG, IMG, device=dev) > 0.9).float()
        try:
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA],
                    with_flops=True) as prof:
                forward(m, kind, obs, env, mask)
            tot = sum(e.flops for e in prof.key_averages()
                      if getattr(e, "flops", 0))
            return tot / bs / 1e9
        except Exception as exc:                      # noqa: BLE001
            print("  [warn] flops 统计失败:", exc)
            return float("nan")

    def latency(m, kind, bs, iters=100, warmup=20):
        obs = torch.rand(bs, 1, IMG, IMG, device=dev)
        env = torch.rand(bs, 2, IMG, IMG, device=dev)
        mask = (torch.rand(bs, 1, IMG, IMG, device=dev) > 0.9).float()
        for _ in range(warmup):
            forward(m, kind, obs, env, mask)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            forward(m, kind, obs, env, mask)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters / bs      # 每样本 ms

    res = {"mode": "latency", "gpu_util": "", "models": {}}
    try:
        import subprocess
        res["gpu_util"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception:                                  # noqa: BLE001
        pass

    for name, m, kind in models:
        r = {"params_m": params(m) / 1e6,
             "flops_g": flops(m, kind, 1),
             "lat_ms_b1": latency(m, kind, 1),
             "lat_ms_b8_each": latency(m, kind, 8, iters=50)}
        res["models"][name] = r
        print("  %-10s params %.2fM  %.1f GFLOPs  %.2f ms/样本(b1)  %.2f ms/样本(b8)"
              % (name, r["params_m"], r["flops_g"], r["lat_ms_b1"], r["lat_ms_b8_each"]),
              flush=True)
    return res


# ----------------------------------------------------------------------
# 出图
# ----------------------------------------------------------------------
COLOR = {"MFFGCN": "#2E7D32", "RadioUNet": "#1f77b4", "RME-GAN": "#ff7f0e"}


def plot(res, outdir, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    mode = res["mode"]
    if mode == "latency":
        return None

    if mode == "missing":
        levels = sorted(int(k) for k in res["models"]["MFFGCN"])
        xlab = "Number of missing buildings"
        title = "Effect of missing buildings (test-time only, %s weights)" % res.get("setup", "")
    elif mode == "degrade":
        levels = sorted(int(k) for k in res["models"]["MFFGCN"])
        xlab = "Environment map resolution (px, upsampled back to 256)"
        title = "Effect of environment map resolution (%s weights)" % res.get("setup", "")
    elif mode == "noise":
        levels = sorted(float(k) for k in res["models"]["MFFGCN"])
        xlab = "Impulse noise ratio on building map"
        title = "Effect of environment map impulse noise (%s weights)" % res.get("setup", "")
    elif mode in ("dist", "side"):
        levels = sorted(int(k) for k in res["models"]["MFFGCN"])
        xlab = ("Distance from map centre (px)" if mode == "dist"
                else "Side length of restricted region (px)")
        title = ("Effect of restricted-region distance (%s weights, side=%s px)"
                 % (res.get("setup", ""), res.get("side_px") or "-"))
        if mode == "side":
            title = "Effect of restricted-region side length (%s weights)" % res.get("setup", "")
    else:
        levels = list(res["models"]["MFFGCN"].keys())
        xlab = "Blind-zone shape (25% area, s2 weights, zero-shot)"
        title = "Generalization to unseen blind-zone shapes (%s weights)" % res.get("setup", "")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), dpi=170)
    for ax, key, ylab in ((axes[0], "nmse", "NMSE"), (axes[1], "rmse", "RMSE")):
        for name in ("RadioUNet", "RME-GAN", "MFFGCN"):
            d = res["models"].get(name)
            if not d:
                continue
            vals = [d[str(l)][key] for l in levels]
            if mode in ("degrade", "noise", "dist", "side"):
                xs = levels
            else:
                xs = np.arange(len(levels))
            ax.plot(xs, vals, "o-", lw=1.8, ms=5, label=name,
                    color=COLOR.get(name, "#555"))
        if mode == "degrade":
            ax.set_xscale("log", base=2)
            ax.set_xticks(levels)
            ax.set_xticklabels([str(l) for l in levels])
        elif mode in ("dist", "side"):
            ax.set_xticks(levels)
            ax.set_xticklabels([str(l) for l in levels])
            ax.set_xlim(min(levels) - 8, max(levels) + 8)
        elif mode == "noise":
            ax.set_xticks(levels)
            ax.set_xticklabels(["%.1f%%" % (l * 100) for l in levels])
        else:
            ax.set_xticks(np.arange(len(levels)))
            ax.set_xticklabels([str(l) for l in levels])
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.25, lw=0.7)
        ax.legend(fontsize=8.5, frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join("../figs", "supp_%s%s.png" % (mode, suffix))
    fig.savefig(out, bbox_inches="tight")
    print("saved:", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["missing", "degrade", "noise", "shape",
                             "dist", "side", "latency"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--setup", default=None,
                    help="不填则按论文口径自动选：missing/degrade/noise→s3，shape→s2")
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--replot", action="store_true",
                    help="只按已有 results/supp_<mode>.json 重画，不跑推理")
    ap.add_argument("--paper-grid", action="store_true",
                    help="横坐标只取论文图上真实存在的点（Fig.14/15/16 逐点比对用）")
    ap.add_argument("--span", action="store_true",
                    help="等间隔铺满整个测试集取图（消除前 n 张的城市偏差，"
                         "让 E[t^2] 与全集一致）。不加则沿用旧的前 n 张行为。")
    a = ap.parse_args()

    suffix = "_paper" if a.paper_grid else ""

    if a.replot:
        path = "../results/supp_%s%s.json" % (a.mode, suffix)
        with open(path, encoding="utf-8") as fh:
            res = json.load(fh)
        plot(res, "../figs", suffix)
        print("replot:", path)
        return

    if a.setup is None:
        a.setup = "s2" if a.mode == "shape" else ("s1" if a.mode == "latency" else "s3")

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    dev = torch.device("cuda")

    models = []
    for name, tpl, kind in MODELS:
        tag = tpl.format(s=a.setup) + a.tag_suffix
        m, k = load_model(tag, kind, dev)
        m.eval()
        models.append((name, m, k))
        print("[load] %-10s %s" % (name, tag), flush=True)

    if a.mode in ("dist", "side"):
        res = run_geom(a, dev, models, a.mode)
    else:
        fn = {"missing": run_missing, "degrade": run_degrade, "noise": run_noise,
              "shape": run_shape, "latency": run_latency}[a.mode]
        res = fn(a, dev, models)

    os.makedirs("../results", exist_ok=True)
    path = "../results/supp_%s%s.json" % (a.mode, suffix)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print("saved:", path)
    plot(res, "../figs")

    for name, _, _ in MODELS:
        print("\n--", name)
        for l, v in res["models"][name].items():
            if isinstance(v, dict) and "nmse" in v:
                print("   %-10s NMSE %.5f  RMSE %.5f" % (l, v["nmse"], v["rmse"]))


if __name__ == "__main__":
    main()
