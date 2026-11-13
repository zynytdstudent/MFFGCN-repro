"""传统插值基线：Kriging 与 KNN（不需要训练）。

论文把这两个作为传统无线电地图估计方法的代表：它们不学习，只用测试样本
自身的稀疏观测点做空间插值，因此天然无法利用建筑环境信息。

实现
----
KNN   反距离加权（IDW），取 k 个最近观测点，权重 1/d^2。论文原文说 KNN
      "uses all data from the testing phase to predict new samples"，即对每个
      待估像素用邻近观测点回归。
Kriging  局部普通克里金。变差函数用指数模型
      gamma(h) = c0 + c1 * (1 - exp(-h / a))
      其中 sill 由该样本观测值的方差自适应给出，range 固定 25 像素（256x256
      地图上约 25 m 的空间相关长度）。每个待估像素解一个 k 邻域的普通克里金
      方程组（带拉格朗日乘子保证无偏）。

评估口径与深度模型一致：全局 / 盲区 / 观测点三个区域分别统计 NMSE 与 RMSE，
观测点采样用与训练完全相同的种子（测试集 = 训练 seed + 2000）。

用法：
    python kriging_knn.py --method knn --setup s2 --workers 24
    python kriging_knn.py --method kriging --setup s1 --workers 24
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

H = W = 256
_GRID = None


def grid():
    global _GRID
    if _GRID is None:
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        _GRID = np.stack([yy, xx], -1).reshape(-1, 2).astype(np.float64)
    return _GRID


def idw(obs, mask, k=8, power=2.0):
    """反距离加权插值。obs/mask: (H, W)。"""
    pts = np.argwhere(mask > 0).astype(np.float64)
    vals = obs[mask > 0].astype(np.float64)
    if len(vals) == 0:
        return np.zeros((H, W), np.float32)
    if len(vals) == 1:
        return np.full((H, W), vals[0], np.float32)
    kk = int(min(k, len(vals)))
    tree = cKDTree(pts)
    d, idx = tree.query(grid(), k=kk)
    d = np.maximum(np.asarray(d, dtype=np.float64), 1e-6)
    idx = np.asarray(idx)
    w = 1.0 / d ** power
    pred = (w * vals[idx]).sum(1) / w.sum(1)
    return pred.reshape(H, W).astype(np.float32)


def kriging(obs, mask, k=16, range_px=25.0, chunk=8192):
    """局部普通克里金。"""
    pts = np.argwhere(mask > 0).astype(np.float64)
    vals = obs[mask > 0].astype(np.float64)
    n = len(vals)
    if n < 8:
        return idw(obs, mask, k=8)
    kk = int(min(k, n))
    # 变差函数参数：sill 由本样本观测值方差自适应，nugget 取其 5%
    sill = float(max(np.var(vals) * 0.8, 1e-4))
    nugget = 0.05 * sill

    def gam(h):
        return nugget + sill * (1.0 - np.exp(-h / range_px))

    tree = cKDTree(pts)
    g = grid()
    out = np.empty(len(g), np.float64)
    eye = np.eye(kk)
    for s in range(0, len(g), chunk):
        gs = g[s:s + chunk]
        m = len(gs)
        d, idx = tree.query(gs, k=kk)
        if kk == 1:
            out[s:s + m] = vals[idx]
            continue
        nb = pts[idx]                                     # (m,k,2)
        dd = np.linalg.norm(nb[:, :, None, :] - nb[:, None, :, :], axis=-1)
        A = gam(dd) + nugget * eye[None]                  # (m,k,k)
        b = gam(np.asarray(d, dtype=np.float64))          # (m,k)
        # 普通克里金方程组（含拉格朗日乘子）
        M = np.zeros((m, kk + 1, kk + 1), np.float64)
        M[:, :kk, :kk] = A
        M[:, :kk, kk] = 1.0
        M[:, kk, :kk] = 1.0
        rhs = np.zeros((m, kk + 1), np.float64)
        rhs[:, :kk] = b
        rhs[:, kk] = 1.0
        try:
            w = np.linalg.solve(M, rhs[..., None])[..., 0][:, :kk]
        except np.linalg.LinAlgError:
            # 退化（重复观测点）：加对角抖动后逐块求解
            M2 = M + np.eye(kk + 1)[None] * 1e-8
            w = np.linalg.solve(M2, rhs[..., None])[..., 0][:, :kk]
        out[s:s + m] = (w * vals[idx]).sum(1)
    return out.reshape(H, W).astype(np.float32)


def _init(cache, split, setup, seed, obs_on_signal):
    global _DS
    from dataset import RadioMapSeer
    _DS = RadioMapSeer(cache, split, setup, seed=seed, obs_on_signal=obs_on_signal)


def _work(task):
    i, method = task
    b = _DS[i]
    obs = b["obs"][0].numpy()
    mask = b["mask"][0].numpy()
    tgt = b["target"][0].numpy()
    allowed = b["allowed"][0].numpy()

    pred = kriging(obs, mask) if method == "kriging" else idw(obs, mask)
    pred = np.clip(pred, 0.0, 1.0)

    regs = {"all": np.ones_like(mask), "constr": 1.0 - allowed, "obs": mask}
    res = {}
    for k, reg in regs.items():
        npx = float(reg.sum())
        if npx == 0:
            continue
        e = (pred - tgt) * reg
        t = tgt * reg
        res[k] = (float((e ** 2).sum()), float((t ** 2).sum()), int(npx))

    bands = {}
    if regs["constr"].sum() > 0:
        # 盲区内部每个像素到最近"可见"像素的距离：edt 的 True 区域取盲区，
        # 返回的就是到最近 False（可见区）的欧氏距离
        sel = allowed < 0.5
        dist = distance_transform_edt(sel)
        dd = dist[sel]
        ee = ((pred - tgt) ** 2)[sel]
        gg = (tgt ** 2)[sel]
        edges = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 1e9]
        lab = ["0-1", "1-2", "2-4", "4-8", "8-16", "16-32", "32+"]
        idx = np.digitize(dd, edges[1:-1])
        for j, name in enumerate(lab):
            m = idx == j
            if m.any():
                bands[name] = (float(ee[m].sum()), float(gg[m].sum()), int(m.sum()))
    return res, bands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["knn", "kriging"], required=True)
    ap.add_argument("--setup", default="s1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--seed", type=int, default=0,
                    help="训练 seed，测试集实际用 seed+2000")
    ap.add_argument("--obs-on-signal", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from dataset import RadioMapSeer
    ds = RadioMapSeer(a.cache, a.split, a.setup, seed=a.seed + 2000,
                      obs_on_signal=a.obs_on_signal)
    n = len(ds) if not a.max_samples else min(a.max_samples, len(ds))
    print(f"[info] {a.method} setup={a.setup} n={n}", flush=True)

    tasks = [(i, a.method) for i in range(n)]
    t0 = time.time()

    import multiprocessing as mp
    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
    acc = {"all": [0.0, 0.0, 0], "constr": [0.0, 0.0, 0], "obs": [0.0, 0.0, 0]}
    band = {}
    with ctx.Pool(a.workers, initializer=_init,
                  initargs=(a.cache, a.split, a.setup, a.seed + 2000,
                            a.obs_on_signal)) as pool:
        for j, (res, bands) in enumerate(pool.imap_unordered(_work, tasks, 64)):
            for k, (se, sq, npx) in res.items():
                acc[k][0] += se
                acc[k][1] += sq
                acc[k][2] += npx
            for k, (se, sq, npx) in bands.items():
                if k not in band:
                    band[k] = [0.0, 0.0, 0]
                band[k][0] += se
                band[k][1] += sq
                band[k][2] += npx
            if (j + 1) % 400 == 0:
                el = time.time() - t0
                print(f"  ... {j+1}/{n}  {el:.0f}s  eta {el/(j+1)*(n-j-1):.0f}s",
                      flush=True)

    out = {"method": a.method, "setup": a.setup, "n_samples": n, "fine": {},
           "bands": {}}
    for k, (se, sq, npx) in acc.items():
        if npx == 0:
            continue
        out["fine"][f"{k}_nmse"] = se / max(sq, 1e-12)
        out["fine"][f"{k}_rmse"] = (se / npx) ** 0.5
    for k, (se, sq, npx) in band.items():
        out["bands"][k] = {"rmse": (se / npx) ** 0.5,
                           "nmse": se / max(sq, 1e-12), "px": npx}

    print(json.dumps(out["fine"], indent=2))
    name = a.out or f"interp_{a.method}_{a.setup}.json"
    with open(name, "w") as fh:
        json.dump(out, fh, indent=1)
    print("[ok] wrote", name, f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
