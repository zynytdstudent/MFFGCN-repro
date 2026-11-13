# -*- coding: utf-8 -*-
"""诊断：NMSE 的分母 E[t^2] 到底是多大，由什么决定（2026-11-13）。

动机
----
论文 Table III 里每个模型都给了 NMSE 和 RMSE。两者满足
        NMSE = SE / SQ,  RMSE = sqrt(SE / N)
对同一批图，可以反解出信号能量
        E[t^2] = SQ / N = RMSE^2 / NMSE
论文 9 个模型 x 3 个 setup 共 27 个格子，反解出来几乎全是 0.052~0.055，
说明论文的 NMSE 是"全数据集累加"（micro）口径，且分母是个稳定常数。

我们所有实验反解出来是 0.0685，比论文高 30%。NMSE 因此被系统性压小 23%。
本脚本要回答的是：这 30% 是哪一步造成的 —— 测试集本身，还是 preprocess_gain
里的"截断 0.2 + 拉伸回 [0,1]"。

用法（项目根目录，CPU 即可）
    python diag_et2.py
    python diag_et2.py --cache ../data/cache
"""
import argparse
import csv
import os

import numpy as np

IMG = 256


def split_rows(cache_dir, split="test"):
    """按城市划分，与 dataset.py 的 split_mode='city' 保持完全一致。"""
    rows = []
    with open(os.path.join(cache_dir, "index.csv")) as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["idx"]), int(r["city_id"]), int(r["tx_id"])))
    cities = sorted({r[1] for r in rows})
    keep = {"train": set(cities[:500]),
            "val": set(cities[500:600]),
            "test": set(cities[600:])}[split]
    return [r for r in rows if r[1] in keep]


def et2(x):
    return float(np.mean(x.astype(np.float64) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--limit", type=int, default=0, help="只读前 N 张（0=全部）")
    a = ap.parse_args()

    rows = split_rows(a.cache, "test")
    if a.limit:
        rows = rows[:a.limit]
    gain = np.load(os.path.join(a.cache, "gain_dpm.npy"), mmap_mode="r")
    print("[info] test samples =", len(rows))

    # 各口径的累加器：sum(x^2) 与 sum(x)
    acc = {k: 0.0 for k in
           ("raw255", "cut20_stretch", "cut20_only", "cut10_only",
            "cut30_only", "cut15_only")}
    n_pix = 0
    per_img_q = []          # 每张图 raw/255 的 90 分位，用来看信号强弱分布

    for i, (idx, city, tx) in enumerate(rows):
        raw = np.asarray(gain[idx], dtype=np.float32)
        x0 = raw / 255.0

        acc["raw255"] += float(np.sum(x0.astype(np.float64) ** 2))

        for thr, key in ((0.2, "cut20_only"), (0.1, "cut10_only"),
                         (0.3, "cut30_only"), (0.15, "cut15_only")):
            y = x0.copy()
            y[y < thr] = 0.0
            acc[key] += float(np.sum(y.astype(np.float64) ** 2))

        # 我们线上实际用的口径
        y = x0.copy()
        y[y < 0.2] = 0.0
        y = np.clip((y - 0.2) / 0.8, 0.0, 1.0)
        acc["cut20_stretch"] += float(np.sum(y.astype(np.float64) ** 2))

        per_img_q.append(float(np.quantile(x0, 0.9)))
        n_pix += x0.size

        if (i + 1) % 2000 == 0:
            print("  %d/%d" % (i + 1, len(rows)), flush=True)

    print("\n=== E[t^2] by preprocessing (test split, %d imgs) ===" % len(rows))
    print("论文 Table III 反解值 = 0.0526  (0.052~0.055)")
    for k, v in acc.items():
        e = v / n_pix
        print("  %-15s %.4f   (vs paper %+.1f%%)" % (k, e, 100 * (e / 0.0526 - 1)))

    q = np.array(per_img_q)
    print("\n=== raw/255 每张图 90 分位（信号强弱）===")
    for p in (5, 25, 50, 75, 95):
        print("  p%-3d %.3f" % (p, np.percentile(q, p)))


if __name__ == "__main__":
    main()
