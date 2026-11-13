"""查 paper_fig_16.py 的"抹掉 N 栋建筑"到底抹掉了多少面积。

怀疑点：drop_buildings 用 ndimage.label 的默认 4-连通，会把相互贴着的建筑并成一个大团；
一旦 200–6000 px 的"正常建筑"不够 n 个，就会退化成"直接删最大的团"，
而 256x256 图上最大的连通域可能占全图 1/3。那样 "1 栋缺失" 实际抹掉半张图，
指标会崩到 0.08 量级（fig16.json 就是这批数）。

用法：python probe_fig16_drop.py --n 1 2 3 4
"""
import argparse
import os

import numpy as np
from scipy import ndimage

from dataset import RadioMapSeer

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def analyse(bld, n):
    b = (bld > 0.5).astype(np.uint8)
    lab, num = ndimage.label(b)
    if num == 0:
        return num, 0, 0.0, 0.0
    areas = np.asarray(ndimage.sum(b, lab, range(1, num + 1)))
    pool = [i for i in range(num) if 200 <= areas[i] <= 6000]
    fallback = False
    if len(pool) < n:
        pool = [i for i in range(num) if areas[i] >= 100]
        fallback = True
    pool.sort(key=lambda i: -areas[i])
    erased = float(areas[pool[:n]].sum()) if pool else 0.0
    return num, len(pool), erased / b.size, fallback


ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="../data/cache")
ap.add_argument("--n", nargs="+", type=int, default=[1, 2, 3, 4])
ap.add_argument("--samples", type=int, default=5)
a = ap.parse_args()

ds = RadioMapSeer(a.cache, "test", "s1", seed=2024)
print("样本  连通域数  候选中  抹掉面积占比  是否退化为删最大团")
worst = {}
for i in range(a.samples):
    bld = ds[i]["env"][0].numpy()
    for n in a.n:
        num, cand, frac, fb = analyse(bld, n)
        print("  #%d   n=%d  %5d  %4d  %7.2f%%  %s"
              % (i, n, num, cand, frac * 100, "是" if fb else "否"))
        worst[n] = max(worst.get(n, 0.0), frac)
print("\n各 n 下最大抹除占比：")
for n in a.n:
    print("  n=%d -> %.1f%%" % (n, worst[n] * 100))
