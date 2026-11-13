"""
数据预处理：把 RadioMapSeer 的原始 PNG 打包成 .npy 缓存
================================================================
只跑一次，之后训练/评估全部读缓存，速度提升很多。

--------------------------------------------------------------------------------
原始数据的目录结构
--------------------------------------------------------------------------------
    RadioMapSeer/
        gain/gain/carsDPM/<city>_<tx>.png     56080 张增益图（city 0..700, tx 0..79）
        png/buildings_complete/<city>.png       701 张建筑物图
        png/antennas/<city>_<tx>.png          每样本一张单发射机指示图
        gain/antenna/<city>.json              每个城市的 80 个发射机坐标

--------------------------------------------------------------------------------
输出（5 个文件，共约 3.5 GB）
--------------------------------------------------------------------------------
    gain_dpm.npy        (N,256,256) uint8    N = 56080，增益图，约 3.5 GB
    buildings.npy       (C,256,256) uint8    C = 701，建筑物图，约 46 MB
    buildings_city.npy  (C,)         int16   每行对应的 city id
    tx_coords.npy       (N,2)        int16   (行, 列)，发射机位置
    index.csv           N 行                 idx, city_id, tx_id, gain_file

--------------------------------------------------------------------------------
两个关键设计
--------------------------------------------------------------------------------
1) 为什么把 56080 张 PNG 压成一个大 npy？
   训练时要随机访问任意样本，PNG 每张都要解码（约 2ms），一个 epoch 40000 样本
   就是 80 秒纯解码开销，而且 DataLoader 的多个 worker 会同时抢磁盘 IO。
   压成 npy 后可以用 mmap 按需分页，多个 worker 共享同一份物理内存，
   操作系统自己管缓存，实测数据加载几乎不再是瓶颈。

2) 发射机坐标为什么不读 antennas/*.png 而是从增益图 argmax 求？
   已验证：argmax(gain 图) 恰好等于 antennas 图里唯一的亮点位置。
   这样省掉 56080 次额外的 PNG 打开，预处理时间减少一半以上。
   ⚠ 代价是：如果某张增益图的最大值不在发射机位置（理论上可能发生，
     比如极远的反射路径恰好更强），坐标就会错。当前的 56080 张全部验证通过。
"""

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
from PIL import Image


def find_dir(root, *cands):
    """在 root 下按候选路径找目录，找不到就整个树遍历一遍按名字匹配。

    这么写是因为不同版本的 RadioMapSeer 压缩包解压后层级不一样
    （有时多套一层 gain/gain/，有时少一层），硬编码路径会直接崩。
    """
    for c in cands:
        p = os.path.join(root, c)
        if os.path.isdir(p):
            return p
    target = cands[0].split("/")[-1]
    for dirpath, _, _ in os.walk(root):
        if os.path.basename(dirpath) == target:
            return dirpath
    return None


def load_gray(path):
    """读成单通道灰度 uint8。

    必须显式 .convert("L")：原图可能是 RGB 或带 alpha，直接 np.array 会得到
    (H,W,3)，后面所有假设 (H,W) 的代码都会错位。
    """
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="RadioMapSeer 解压后的根目录")
    ap.add_argument("--out", required=True, help="缓存输出目录，如 ../data/cache")
    ap.add_argument("--sim", default="carsDPM",
                    help="仿真器：carsDPM（默认，主流做法）| carsIRT4")
    args = ap.parse_args()

    root = args.root
    gain_dir = find_dir(root, f"gain/gain/{args.sim}", f"gain/{args.sim}", args.sim)
    bld_dir = find_dir(root, "png/buildings_complete")
    for name, d in [("gain", gain_dir), ("buildings", bld_dir)]:
        if d is None:
            sys.exit(f"[ERROR] cannot locate {name} under {root}")
        print(f"[info] {name:10s} -> {d}")

    os.makedirs(args.out, exist_ok=True)

    # ==================================================================
    # 第一部分：建筑物图（701 张，很小，直接全读进内存）
    # ==================================================================
    def numkey(p):
        """从文件名里抽数字，作为排序键。

        必须按数字排而不是字符串排：字符串排序会把 "10.png" 排在 "2.png" 前面，
        导致 buildings.npy 的行序和 city id 对不上，
        进而让 dataset.py 里的 city_to_bidx 映射整体错位。
        """
        return int(re.findall(r"(\d+)", os.path.basename(p))[0])

    bld_files = sorted(glob.glob(os.path.join(bld_dir, "*.png")), key=numkey)
    bld_city = np.array([numkey(f) for f in bld_files], dtype=np.int16)
    print(f"[info] {len(bld_files)} city maps, id {bld_city.min()}..{bld_city.max()}")

    buildings = np.zeros((len(bld_files), 256, 256), dtype=np.uint8)
    for i, f in enumerate(bld_files):
        g = load_gray(f)
        if g.shape != (256, 256):
            # 少数图不是 256x256，双线性缩放（不是最近邻，避免边缘锯齿）
            g = np.array(Image.fromarray(g).resize((256, 256), Image.BILINEAR))
        buildings[i] = g
    np.save(os.path.join(args.out, "buildings.npy"), buildings)
    np.save(os.path.join(args.out, "buildings_city.npy"), bld_city)
    print(f"[ok] buildings.npy {buildings.shape}")

    # ==================================================================
    # 第二部分：增益图（56080 张，3.5 GB，用 memmap 边读边写）
    # ==================================================================
    # 文件名形如 "123_45.png"：city=123, tx=45
    pat = re.compile(r"^(\d+)_(\d+)\.png$")
    records = []
    for f in glob.glob(os.path.join(gain_dir, "*.png")):
        m = pat.match(os.path.basename(f))
        if m:
            records.append((int(m.group(1)), int(m.group(2)), f))
    # ⚠ 排序方式决定了全局 idx 的语义，必须与 dataset.py / tx_coords.npy 一致。
    # 按 (city, tx) 字典序升序，这样：
    #   city 0 占 idx 0..79，city 1 占 80..159，…
    # 于是"按城市切分"就等价于"按 idx 区间切分"，训练集永远从 idx=0 开始
    # —— 这也是当年 tx 坐标 bug 在训练集上"看起来正常"的原因。
    records.sort(key=lambda r: (r[0], r[1]))
    N = len(records)
    print(f"[info] {N} gain maps")

    # open_memmap 不占内存，直接往磁盘写，3.5 GB 也不会 OOM
    gain = np.lib.format.open_memmap(
        os.path.join(args.out, "gain_dpm.npy"), mode="w+",
        dtype=np.uint8, shape=(N, 256, 256))
    tx = np.zeros((N, 2), dtype=np.int16)

    for i, (cid, tid, f) in enumerate(records):
        g = load_gray(f)
        if g.shape != (256, 256):
            g = np.array(Image.fromarray(g).resize((256, 256), Image.BILINEAR))
        gain[i] = g
        # 从增益图里反推发射机位置：取最大值的坐标（见模块文档的说明）
        # unravel_index 把一维 argmax 转成 (行, 列)
        r, c = np.unravel_index(int(g.argmax()), g.shape)
        tx[i] = (r, c)
        if (i + 1) % 5000 == 0:
            print(f"  ... {i+1}/{N}  last={os.path.basename(f)} "
                  f"tx=({r},{c})", flush=True)

    gain.flush()                 # memmap 必须 flush 才真正落盘
    del gain                     # 释放文件句柄
    np.save(os.path.join(args.out, "tx_coords.npy"), tx)

    # ==================================================================
    # 第三部分：索引表
    # ==================================================================
    with open(os.path.join(args.out, "index.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "city_id", "tx_id", "gain_file"])
        for i, (cid, tid, f) in enumerate(records):
            w.writerow([i, cid, tid, os.path.basename(f)])

    cities = sorted({r[0] for r in records})
    print(f"[ok] gain_dpm.npy {(N,256,256)} = {N*65536/1e9:.2f} GB")
    print(f"[ok] {len(cities)} cities ({cities[0]}..{cities[-1]}), "
          f"{N/len(cities):.0f} tx per city")
    print(f"[ok] saved to {args.out}")


if __name__ == "__main__":
    main()
