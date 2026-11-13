"""
RadioMapSeer -> compact .npy cache.

This release of the dataset is organised as:

    RadioMapSeer/
        gain/gain/carsDPM/<city>_<tx>.png     56080 DPM gain maps  (city 0..700, tx 0..79)
        png/buildings_complete/<city>.png     701 building maps
        png/antennas/<city>_<tx>.png          one single-Tx map per sample
        gain/antenna/<city>.json              80 Tx coordinates (order does NOT follow tx id)

Verified: argmax(gain map) == the single lit pixel of png/antennas/<city>_<tx>.png,
so the transmitter position is read straight off the gain map and the antenna
PNGs never have to be opened.

Outputs:
    gain_dpm.npy        (N,256,256) uint8
    buildings.npy       (C,256,256) uint8
    buildings_city.npy  (C,)  int16   city id for every row of buildings.npy
    tx_coords.npy       (N,2) int16   (row, col)
    index.csv           idx, city_id, tx_id, gain_file
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
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim", default="carsDPM",
                    help="carsDPM (default) | carsIRT4")
    args = ap.parse_args()

    root = args.root
    gain_dir = find_dir(root, f"gain/gain/{args.sim}", f"gain/{args.sim}", args.sim)
    bld_dir = find_dir(root, "png/buildings_complete")
    for name, d in [("gain", gain_dir), ("buildings", bld_dir)]:
        if d is None:
            sys.exit(f"[ERROR] cannot locate {name} under {root}")
        print(f"[info] {name:10s} -> {d}")

    os.makedirs(args.out, exist_ok=True)

    # ---- buildings ------------------------------------------------------
    def numkey(p):
        return int(re.findall(r"(\d+)", os.path.basename(p))[0])

    bld_files = sorted(glob.glob(os.path.join(bld_dir, "*.png")), key=numkey)
    bld_city = np.array([numkey(f) for f in bld_files], dtype=np.int16)
    print(f"[info] {len(bld_files)} city maps, id {bld_city.min()}..{bld_city.max()}")

    buildings = np.zeros((len(bld_files), 256, 256), dtype=np.uint8)
    for i, f in enumerate(bld_files):
        g = load_gray(f)
        if g.shape != (256, 256):
            g = np.array(Image.fromarray(g).resize((256, 256), Image.BILINEAR))
        buildings[i] = g
    np.save(os.path.join(args.out, "buildings.npy"), buildings)
    np.save(os.path.join(args.out, "buildings_city.npy"), bld_city)
    print(f"[ok] buildings.npy {buildings.shape}")

    # ---- gain maps ------------------------------------------------------
    pat = re.compile(r"^(\d+)_(\d+)\.png$")
    records = []
    for f in glob.glob(os.path.join(gain_dir, "*.png")):
        m = pat.match(os.path.basename(f))
        if m:
            records.append((int(m.group(1)), int(m.group(2)), f))
    records.sort(key=lambda r: (r[0], r[1]))
    N = len(records)
    print(f"[info] {N} gain maps")

    gain = np.lib.format.open_memmap(
        os.path.join(args.out, "gain_dpm.npy"), mode="w+",
        dtype=np.uint8, shape=(N, 256, 256))
    tx = np.zeros((N, 2), dtype=np.int16)

    for i, (cid, tid, f) in enumerate(records):
        g = load_gray(f)
        if g.shape != (256, 256):
            g = np.array(Image.fromarray(g).resize((256, 256), Image.BILINEAR))
        gain[i] = g
        r, c = np.unravel_index(int(g.argmax()), g.shape)
        tx[i] = (r, c)
        if (i + 1) % 5000 == 0:
            print(f"  ... {i+1}/{N}  last={os.path.basename(f)} "
                  f"tx=({r},{c})", flush=True)

    gain.flush()
    del gain
    np.save(os.path.join(args.out, "tx_coords.npy"), tx)

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
