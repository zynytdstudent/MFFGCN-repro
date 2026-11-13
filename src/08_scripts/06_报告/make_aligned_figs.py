# -*- coding: utf-8 -*-
"""在服务器上生成「坐标轴与论文一致」的两张复刻图。

Fig.6  训练 loss：xlim 0-100、ylim 0-0.010，与论文完全同尺度，逐 setup 单独出图
Fig.8  受限区域：7 方法 × Setup2/3 双柱，y 轴与论文同尺度（NMSE 0-0.14 / RMSE 0-0.095）
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/public2/czz/ZhangYu/project/mffgcn"
SRC = BASE + "/src/runs"
OUT = "/tmp/aligned"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.bbox": "tight"})


def load_loss(path, col="train_loss"):
    if not os.path.exists(path):
        return []
    pts = []
    for r in csv.DictReader(open(path)):
        try:
            pts.append((float(r["epoch"]), float(r[col])))
        except (KeyError, ValueError):
            pass
    return pts


# ---------------- Fig.6：严格对齐论文坐标 ----------------
for s, lab in (("s1", "(a)"), ("s2", "(b)"), ("s3", "(c)")):
    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    m = load_loss(f"{SRC}/txf_mffgcn_{s}/train_log.csv")
    r = load_loss(f"{SRC}/bl_radiounet_{s}/train_log_stage2.csv")
    if m:
        ax.plot(*zip(*m), color="#d62728", lw=1.6, label="MFFGCN train loss")
    if r:
        ax.plot(*zip(*r), color="#1f77b4", lw=1.6, label="RadioUNet train loss")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.010)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 0.002, 0.004, 0.006, 0.008, 0.010])
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("L2 Loss", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right", frameon=True)
    ax.set_title(lab, y=-0.30, fontsize=13)
    fig.savefig(f"{OUT}/loss_{s}.png", dpi=180)
    plt.close(fig)
    print("loss", s, "MFFGCN", round(m[-1][1], 6) if m else None,
          "RadioUNet", round(r[-1][1], 6) if r else None)


# ---------------- Fig.8：7 方法 × Setup2/3 ----------------
RUNS = {
    "AE": "bl_ae", "DeepAE": "bl_deepae", "UNet": "txf_single",
    "RadioUNet": "bl_radiounet", "Pix2pix+": "bl_pix2pix",
    "RME-GAN": "bl_rmegan", "MFFGCN": "txf_mffgcn",
}


def metric(tag, key):
    p = f"{SRC}/{tag}/test_metrics.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8")).get(key)


PANELS = [
    ("nmse", "NMSE", 0.14, [0, .02, .04, .06, .08, .10, .12], "(a)"),
    ("rmse", "RMSE", 0.095, [0, .02, .04, .06, .08], "(b)"),
]

fig, axes = plt.subplots(2, 1, figsize=(5.3, 7.4))
for ax, (key, ylab, ymax, yticks, lab) in zip(axes, PANELS):
    xs = range(len(RUNS))
    w = 0.38
    v2 = [metric(f"{t}_s2", key) for t in RUNS.values()]
    v3 = [metric(f"{t}_s3", key) for t in RUNS.values()]
    b2 = ax.bar([x - w / 2 for x in xs], [v or 0 for v in v2], w,
                label="Setup 2", color="#1f77b4")
    b3 = ax.bar([x + w / 2 for x in xs], [v or 0 for v in v3], w,
                label="Setup 3", color="#ff7f0e")
    # 缺数据的格子画个空心叉，避免被误读成 0
    for bars, vals in ((b2, v2), (b3, v3)):
        for bb, v in zip(bars, vals):
            if v is None:
                bb.set_visible(False)
                ax.text(bb.get_x() + bb.get_width() / 2, ymax * 0.03, "n/a",
                        ha="center", va="bottom", fontsize=6.5, color="#999")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(list(RUNS), fontsize=8.5)
    ax.set_ylim(0, ymax)
    ax.set_yticks(yticks)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Model", fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    ax.legend(fontsize=9, loc="upper right", frameon=True)
    ax.title.set_text("")
    ax.text(0.5, -0.20, lab, transform=ax.transAxes, ha="center", fontsize=13)

fig.tight_layout(h_pad=3.0)
fig.savefig(f"{OUT}/fig8.png", dpi=180)
plt.close(fig)
print("fig8 done")

# 顺便把数值打印出来，便于核对
for name, tag in RUNS.items():
    print(f"{name:10s} s2 nmse={metric(tag+'_s2','nmse')} rmse={metric(tag+'_s2','rmse')} | "
          f"s3 nmse={metric(tag+'_s3','nmse')} rmse={metric(tag+'_s3','rmse')}")
