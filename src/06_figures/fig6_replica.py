"""复刻论文 Fig.6：MFFGCN vs RadioUNet 训练 loss 曲线，三个 setup 并排。

严格对齐论文版式：3 个子图横排，横轴 Epoch，纵轴 L2 Loss，
红线 MFFGCN、蓝线 RadioUNet。差异在于我们只训练了 50 epoch（论文 100）。
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/public2/czz/ZhangYu/project/mffgcn/src/runs"
OUT = "/tmp/fig6_ours.png"
OUT_NORM = "/tmp/fig6_ours_norm.png"


def load(path, col="train_loss"):
    if not os.path.exists(path):
        return []
    rows = list(csv.DictReader(open(path)))
    pts = []
    for r in rows:
        try:
            pts.append((float(r["epoch"]), float(r[col])))
        except (KeyError, ValueError):
            pass
    return pts


SETUPS = ["s1", "s2", "s3"]
LABELS = ["(a)", "(b)", "(c)"]

# ---------- 版本 1：原始口径 ----------
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.4))
for ax, s, lab in zip(axes, SETUPS, LABELS):
    m = load(f"{BASE}/txf_mffgcn_{s}/train_log.csv")
    r = load(f"{BASE}/bl_radiounet_{s}/train_log_stage2.csv")
    if m:
        ax.plot([p[0] for p in m], [p[1] for p in m],
                color="#d62728", lw=1.7, label="MFFGCN train loss")
    if r:
        ax.plot([p[0] for p in r], [p[1] for p in r],
                color="#1f77b4", lw=1.7, label="RadioUNet train loss")
    ax.set_xlim(0, 50)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("L2 Loss", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(lab, y=-0.38, fontsize=13)
fig.suptitle("Ours: MFFGCN vs RadioUNet training loss (50 epochs, our loss definition)",
             fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print("saved", OUT)

# ---------- 版本 2：归一化到各自第 1 轮 ----------
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.4))
for ax, s, lab in zip(axes, SETUPS, LABELS):
    m = load(f"{BASE}/txf_mffgcn_{s}/train_log.csv")
    r = load(f"{BASE}/bl_radiounet_{s}/train_log_stage2.csv")
    if m:
        b = m[0][1]
        ax.plot([p[0] for p in m], [p[1] / b for p in m],
                color="#d62728", lw=1.7, label="MFFGCN")
    if r:
        b = r[0][1]
        ax.plot([p[0] for p in r], [p[1] / b for p in r],
                color="#1f77b4", lw=1.7, label="RadioUNet")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("loss / loss(epoch 1)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(lab, y=-0.38, fontsize=13)
fig.suptitle("Normalized to epoch-1 loss (shape comparison, magnitude removed)",
             fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(OUT_NORM, dpi=170, bbox_inches="tight")
print("saved", OUT_NORM)

# 打点关键数字
for s in SETUPS:
    m = load(f"{BASE}/txf_mffgcn_{s}/train_log.csv")
    r = load(f"{BASE}/bl_radiounet_{s}/train_log_stage2.csv")
    print(f"{s}: MFFGCN {m[0][1]:.5f}->{m[-1][1]:.5f} | "
          f"RadioUNet {r[0][1]:.5f}->{r[-1][1]:.5f}")
