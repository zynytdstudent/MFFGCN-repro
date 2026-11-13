"""gen_report_figs.py - 用当前最新数据重画报告图（在服务器上运行，CPU 即可）。

数据来源：
  src/runs/*/test_metrics.json          训练完成后的测试指标
  eval_ROB/metrics_rob_*.json           鲁棒性 21 个配置
  src/interp_*.json                     插值基线

输出：/tmp/report_figs/*.png
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/public2/czz/ZhangYu/project/mffgcn"
SRC = BASE + "/src"
OUT = "/tmp/report_figs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

C_MAIN, C_BASE, C_GRAY, C_ACC = "#2E5A88", "#C0504D", "#9AA0A6", "#E8A33D"
C3 = ["#2E5A88", "#E8A33D", "#5B9A5B"]


def load_runs():
    out = {}
    for p in glob.glob(SRC + "/runs/*/test_metrics.json"):
        tag = p.split("/")[-2]
        try:
            out[tag] = json.load(open(p, encoding="utf-8"))
        except Exception:
            pass
    return out


def load_rob():
    out = {}
    for p in glob.glob(BASE + "/eval_ROB/metrics_rob_*.json"):
        try:
            m = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        out[m["name"]] = m
    return out


def load_interp():
    out = {}
    for p in glob.glob(SRC + "/interp_*.json"):
        try:
            m = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        out[(m["method"].lower(), m["setup"])] = m["fine"]
    return out


R, ROB, ITP = load_runs(), load_rob(), load_interp()

# ---------- 主表：模型 -> setup -> tag ----------
MAIN = [
    ("MFFGCN",     ["txf_mffgcn_s1", "txf_mffgcn_s2", "txf_mffgcn_s3"]),
    ("U-Net",      ["txf_single_s1", None, None]),
    ("RadioUNet",  ["bl_radiounet_s1", None, None]),
    ("Pix2Pix+",   ["bl_pix2pix_s1", "bl_pix2pix_s2", None]),
    ("RME-GAN",    ["bl_rmegan_s1", None, None]),
]


def nmse(tag):
    if not tag or tag not in R:
        return None
    return R[tag].get("nmse")


# ================= 图 1：主表三场景（深度模型） =================
fig, ax = plt.subplots(figsize=(8.2, 4.2))
n = len(MAIN)
w = 0.26
xs = np.arange(n)
colors = [C_MAIN, C_ACC, "#5B9A5B"]
for j, s in enumerate(["s1", "s2", "s3"]):
    vals = [nmse(m[1][j]) or 0 for m in MAIN]
    mask = [nmse(m[1][j]) is not None for m in MAIN]
    pos = xs + (j - 1) * w
    bars = ax.bar([p for p, k in zip(pos, mask) if k],
                  [v for v, k in zip(vals, mask) if k],
                  width=w * 0.92, color=colors[j],
                  label=f"Setup {j+1}", zorder=3)
    for b, v in zip(bars, [v for v, k in zip(vals, mask) if k]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0003, f"{v:.4f}",
                ha="center", va="bottom", fontsize=7)

ax.set_xticks(xs)
ax.set_xticklabels([m[0] for m in MAIN])
ax.set_ylabel("NMSE (lower is better)")
ax.set_title("Main comparison across three sampling setups\n"
             "(clean observations, best.pt, txf_* / bl_* runs)", fontsize=11)
ax.legend(frameon=False, ncol=3, loc="upper left")
ax.set_ylim(0, 0.0168)
fig.savefig(OUT + "/figA_main.png")
plt.close(fig)

# ================= 图 2：深度学习 vs 插值（log 轴） =================
fig, ax = plt.subplots(figsize=(7.6, 4.2))
groups = [
    ("MFFGCN",   [nmse("txf_mffgcn_s1"), nmse("txf_mffgcn_s2"), nmse("txf_mffgcn_s3")], C_MAIN, "o", "-", (0, 10)),
    ("DeepAE",   [nmse("bl_deepae_s1"), nmse("bl_deepae_s2"), None], "#7F77DD", "s", "--", (0, 10)),
    ("KNN",      [ITP.get(("knn", s), {}).get("all_nmse") for s in ("s1", "s2", "s3")], C_ACC, "^", "-", (0, 11)),
    ("Kriging",  [ITP.get(("kriging", s), {}).get("all_nmse") for s in ("s1", "s2", "s3")], C_BASE, "v", "-", (0, -17)),
]
for name, vals, c, mk, ls, off in groups:
    xs_ = [i for i, v in enumerate(vals) if v]
    ys_ = [v for v in vals if v]
    ax.plot(xs_, ys_, marker=mk, color=c, linestyle=ls, label=name,
            linewidth=1.8, markersize=6, zorder=3)
    for idx, (x_, y_) in enumerate(zip(xs_, ys_)):
        if idx not in (0, len(xs_) - 1):
            continue
        ax.annotate(f"{y_:.4f}", (x_, y_), textcoords="offset points",
                    xytext=off, ha="center", fontsize=7.5, color=c)
ax.set_yscale("log")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Setup 1\n(uniform)", "Setup 2\n(25% blind)", "Setup 3\n(10-50% blind)"])
ax.set_ylabel("NMSE (log scale, lower is better)")
ax.set_title("Deep learning vs classical interpolation\nthe gap is roughly one order of magnitude",
             fontsize=11)
ax.legend(frameon=False, loc="center left")
fig.savefig(OUT + "/figB_interp.png")
plt.close(fig)

# ================= 图 3 / 4：鲁棒性 =================
def rob_val(setup, cfg):
    m = ROB.get(f"rob_{setup}_{cfg}")
    return m["fine"]["all_nmse"] if m else None


fig, ax = plt.subplots(figsize=(7.4, 4.2))
cfgs = ["clean", "snr30", "snr20", "snr10"]
labels = ["clean", "30 dB", "20 dB", "10 dB"]
for i, s in enumerate(["s1", "s2", "s3"]):
    ys = [rob_val(s, c) for c in cfgs]
    ax.plot(range(4), ys, marker="o", color=C3[i], linewidth=1.9,
            markersize=6, label=f"Setup {i+1}", zorder=3)
    ax.annotate(f"{ys[-1]:.4f}", (3, ys[-1]), textcoords="offset points",
                xytext=(6, -2), fontsize=8, color=C3[i])
ax.set_xticks(range(4))
ax.set_xticklabels(labels)
ax.set_xlabel("observation noise (SNR, lower = noisier)")
ax.set_ylabel("NMSE")
ax.set_title("Robustness to measurement noise\nsame weights, only the input is corrupted",
             fontsize=11)
ax.legend(frameon=False)
fig.savefig(OUT + "/figC_noise.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.4, 4.2))
cfgs = ["bud0.5", "clean", "bud2.0", "bud4.0"]
labels = ["x0.5", "x1 (base)", "x2", "x4"]
for i, s in enumerate(["s1", "s2", "s3"]):
    ys = [rob_val(s, c) for c in cfgs]
    ax.plot(range(4), ys, marker="s", color=C3[i], linewidth=1.9,
            markersize=6, label=f"Setup {i+1}", zorder=3)
    ax.annotate(f"{ys[-1]:.4f}", (3, ys[-1]), textcoords="offset points",
                xytext=(6, -2), fontsize=8, color=C3[i])
ax.set_xticks(range(4))
ax.set_xticklabels(labels)
ax.set_xlabel("observation budget multiplier")
ax.set_ylabel("NMSE")
ax.set_title("Robustness to observation budget\n"
             "each doubling of the sensor count lowers NMSE by roughly 20%",
             fontsize=11)
ax.legend(frameon=False)
fig.savefig(OUT + "/figD_budget.png")
plt.close(fig)

# ================= 图 5：消融（干净口径） =================
fig, ax = plt.subplots(figsize=(7.6, 4.0))
items = [
    ("MFFGCN (full)", nmse("txf_mffgcn_s1"), C_MAIN),
    ("AFM: sigmoid", nmse("txf_sigmoid_s1"), "#7F77DD"),
    ("AFM: spatial", nmse("txf_spatial_s1"), "#7F77DD"),
    ("Single encoder", nmse("txf_single_s1"), C_ACC),
]
base = items[0][1]
names = [i[0] for i in items]
vals = [i[1] for i in items]
cols = [i[2] for i in items]
bars = ax.barh(range(len(items))[::-1], vals, color=cols, height=0.62, zorder=3)
ax.set_yticks(range(len(items))[::-1])
ax.set_yticklabels(names)
for b, v in zip(bars, vals):
    dev = (v - base) / base * 100
    txt = "baseline" if abs(dev) < 1e-9 else f"{v:.5f}  (+{dev:.1f}%)"
    ax.text(v + 0.00012, b.get_y() + b.get_height() / 2, txt,
            va="center", fontsize=8.5)
ax.set_xlim(0, max(vals) * 1.32)
ax.set_xlabel("NMSE, Setup 1 (clean, current bug-free weights)")
ax.set_title("Ablation with clean weights\nonly four variants have re-trained data so far",
             fontsize=11)
fig.savefig(OUT + "/figE_ablation.png")
plt.close(fig)

# ================= 图 6：五种子稳定性 =================
fig, ax = plt.subplots(figsize=(6.6, 4.0))
def seed_vals(prefix):
    v = [nmse(prefix)]
    for k in range(1, 5):
        v.append(nmse(f"{prefix}_seed{k}"))
    return [x for x in v if x]

mff = seed_vals("txf_mffgcn_s1")
unet = seed_vals("txf_single_s1")
for i, (vals, c, lab) in enumerate([(mff, C_MAIN, "MFFGCN"), (unet, C_ACC, "Single-encoder U-Net")]):
    xs_ = np.full(len(vals), i) + np.linspace(-0.10, 0.10, len(vals))
    ax.scatter(xs_, vals, color=c, s=48, zorder=3, label=f"{lab} (n={len(vals)})")
    mu, sd = float(np.mean(vals)), float(np.std(vals))
    ax.hlines(mu, i - 0.26, i + 0.26, color=c, linewidth=2, zorder=4)
    ax.text(i + 0.30, mu, f"{mu:.5f} ± {sd:.5f}", va="center", fontsize=9, color=c)
ax.set_xticks([0, 1])
ax.set_xticklabels(["MFFGCN", "U-Net"])
ax.set_xlim(-0.6, 2.1)
ax.set_ylabel("NMSE, Setup 1")
ax.set_title("Seed variance: the two five-seed ranges do not overlap", fontsize=11)
ax.legend(frameon=False, loc="upper left")
ax.text(1.0, min(mff + unet) - 0.00006,
        f"worst MFFGCN {max(mff):.5f}  <  best U-Net {min(unet):.5f}",
        ha="center", fontsize=9, color=C_BASE)
fig.savefig(OUT + "/figF_seeds.png")
plt.close(fig)

# ================= 图 7：粗估计 vs 残差精修 =================
fig, ax = plt.subplots(figsize=(7.0, 4.0))
xs_ = np.arange(3)
coarse = [rob_val(s, "clean") and ROB[f"rob_{s}_clean"]["coarse"]["all_nmse"] for s in ("s1", "s2", "s3")]
fine = [rob_val(s, "clean") for s in ("s1", "s2", "s3")]
w = 0.34
b1 = ax.bar(xs_ - w / 2, coarse, width=w, color=C_GRAY, label="coarse (U-Net stage)", zorder=3)
b2 = ax.bar(xs_ + w / 2, fine, width=w, color=C_MAIN, label="fine (+ GCN residual)", zorder=3)
for bars, vals, c in [(b1, coarse, "#444441"), (b2, fine, C_MAIN)]:
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.00016, f"{v:.5f}",
                ha="center", fontsize=8, color=c)
for i, (c_, f_) in enumerate(zip(coarse, fine)):
    ax.text(i, max(c_, f_) * 1.16, f"-{(c_-f_)/c_*100:.1f}%", ha="center",
            fontsize=9, color=C_BASE)
ax.set_xticks(xs_)
ax.set_xticklabels(["Setup 1", "Setup 2", "Setup 3"])
ax.set_ylabel("NMSE (clean)")
ax.set_ylim(0, max(coarse) * 1.30)
ax.set_title("GCN refinement pays off more as the blind area grows\n"
             "same checkpoint, before and after refinement", fontsize=11)
ax.legend(frameon=False)
fig.savefig(OUT + "/figG_residual.png")
plt.close(fig)

print("done")
for f in sorted(glob.glob(OUT + "/*.png")):
    print(os.path.basename(f), os.path.getsize(f))
