# -*- coding: utf-8 -*-
"""表 V（消融）现版汇总 —— 直接从服务器 runs/*/test_metrics.json 取数。

为什么不用 compare_all.py：
    那个脚本读的是 `eval_ALL/metrics_*.json`，而 eval_ALL 是 11-01～11-03 的旧目录
    （坐标 bug 修复前的口径，add_s2 的 NMSE 还是 0.0794）。现在消融全部由
    `train.py` 在训练结束时用**全量 test 集**（n=8080）算好写进 test_metrics.json，
    这才是可比的同口径数据，所以这里直接读 run 目录。

行对应关系（论文表 V 有 8 行 + 本文完整模型）：
    Single Encoder   -> txf_single_s*        （早期融合单编码器，主表里叫 U-Net）
    Additive         -> txf_ab_add_s*
    Concatenation    -> txf_ab_concat_s*
    Channel Attention-> txf_ab_channel_s*    （SE，本轮新补）
    Cross-attention  -> txf_ab_cross_s*
    w/o GCN          -> txf_ab_nogcn_s*
    w/o Dynamic      -> txf_ab_static_s*
    w/o Residual     -> txf_ab_nores_s*
    Proposed         -> txf_mffgcn_s*        （softmax 门控 + 动态图 + 残差）

用法：
    python table_v.py                # 取服务器数据，出 md / json / png
    python table_v.py --offline      # 用 results/runs_snapshot.json 里上次的快照
"""
import argparse
import json
import os
import subprocess
import sys

REMOTE = "zjut148"
BASE = "/public2/czz/ZhangYu/project/mffgcn"

# 论文表 V 的 NMSE（s1, s2, s3），来自 compare_all.py 的 PAPER_T5
PAPER_T5 = {
    "single":  ("Single Encoder",    [0.0048, 0.0053, 0.0053]),
    "add":     ("Additive",          [0.0050, 0.0056, 0.0059]),
    "concat":  ("Concatenation",     [0.0047, 0.0053, 0.0053]),
    "channel": ("Channel Attention", [0.0045, 0.0051, 0.0053]),
    "cross":   ("Cross-attention",   [0.0046, 0.0052, 0.0053]),
    "nogcn":   ("w/o GCN",           [0.0049, 0.0054, 0.0055]),
    "static":  ("w/o Dynamic",       [0.0046, 0.0049, 0.0049]),
    "nores":   ("w/o Residual",      [0.0231, 0.0234, 0.0237]),
}
PAPER_FULL = [0.0043, 0.0045, 0.0046]
SETUPS = ["s1", "s2", "s3"]
ORDER = ["single", "add", "concat", "channel", "cross", "nogcn", "static", "nores"]


def ssh(cmd):
    for _ in range(2):
        p = subprocess.run(["ssh", REMOTE, cmd], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=180)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout
    print(f"[warn] 远程返回空: {(p.stderr or '').strip()[-200:]}", file=sys.stderr)
    return ""


def fetch():
    cmd = (f"cd {BASE}/src && for f in runs/*/test_metrics.json; do "
           f"python3 -c \"import json,sys;"
           f"print(sys.argv[1], json.dumps(json.load(open(sys.argv[1]))))\" "
           f"\"$f\" 2>/dev/null; done")
    runs = {}
    for line in ssh(cmd).splitlines():
        if not line.startswith("runs/"):
            continue
        path, _, payload = line.partition(" ")
        try:
            runs[path.split("/")[1]] = json.loads(payload)
        except json.JSONDecodeError:
            pass
    return runs


def pick(runs, table):
    """table: {(变体, setup): tag} -> 填上指标"""
    out = {}
    for key, tag in table.items():
        m = runs.get(tag)
        if not m:
            continue
        out[key] = {"tag": tag, "nmse": m.get("nmse"), "rmse": m.get("rmse"),
                    "coarse_nmse": m.get("coarse_nmse"),
                    "best_epoch": m.get("best_epoch"),
                    "n": m.get("n_images")}
    return out


def fetch_local(runs_dir="runs"):
    """在本机（服务器上）直接扫 runs/*/test_metrics.json。"""
    import glob
    out = {}
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "test_metrics.json"))):
        try:
            with open(p, encoding="utf-8") as fh:
                out[os.path.basename(os.path.dirname(p))] = json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="用 results/runs_snapshot.json 的快照，不连服务器")
    ap.add_argument("--local", action="store_true",
                    help="直接在服务器上扫 runs/ 目录（脚本本身跑在服务器时用）")
    ap.add_argument("--out", default="../../results")
    a = ap.parse_args()

    snap = os.path.join(a.out, "runs_snapshot.json")
    if a.local:
        runs = fetch_local()
    elif a.offline:
        with open(snap, encoding="utf-8") as fh:
            runs = json.load(fh)
    else:
        runs = fetch()
        os.makedirs(a.out, exist_ok=True)
        with open(snap, "w", encoding="utf-8") as fh:
            json.dump(runs, fh, ensure_ascii=False, indent=1)
    print(f"[info] runs={len(runs)}")

    # ---- 组装 ----
    table = {}
    for v in ORDER:
        for s in SETUPS:
            table[(v, s)] = f"txf_ab_{v}_{s}"
    table[("single", "s1")] = "txf_single_s1"
    table[("single", "s2")] = "txf_single_s2"
    table[("single", "s3")] = "txf_single_s3"
    for s in SETUPS:
        table[("full", s)] = f"txf_mffgcn_{s}"
    cur = pick(runs, table)

    # ---- md 表 ----
    lines = ["# 表 V（消融）现版 —— 同口径全量 test（n=8080）", "",
             "数值为 NMSE；括号内是相对论文表 V 的偏差。`—` 表示该 run 缺失。", "",
             "| 变体 | 论文对应行 | s1 | s2 | s3 |", "|---|---|---:|---:|---:|"]
    rows = [("full", "Proposed (full model)")] + [(v, PAPER_T5[v][0]) for v in ORDER]
    for key, label in rows:
        paper = PAPER_FULL if key == "full" else PAPER_T5[key][1]
        cells = []
        for j, s in enumerate(SETUPS):
            m = cur.get((key, s))
            if not m:
                cells.append("—")
                continue
            d = (m["nmse"] - paper[j]) / paper[j] * 100
            cells.append(f'{m["nmse"]:.5f} ({d:+.1f}%)')
        mark = "**" if key == "full" else ""
        lines.append(f"| {mark}{label}{mark} | {paper[0]:.4f} / {paper[1]:.4f} / {paper[2]:.4f} | "
                     + " | ".join(cells) + " |")
    lines += ["", "## RMSE（论文只报 NMSE，这里补上便于与时延/鲁棒性表对齐）", "",
              "| 变体 | s1 | s2 | s3 |", "|---|---:|---:|---:|"]
    for key, label in rows:
        cells = []
        for s in SETUPS:
            m = cur.get((key, s))
            cells.append(f'{m["rmse"]:.5f}' if m else "—")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    path_md = os.path.join(a.out, "table_v_current.md")
    with open(path_md, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)
    print("saved:", path_md)

    with open(os.path.join(a.out, "table_v_current.json"), "w", encoding="utf-8") as fh:
        json.dump({"+".join(k): v for k, v in cur.items()}, fh,
                  ensure_ascii=False, indent=1)

    # ---- 图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    C_P, C_O = "#2E7D59", "#C0392B"
    titles = ["Setup 1 (uniform)", "Setup 2 (25% restricted)", "Setup 3 (10-50% restricted)"]
    labels = [PAPER_T5[v][0] for v in ORDER] + ["Proposed (full model)"]
    y = np.arange(len(labels))[::-1]
    w = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), sharey=True)
    for j, (ax, s) in enumerate(zip(axes, SETUPS)):
        ours, paper = [], []
        for v in ORDER:
            m = cur.get((v, s))
            ours.append(m["nmse"] if m else np.nan)
            paper.append(PAPER_T5[v][1][j])
        m = cur.get(("full", s))
        ours.append(m["nmse"] if m else np.nan)
        paper.append(PAPER_FULL[j])
        ax.barh(y + w / 2, paper, w, color=C_P, label="Paper (Table V)", zorder=3)
        ax.barh(y - w / 2, ours, w, color=C_O, label="Ours (fixed coords)", zorder=3)
        for yi, v in zip(y - w / 2, ours):
            if v == v:
                ax.text(v * 1.04, yi, f"{v:.4f}", va="center", fontsize=7.4, color="#333")
        ax.set_xscale("log")
        ax.set_xlim(0.003, 0.9)
        ax.set_title(titles[j], fontsize=11)
        ax.set_xlabel("NMSE (log scale)", fontsize=9)
        ax.grid(axis="x", color="#E8E8E8", lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9.4)
    axes[0].get_yticklabels()[-1].set_fontweight("bold")
    axes[0].get_yticklabels()[-1].set_color(C_O)
    axes[0].legend(fontsize=8.8, frameon=False, loc="lower right")
    fig.suptitle("Table V ablation: paper vs reproduction (full test, n=8080)",
                 fontsize=12.5, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    figdir = os.path.join(a.out, "..", "figs")
    os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, "fig3_ablation_current.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
