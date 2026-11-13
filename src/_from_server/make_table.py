"""make_table.py - 把 eval_sweep_*.jsonl 汇总成 markdown 表格（不需要 torch）。

用法：
  python make_table.py eval_sweep_test.jsonl
"""
import json
import sys
from collections import defaultdict

PAPER = {
    # 论文 Table III / Setup 1 的 NMSE 与 RMSE
    "AE": (0.0456, 0.0758),
    "U-Net": (0.0063, 0.0182),
    "RadioUNet": (0.0049, 0.0160),
    "Pix2pix+": (0.0016, 0.0106),
    "RME-GAN": (0.0103, 0.0153),
    "MFFGCN": (0.0043, 0.0150),
}

# tag 后缀 -> setup 名
VARIANT_LABEL = {
    "ab_single": "U-Net 单编码器（基线）",
    "ab_nogcn": "去掉 GCN",
    "ab_static": "GCN 改静态邻接",
    "ab_nores": "去掉残差细化",
    "ab_cross": "AFM 改 cross-attention",
    "ab_concat": "AFM 改 concat",
    "ab_add": "AFM 改 add",
    "ab_channel": "AFM 改 sigmoid 门控",
    "afm_spatial": "AFM 改 spatial",
    "bs15_s1": "完整 MFFGCN",
    "FINAL": "完整 MFFGCN（正式）",
}


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_tag(tag):
    """把 tag 拆成 (variant, setup)"""
    if tag.startswith("FINAL_s"):
        return "FINAL", tag[-2:]
    if tag.startswith("afm_spatial_s"):
        return "afm_spatial", tag[-2:]
    for v in ("ab_single", "ab_nogcn", "ab_static", "ab_nores",
              "ab_cross", "ab_concat", "ab_add", "ab_channel"):
        if tag.startswith(v + "_s"):
            return v, tag[-2:]
    return tag, "s1"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "eval_sweep_test.jsonl"
    rows = load(path)
    byk = {(r["tag"], r["which"]): r for r in rows}

    print("## 0. best.pt 为何不可信（同一 checkpoint 目录两种权重对比）\n")
    print("| run | best 轮次 | best NMSE | last 轮次 | last NMSE |")
    print("|---|---|---|---|---|")
    for tag in sorted({r["tag"] for r in rows}):
        b, l = byk.get((tag, "best")), byk.get((tag, "last"))
        if not (b and l):
            continue
        flag = " ←" if b["nmse"] > l["nmse"] * 1.5 else ""
        print("| %s | %s | %.4f | %s | %.4f |%s"
              % (tag, b["epoch"], b["nmse"], l["epoch"], l["nmse"], flag))

    print("\n## 1. 主结果：五个采样 setup（last.pt，50 epoch）\n")
    print("| setup | best 轮次 | NMSE | RMSE | 粗估计 NMSE | 参数量 M |")
    print("|---|---|---|---|---|---|")
    for s in ("s1", "s2", "s3", "s4", "s5"):
        r = byk.get(("FINAL_" + s, "last"))
        if r:
            print("| %s | %s | %.4f | %.4f | %.4f | %.2f |"
                  % (s, r["epoch"], r["nmse"], r["rmse"],
                     r["coarse_nmse"], r["params_m"]))

    print("\n## 2. 消融（按 setup 并排，last.pt）\n")
    print("| 配置 | s1 | s2 | s3 | 参数量 M |")
    print("|---|---|---|---|---|")
    order = ["bs15_s1", "FINAL", "ab_single", "ab_nogcn", "ab_static",
             "ab_nores", "ab_cross", "ab_concat", "ab_add", "ab_channel",
             "afm_spatial"]
    for v in order:
        cells, par = [], "—"
        for s in ("s1", "s2", "s3"):
            tag = v if v == "bs15_s1" else "%s_%s" % (v, s)
            r = byk.get((tag, "last"))
            if r:
                cells.append("%.4f" % r["nmse"])
                par = "%.2f" % r["params_m"]
            else:
                cells.append("—")
        print("| %s | %s | %s | %s | %s |"
              % (VARIANT_LABEL.get(v, v), cells[0], cells[1], cells[2], par))

    print("\n## 3. 与论文 Table III（Setup 1）对照\n")
    print("| 方法 | 论文 NMSE | 论文 RMSE | 本次 NMSE | 本次 RMSE |")
    print("|---|---|---|---|---|")
    mapping = [("AE", None), ("U-Net", "calib_single"), ("RadioUNet", None),
               ("Pix2pix+", None), ("RME-GAN", None),
               ("MFFGCN", "FINAL_s1")]
    for name, tag in mapping:
        pn, pr = PAPER[name]
        if tag and (tag, "last") in byk:
            r = byk[(tag, "last")]
            print("| %s | %.4f | %.4f | %.4f | %.4f |"
                  % (name, pn, pr, r["nmse"], r["rmse"]))
        else:
            print("| %s | %.4f | %.4f | 未复现 | — |" % (name, pn, pr))

    print("\n## 4. 校准实验（论文协议 obs_on_signal=False）\n")
    print("| run | NMSE | RMSE | obs_on_signal |")
    print("|---|---|---|---|")
    for tag in ("calib_single", "calib_mffgcn", "calib_sigmoid"):
        for w in ("best", "last"):
            r = byk.get((tag, w))
            if r:
                print("| %s (%s) | %.4f | %.4f | %d |"
                      % (tag, w, r["nmse"], r["rmse"], r["obs_on_signal"]))


if __name__ == "__main__":
    main()
