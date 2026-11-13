"""把重评结果、插值基线、新训练结果汇总成论文表 III / 表 V 的复现对照表。

数据来源（全部在服务器 src/ 下）：
    reeval_all.json      修复坐标 bug 后重评的 24 个消融 checkpoint
    reeval_extra.json    完整模型与标定组的重评
    interp_*.json        KNN / Kriging 两个插值基线
    runs/<tag>/test_metrics.json   修复后重新训练的主线（只认 txf_/bl_/ab2_）

优先级：新训练 > 旧权重重评。旧的 test_metrics.json 是错误坐标下算出来的，
一律不读。

用法：
    python agg_report.py --out 05-完整对比表.md
"""
import argparse
import glob
import json
import os

NEW_PREFIX = ("txf_", "bl_", "ab2_")
BANDS = ["0-2", "2-4", "4-8", "8-16", "16-32", "32+"]


def load_json(p, default=None):
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return default if default is not None else []


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    return f"{v:.{nd}f}"


def merge_bands(b):
    """把 0-1 与 1-2 两档合并成 0-2（盲区内部离边界最近的那一圈）。"""
    if not b:
        return {}
    out = {}
    se, px = 0.0, 0
    for k in ("0-1", "1-2"):
        if k in b:
            rm = b[k].get("rmse")
            n = b[k].get("px", 0)
            if rm is not None:
                se += rm ** 2 * n
                px += n
    if px:
        out["0-2"] = (se / px) ** 0.5
    for k in ("2-4", "4-8", "8-16", "16-32", "32+"):
        if k in b:
            out[k] = b[k].get("rmse")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".")
    ap.add_argument("--out", default="05-完整对比表.md")
    a = ap.parse_args()

    re_ = {}
    for f in ("reeval_all.json", "reeval_extra.json"):
        for r in load_json(os.path.join(a.src, f), []):
            re_[(r["tag"], r.get("setup"))] = r

    interp = {}
    for m in ("knn", "kriging"):
        for s in ("s1", "s2", "s3"):
            d = load_json(os.path.join(a.src, f"interp_{m}_{s}.json"), None)
            if d:
                interp[(m, s)] = d

    new_runs = {}
    for p in glob.glob(os.path.join(a.src, "runs", "*", "test_metrics.json")):
        tag = os.path.basename(os.path.dirname(p))
        if tag.startswith(NEW_PREFIX):
            d = load_json(p, {})
            if d:
                new_runs[tag] = d

    SETS = {"s1": "均匀撒点", "s2": "中等盲区", "s3": "大盲区"}
    lines = ["# 论文四大实验的复现对照表", "",
             "全部数字取自修复发射机坐标 bug 之后。", "",
             "口径提示：标 `*` 的行来自旧权重重评，训练时观测点只放在有信号的",
             "位置；新训练的行用论文口径（全图均匀撒点）。实测两者相差 4–9%，",
             "所以横向比较时以同一来源内部对比为准。", ""]

    def cell(new_tag, old_tag, setup):
        # 新训练的 run 目前只有 setup=1 一种口径，不能拿去填 s2/s3 的行，
        # 否则同一组数字会在三个场景里重复出现，看着像跑过其实没跑。
        d = new_runs.get(new_tag) if setup == "s1" else None
        if d:
            return (d.get("nmse"), d.get("rmse"), None, "新训练", "")
        r = re_.get((old_tag, setup))
        if r:
            f = r["fine"]
            return (f.get("all_nmse"), f.get("all_rmse"),
                    f.get("constr_rmse"), "重评", "*")
        return (None, None, None, "—", "")

    # ---------------- 表 III ----------------
    lines += ["## 一、基准性能对比（对应论文表 III）", "",
              "| 模型 | 场景 | NMSE | RMSE | 盲区 RMSE | 来源 |",
              "|---|---|---|---|---|---|"]
    for setup in ("s1", "s2", "s3"):
        for name, ntag, otag in (("MFFGCN（本文）", "txf_mffgcn_s1", "FINAL_s1"),
                                 ("U-Net（单编码器）", "txf_single_s1", "ab_single_s1"),
                                 ("AE", "bl_ae_s1", None),
                                 ("DeepAE", "bl_deepae_s1", None),
                                 ("RadioUNet", "bl_radiounet_s1", None),
                                 ("Pix2pix+", "bl_pix2pix_s1", None),
                                 ("RME-GAN", "bl_rmegan_s1", None)):
            nm, rm, cr, src, star = cell(ntag, otag, setup)
            if nm is None:
                continue
            lines.append(f"| {name}{star} | {SETS[setup]} | {fmt(nm)} | "
                         f"{fmt(rm)} | {fmt(cr)} | {src} |")
        for m, nm2 in (("knn", "KNN"), ("kriging", "Kriging")):
            d = interp.get((m, setup))
            if d:
                f = d["fine"]
                lines.append(f"| {nm2} | {SETS[setup]} | "
                             f"{fmt(f.get('all_nmse'))} | "
                             f"{fmt(f.get('all_rmse'))} | "
                             f"{fmt(f.get('constr_rmse'))} | 插值实现 |")

    # ---------------- 表 V ----------------
    VARIANTS = [
        ("完整模型", "FINAL"),
        ("换融合：拼接", "ab_concat"),
        ("换融合：相加", "ab_add"),
        ("换融合：sigmoid 门控", "ab_channel"),
        ("换融合：交叉注意力", "ab_cross"),
        ("去掉图卷积", "ab_nogcn"),
        ("静态图（不自适应）", "ab_static"),
        ("去掉残差", "ab_nores"),
        ("单编码器", "ab_single"),
    ]
    lines += ["", "## 二、消融验证（对应论文表 V）", "",
              "| 变体 | 均匀场景 RMSE | 中等盲区 RMSE | 大盲区 RMSE |",
              "|---|---|---|---|"]
    for label, base in VARIANTS:
        vals = []
        for s in ("s1", "s2", "s3"):
            r = re_.get((f"{base}_{s}", s))
            vals.append(fmt(r["fine"].get("all_rmse")) if r else "—")
        lines.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} |")

    # ---------------- 盲区深度 ----------------
    lines += ["", "## 三、盲区专项：误差随深度的变化", "",
              "行是变体，列是盲区内像素离最近可见区的距离（像素）。"
              "读数方法：若左侧（贴近边界）明显高于右侧（盲区中心），"
              "说明误差主要来自边界定位而不是盲区内部无信息可用。", "",
              "| 变体 | " + " | ".join(BANDS) + " |",
              "|" + "---|" * (len(BANDS) + 1)]
    for label, base in VARIANTS:
        for s in ("s2", "s3"):
            r = re_.get((f"{base}_{s}", s))
            if not r:
                continue
            mb = merge_bands(r.get("bands"))
            if not mb:
                continue
            lines.append(f"| {label}（{SETS[s]}） | " +
                         " | ".join(fmt(mb.get(k)) for k in BANDS) + " |")

    # ---------------- 复杂度 ----------------
    lines += ["", "## 四、复杂度（对应论文表 VI–VIII）", "",
              "| 变体 | 参数量 | GMACs |", "|---|---|---|",
              "| MFFGCN（本文） | 24.55 M | 29.9 |",
              "| 去掉自适应图 | 24.55 M | 29.9 |",
              "| 去掉图卷积 | 24.15 M | 29.8 |",
              "| 单编码器 | 7.76 M | 11.5 |",
              "", "论文报 40.32 G / 5.17 ms。MACs×2 = 59.8 GFLOPs，与论文口径",
              "不一致（论文可能把 kNN 图构建也算进去了），时延需要在空闲卡上",
              "重测才准。"]

    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[ok] wrote", a.out, f"({len(lines)} 行)")


if __name__ == "__main__":
    main()
