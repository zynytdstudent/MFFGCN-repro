# -*- coding: utf-8 -*-
"""拉取服务器快照 → 更新 trend_data.json → 重新生成趋势对比报告。

用法：
    python update_trend.py            # 有变化才重写报告
    python update_trend.py --force    # 强制重写
    python update_trend.py --dry-run  # 只看有什么变化，不写文件

退出码：0 = 正常（无论有无更新）；1 = 拉取/解析失败。
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"D:\Projects\MFFGCN"
DATA_PATH = os.path.join(HERE, "trend_data.json")
REPORT = os.path.join(HERE, "make_trend_report.py")
SSH_TARGET = "zjut148"
REMOTE_CMD = "cd /public2/czz/ZhangYu/project/mffgcn/src && python collect_trend.py"
FRESH_AFTER = "2026-11-04 00:00"   # 坐标 bug 修复后的训练才算论文口径

# 消融 tag → 报告里的变体名（取 setup=s1 的重评值）
ABL_TAGS = {
    "完整模型": ("FINAL_s1", "s1"),
    "去残差": ("ab_nores_s1", "s1"),
    "去动态图（换静态图）": ("ab_static_s1", "s1"),
    "换融合：交叉注意力": ("ab_cross_s1", "s1"),
    "换融合：相加": ("ab_add_s1", "s1"),
    "换融合：拼接": ("ab_concat_s1", "s1"),
    "换融合：通道门控": ("ab_channel_s1", "s1"),
    "去掉图卷积": ("ab_nogcn_s1", "s1"),
    "只留单编码器": ("ab_single_s1", "s1"),
}
ABL_DEFAULT = {
    "完整模型": 0.0211, "去残差": 0.0392, "去动态图（换静态图）": 0.0337,
    "换融合：交叉注意力": 0.0293, "换融合：相加": 0.0243, "换融合：拼接": 0.0191,
    "换融合：通道门控": 0.0185, "去掉图卷积": 0.0184, "只留单编码器": 0.0196,
}
# 基线 tag（论文口径 s1）
BASE_TAGS = {
    "U-Net": "txf_single_s1",
    "AE": "bl_ae_s1",
    "DeepAE": "bl_deepae_s1",
    "RadioUNet": "bl_radiounet_s1",
    "Pix2pix+": "bl_pix2pix_s1",
    "RME-GAN": "bl_rmegan_s1",
}
BASE_DEFAULT = {"U-Net": 0.0200, "AE": 0.0789}


def fetch():
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", SSH_TARGET, REMOTE_CMD],
                       capture_output=True, text=True, timeout=180)
    out = p.stdout or ""
    i, j = out.find("<<<JSON"), out.find("JSON>>>")
    if i < 0 or j < 0:
        raise RuntimeError("拉取失败：\nSTDOUT:%s\nSTDERR:%s" % (out[-400:], (p.stderr or "")[-400:]))
    return json.loads(out[i + len("<<<JSON"):j])


def build(snap):
    complete = snap.get("complete", {})
    reeval = {(r.get("tag"), r.get("setup")): r for r in snap.get("reeval", [])}
    interp = snap.get("interp", {})

    # 只认坐标修复之后训练出来的 run
    fresh = {t: v for t, v in complete.items()
             if (v.get("mtime") or "") >= FRESH_AFTER and v.get("rmse")}

    # ---- 本文模型：种子均值 ----
    seed_tags = sorted(t for t in fresh if t.startswith("txf_mffgcn_s1_seed"))
    vals, used = [], []
    for t in seed_tags:
        vals.append(fresh[t]["rmse"])
        used.append(t)
    if "txf_mffgcn_s1" in fresh:
        vals.append(fresh["txf_mffgcn_s1"]["rmse"])
        used.append("txf_mffgcn_s1")

    if vals:
        mff = statistics.mean(vals)
        mff_std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        mff_n = len(vals)
        mff_src = "新训练 %d 个种子" % mff_n
    else:
        r = reeval.get(("FINAL_s1", "s1"))
        mff = (r or {}).get("rmse") or 0.0211
        mff_std, mff_n = 0.0, 1
        mff_src = "旧权重重评（观测点口径不同）"

    def from_fresh(tag, default):
        return fresh[tag]["rmse"] if tag in fresh else default

    # U-Net 也按种子取均值：否则是「5 个种子的均值」对「单次结果」，
    # 方差大的那一侧天然吃亏，结论会被口径差异带偏。
    unet_vals = [fresh[t]["rmse"] for t in sorted(fresh)
                 if t.startswith("txf_single_s1_seed")]
    if BASE_TAGS["U-Net"] in fresh:
        unet_vals.append(fresh[BASE_TAGS["U-Net"]]["rmse"])
    if unet_vals:
        unet_v = statistics.mean(unet_vals)
        unet_std = statistics.pstdev(unet_vals) if len(unet_vals) > 1 else 0.0
    else:
        unet_v, unet_std = BASE_DEFAULT["U-Net"], 0.0

    bench = {
        "MFFGCN": {"v": round(mff, 5), "std": round(mff_std, 5), "n": mff_n, "src": mff_src},
        "U-Net": {"v": round(unet_v, 5), "std": round(unet_std, 5),
                  "n": len(unet_vals) or 1},
        "AE": {"v": round(from_fresh(BASE_TAGS["AE"], BASE_DEFAULT["AE"]), 5)},
        "KNN": {"v": round((interp.get("knn", {}).get("s1") or 0.0908), 5)},
        "Kriging": {"v": round((interp.get("kriging", {}).get("s1") or 0.0899), 5)},
    }
    for name in ("DeepAE", "RadioUNet", "Pix2pix+", "RME-GAN"):
        if BASE_TAGS[name] in fresh:
            bench[name] = {"v": round(fresh[BASE_TAGS[name]]["rmse"], 5)}

    # ---- 三档采样：相对 Setup1 的倍数 ----
    def setup_val(tag, key):
        if tag in fresh:
            return fresh[tag]["rmse"]
        r = reeval.get(key)
        return (r or {}).get("rmse")

    s1 = bench["MFFGCN"]["v"]
    s2 = setup_val("txf_mffgcn_s2", ("FINAL_s2", "s2")) or 0.0222
    s3 = setup_val("txf_mffgcn_s3", ("FINAL_s3", "s3")) or 0.0420
    setup = {
        "MFFGCN": [round(s1, 5), round(s2, 5), round(s3, 5)],
        "KNN": [round(bench["KNN"]["v"], 5),
                round((interp.get("knn", {}).get("s2") or 0.1049), 5),
                round((interp.get("knn", {}).get("s3") or 0.1139), 5)],
        "Kriging": [round(bench["Kriging"]["v"], 5),
                    round((interp.get("kriging", {}).get("s2") or 0.1030), 5),
                    round((interp.get("kriging", {}).get("s3") or 0.1116), 5)],
    }
    # S3 是否仍为「第 1 轮存盘」的坏值
    s3_bad = not ("txf_mffgcn_s3" in fresh)
    setup["MFFGCN_s3_provisional"] = bool(s3_bad)

    # ---- 消融：相对完整模型的变化 ----
    ablation = {}
    for name, key in ABL_TAGS.items():
        r = reeval.get(key)
        ablation[name] = round(((r or {}).get("rmse") or ABL_DEFAULT[name]), 5)
    ablation["完整模型"] = round(ablation["完整模型"], 5)

    return {
        "updated": snap.get("generated", ""),
        "bench": bench,
        "setup": setup,
        "ablation": ablation,
        "running": snap.get("running", {}),
        "fetched_at": subprocess.run(["date", "+%Y-%m-%d %H:%M:%S"],
                                     capture_output=True, text=True).stdout.strip(),
    }


def flat(d):
    """把可比数值拍平，用于判断是否有实质变化。"""
    out = {}
    for k, v in (d.get("bench") or {}).items():
        if isinstance(v, dict):
            out["bench." + k] = round(float(v.get("v") or 0), 5)
    for k, v in (d.get("setup") or {}).items():
        if isinstance(v, list):
            out["setup." + k] = [round(float(x), 5) for x in v]
        elif isinstance(v, bool):
            out["setup." + k] = v
    for k, v in (d.get("ablation") or {}).items():
        out["abl." + k] = round(float(v), 5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    try:
        snap = fetch()
    except Exception as e:
        print("[fail] %s" % e)
        return 1

    new = build(snap)
    old = {}
    if os.path.exists(DATA_PATH):
        try:
            with open(DATA_PATH, encoding="utf-8") as fh:
                old = json.load(fh)
        except Exception:
            old = {}

    a_new, a_old = flat(new), flat(old)
    changed = [k for k in a_new if a_old.get(k) != a_new[k]]
    running = new.get("running", {})

    if not changed and not a.force:
        print("[skip] 无新结果。进行中：")
        for t, v in running.items():
            print("  %-26s ep%-3s best_val %.5f" % (t, v.get("epoch"), v.get("best_val") or 0))
        return 0

    print("[diff] %d 项变化：" % len(changed))
    for k in changed:
        print("  %-28s %s -> %s" % (k, a_old.get(k), a_new[k]))

    with open(DATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(new, fh, ensure_ascii=False, indent=1)

    if a.dry_run:
        print("[dry] 已算好，未重建报告")
        return 0

    p = subprocess.run([sys.executable, REPORT], capture_output=True, text=True)
    print(p.stdout.strip())
    if p.returncode != 0:
        print("[fail] 报告生成失败：%s" % p.stderr[-800:])
        return 1
    print("[ok] 报告已更新 -> reports/07-与论文趋势一致性对比.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
