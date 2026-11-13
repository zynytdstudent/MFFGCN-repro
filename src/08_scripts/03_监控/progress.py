# -*- coding: utf-8 -*-
"""论文复现完成度统计 —— 按论文的图表格子逐格核对。

用法：
    python progress.py            # 只看汇总
    python progress.py -v         # 展开每一格的来源

统计口径（80 格）：
    Table III  主表       9 模型 × 3 场景          27
    Table V    消融       8 变体 × 3 场景          24
    Fig        深度剖面   3 场景                    3
    Fig        鲁棒性     6 档扰动 × 3 场景        18
    Fig        定性       3 场景                    3
    Table      复杂度     1                         1
    §          补充实验   4 项                      4

判定规则见 classify()。数据直接从服务器实时读取，不做本地缓存，
避免"脚本说完成了但服务器上其实是旧数据"。
"""

import json
import re
import subprocess
import sys

REMOTE = "zjut148"
BASE = "/public2/czz/ZhangYu/project/mffgcn"

# tag -> 主表模型名。只匹配无后缀的正式 run（排除 _seedN / 探针）。
MAIN_PATTERNS = [
    (r"^txf_mffgcn_s([123])$",  "MFFGCN"),
    (r"^txf_single_s([123])$",   "U-Net"),
    (r"^bl_radiounet_s([123])$", "RadioUNet"),
    (r"^bl_pix2pix_s([123])$",   "Pix2Pix+"),
    (r"^bl_rmegan_s([123])$",    "RME-GAN"),
    (r"^bl_ae_s([123])$",        "AE"),
    (r"^bl_deepae_s([123])$",    "DeepAE"),
]

MAIN_MODELS = ["MFFGCN", "U-Net", "RadioUNet", "Pix2Pix+",
               "RME-GAN", "AE", "DeepAE", "KNN", "Kriging"]

# 消融变体：旧 ab_* 前缀（s1 可用，s2/s3 是 tx 坐标 bug 期数据）
AB_VARIANTS = ["concat", "add", "sigmoid", "cross",
               "nogcn", "static", "nores", "channel"]

# 鲁棒性计入分母的 6 个扰动点。clean 只是曲线参照点，不单独算一格。
ROB_POINTS = ["snr30", "snr20", "snr10", "bud0.5", "bud2.0", "bud4.0"]


def ssh(cmd, retries=1):
    """在服务器上执行命令，返回 stdout；失败重试一次并报警。

    以前这里写的是 `return p.stdout or ""`，连接被拒/超时时会把失败静默成
    空字符串，屏幕上看就是"已完成 run 0 个"，像数据丢了。并发跑两条 ssh 时
    最容易踩到（Windows OpenSSH 的会话数限制），所以失败必须可见。
    """
    last = None
    for _ in range(retries + 1):
        p = subprocess.run(["ssh", REMOTE, cmd],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=180)
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout
        last = p
    err = (last.stderr or "").strip().splitlines()
    print(f"[warn] 远程命令返回空 (rc={last.returncode}): "
          f"{err[-1] if err else '无 stderr'}", file=sys.stderr)
    return last.stdout or ""


def fetch_runs():
    """取所有 runs/*/test_metrics.json，返回 {tag: metrics}。

    服务端先用 python3 把每个 json 压成一行再回传 —— 多行 json 直接按行
    解析会把内容截断，之前就是这么漏掉 55 个结果的。
    """
    cmd = (f"cd {BASE}/src && for f in runs/*/test_metrics.json; do "
           f"python3 -c \"import json,sys;"
           f"print(sys.argv[1], json.dumps(json.load(open(sys.argv[1]))))\" "
           f"\"$f\" 2>/dev/null; done")
    out = ssh(cmd)
    runs = {}
    for line in out.splitlines():
        if not line.startswith("runs/"):
            continue
        path, _, payload = line.partition(" ")
        try:
            runs[path.split("/")[1]] = json.loads(payload)
        except json.JSONDecodeError:
            pass
    return runs


def fetch_rob():
    """取 eval_ROB 下已完成的鲁棒性配置名（s1/s2/s3 × 各档位）。"""
    out = ssh(f"cd {BASE} && ls eval_ROB/ 2>/dev/null")
    names = set()
    for line in out.splitlines():
        m = re.search(r"metrics_(rob_s[123]_[\w.]+)\.json", line.strip())
        if m:
            names.add(m.group(1))
    return names


def fetch_interp():
    """取插值基线结果，返回 {('KNN','s1'): rmse, ...}。"""
    cmd = (f"cd {BASE}/src && for f in interp_*.json; do "
           f"python3 -c \"import json,sys;"
           f"print(sys.argv[1], json.dumps(json.load(open(sys.argv[1]))))\" "
           f"\"$f\" 2>/dev/null; done")
    out = ssh(cmd)
    res = {}
    for line in out.splitlines():
        if not line.startswith("interp_"):
            continue
        path, _, payload = line.partition(" ")
        m = re.search(r"(knn|kriging)_s([123])", path, re.I)
        if not m:
            continue
        try:
            d = json.loads(payload)
        except json.JSONDecodeError:
            continue
        name = "KNN" if m.group(1).lower() == "knn" else "Kriging"
        # 字段是嵌套的 fine.all_rmse，不是顶层 rmse
        fine = d.get("fine") or {}
        res[(name, "s" + m.group(2))] = fine.get("all_rmse") or d.get("rmse")
    return res


def build_main(runs, interp):
    """主表 9×3。返回 {(模型, 场景): 数值}，缺失为 None。"""
    grid = {(m, s): None for m in MAIN_MODELS for s in ("s1", "s2", "s3")}
    for tag, met in runs.items():
        for pat, model in MAIN_PATTERNS:
            m = re.match(pat, tag)
            if m and isinstance(met, dict):
                grid[(model, "s" + m.group(1))] = met.get("rmse")
                break
    for (name, setup), rmse in interp.items():
        grid[(name, setup)] = rmse
    return grid


def build_ablation(runs):
    """消融 8×3。返回 {(变体, 场景): ('ok'|'stale', rmse)}。

    旧 ab_*_s1 是坐标 bug 修复后用旧权重重评的，只能做相对排序，
    标记 stale；s2/s3 全部丢弃（NMSE 0.09–0.13，是修复后口径的 20 倍）。
    新的 txf_ab_* 才是干净数据。
    """
    cell = {}
    for tag, met in runs.items():
        m = re.match(r"^txf_ab_(\w+?)_s([123])$", tag)
        if m and isinstance(met, dict) and met.get("rmse"):
            cell[(m.group(1), "s" + m.group(2))] = ("ok", met["rmse"])
            continue
        m = re.match(r"^ab_(\w+?)_s([123])$", tag)
        if m and isinstance(met, dict) and met.get("rmse"):
            key = (m.group(1), "s" + m.group(2))
            if key in cell:
                continue
            # 只认 s1，且标为口径待确
            if m.group(2) == "1":
                cell[key] = ("stale", met["rmse"])
    return cell


def main():
    verbose = "-v" in sys.argv

    print("读取服务器数据 ...", flush=True)
    runs = fetch_runs()
    rob = fetch_rob()
    interp = fetch_interp()
    print(f"  已完成 run {len(runs)} 个，鲁棒性配置 {len(rob)} 个，"
          f"插值结果 {len(interp)} 个\n")

    # ---- 1. 主表 ----
    grid = build_main(runs, interp)
    main_done = sum(1 for v in grid.values() if v)
    print("Table III 主表（RMSE，越低越好）")
    print("  " + "模型".ljust(12) + "".join(s.rjust(11) for s in ("s1", "s2", "s3")))
    for m in MAIN_MODELS:
        row = m.ljust(12)
        for s in ("s1", "s2", "s3"):
            v = grid[(m, s)]
            row += (f"{v:.5f}".rjust(11) if v else "—".rjust(11))
        print("  " + row)
    print(f"  小计 {main_done}/27\n")

    # ---- 2. 消融 ----
    abcell = build_ablation(runs)
    ab_ok = sum(1 for v in abcell.values() if v[0] == "ok")
    ab_stale = sum(1 for v in abcell.values() if v[0] == "stale")
    if verbose:
        print("Table V 消融")
        for var in AB_VARIANTS:
            row = "  " + var.ljust(12)
            for s in ("s1", "s2", "s3"):
                v = abcell.get((var, s))
                if not v:
                    row += "—".rjust(11)
                else:
                    tag = "" if v[0] == "ok" else "*"
                    row += f"{v[1]:.5f}{tag}".rjust(11)
            print(row)
        print("  * = 旧权重口径，只能用于变体间相对排序")
    print(f"Table V 消融  小计 干净 {ab_ok}/24，旧口径 {ab_stale}/24\n")

    # ---- 3. 鲁棒性 ----
    rob_done = sum(1 for s in ("s1", "s2", "s3")
                   for c in ROB_POINTS if f"rob_{s}_{c}" in rob)
    if verbose:
        for s in ("s1", "s2", "s3"):
            got = [c for c in ROB_POINTS if f"rob_{s}_{c}" in rob]
            miss = [c for c in ROB_POINTS if c not in got]
            print(f"  {s}: {len(got)}/6  已有 {' '.join(got) or '空'} | "
                  f"缺 {' '.join(miss) or '无'}")
    print(f"鲁棒性        小计 {rob_done}/18\n")

    # ---- 4. 手工项 ----
    # 这些没法从文件名自动判断，状态随人工确认更新
    MANUAL = {
        "深度剖面": (2, 3, "s2/s3 各一张 fig_gap_sN.png；s1 无盲区，分带无定义"),
        "定性可视化": (3, 3, "fig5_quali.png（三 setup 纵向拼接）+ quali_sN.png"),
        "复杂度": (1, 1, "参数量/GMACs 已算，口径待与论文统一"),
        # 补充实验按论文实际章节拆四项：表 VI 形状 / 表 VII λ / Fig.14-16 环境退化 / 表 VIII 时延
        "补充实验": (2, 4, "形状(表VI)✅ 环境退化(Fig14-16)✅；λ 与 时延 在排队等卡"),
    }
    for k, (d, t, note) in MANUAL.items():
        print(f"{k:<14}小计 {d}/{t}  ({note})")

    # ---- 汇总 ----
    man = {k: v[0] for k, v in MANUAL.items()}
    total = 27 + 24 + 18 + 3 + 3 + 1 + 4
    # 主表/消融/鲁棒性/深度/定性/复杂度/补充
    got = (main_done + ab_ok + rob_done + man["深度剖面"]
           + man["定性可视化"] + man["复杂度"] + man["补充实验"])
    got_loose = (main_done + ab_ok + ab_stale + rob_done + man["深度剖面"]
                 + man["定性可视化"] + man["复杂度"] + man["补充实验"])
    print("\n" + "=" * 44)
    print(f"严格口径（旧口径消融不计）  {got}/{total} = {got / total:.0%}")
    print(f"宽松口径（旧口径消融也计）  {got_loose}/{total} = {got_loose / total:.0%}")
    print("=" * 44)


if __name__ == "__main__":
    main()
