# -*- coding: utf-8 -*-
"""扫描 runs/ 与评估 json，输出统一快照（供本地 update_trend.py 拉取）。

只用标准库，服务器上任何 python3 都能跑：
    cd /public2/czz/ZhangYu/project/mffgcn/src && python collect_trend.py
"""
import glob
import json
import os
import time

BASE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(BASE, "runs")


def rd(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def active_tags():
    """扫 /proc 找还在跑的 run：只认命令行里带 --tag <tag> 的 python 进程。

    光看 runs/<tag>/ 目录存在不够——autopilot 会先建目录写表头，
    进程被 OOM 杀掉后目录还留着，容易误报成「在训练」。
    """
    tags = set()
    for p in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            with open(p, "rb") as fh:
                parts = fh.read().decode("utf-8", "ignore").split("\x00")
        except Exception:
            continue
        if "--tag" in parts:
            i = parts.index("--tag")
            if i + 1 < len(parts):
                tags.add(parts[i + 1])
    return tags


ACTIVE = active_tags()


out = {
    "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    "complete": {},
    "running": {},
    "interp": {},
    "reeval": [],
}

for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
    if not os.path.isdir(d):
        continue
    tag = os.path.basename(d)
    tm = os.path.join(d, "test_metrics.json")
    if os.path.exists(tm):
        m = rd(tm) or {}
        out["complete"][tag] = {
            "rmse": m.get("rmse"),
            "nmse": m.get("nmse"),
            "epoch": m.get("best_epoch"),
            "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(tm))),
        }
        continue
    lg = os.path.join(d, "train_log.csv")
    if os.path.exists(lg):
        try:
            rows = [l.split(",") for l in open(lg).read().strip().split("\n")
                    if l and not l.startswith("epoch")]
            rows = [r for r in rows if len(r) >= 7]
            if rows:
                vals = [float(r[6]) for r in rows]
                out["running"][tag] = {
                    "epoch": int(float(rows[-1][0])),
                    "val_rmse": round(vals[-1], 5),
                    "best_val": round(min(vals), 5),
                    "best_epoch": int(float(rows[vals.index(min(vals))][0])),
                    "alive": tag in ACTIVE,
                }
            else:
                # 只有表头：要么刚开训，要么进程已经死了只留个空目录
                out["running"][tag] = {"epoch": 0, "val_rmse": None,
                                       "best_val": None, "best_epoch": None,
                                       "just_started": True,
                                       "alive": tag in ACTIVE}
        except Exception as e:  # 日志写一半时不要崩
            out["running"][tag] = {"error": str(e)}

for meth in ("knn", "kriging"):
    for s in ("s1", "s2", "s3"):
        m = rd(os.path.join(BASE, "interp_%s_%s.json" % (meth, s)))
        if m:
            out["interp"].setdefault(meth, {})[s] = (m.get("fine") or {}).get("all_rmse")

for f in ("reeval_all.json", "reeval_extra.json"):
    m = rd(os.path.join(BASE, f))
    if isinstance(m, list):
        for r in m:
            fine = r.get("fine") or {}
            out["reeval"].append({
                "tag": r.get("tag"),
                "setup": r.get("setup"),
                "rmse": fine.get("all_rmse"),
                "nmse": fine.get("all_nmse"),
                "epoch": r.get("best_epoch"),
                "arch": r.get("arch"),
                "afm": r.get("afm_mode"),
                "gcn": r.get("gcn_mode"),
                "no_residual": r.get("no_residual"),
            })

print("<<<JSON")
print(json.dumps(out, ensure_ascii=False))
print("JSON>>>")
