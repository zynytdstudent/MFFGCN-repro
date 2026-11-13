# -*- coding: utf-8 -*-
"""同一份 AE / DeepAE 权重，只改 skip 口径与观测点数，看 Fig.7 的斜率从哪来。

背景
----
论文 Fig.7 里 AE / DeepAE 随观测点数从 ~0.062 掉到 ~0.018（3.4x），
而我们的 skip=none 实现是 0.084 -> 0.078，几乎水平。原因不是训练不足，
是**结构性的**：skip=none 时整张 256x256 的输出只能由 64 个浮点数决定

    to_code   : 256*16*16 = 65536  ->  64
    from_code : 64  ->  65536

要"记住" 300 个观测点在哪、值多少，至少要 300*3 = 900 个数，64 维瓶颈
装不下，所以再多的观测点也进不去。论文原文（paper_text.txt 第 772 行）写
"Autoencoder consists of a symmetric encoder and decoder **and a skip
connection between them**"——有 skip 时空间特征绕过瓶颈，观测点才用得上。

本脚本只做推理，不做训练，用来：
  1. 验证 skip=none 的斜率确实被瓶颈锁死；
  2. 对比 skip=enc（编码器末层 -> 解码器入口，绕开 code）与
     skip=obs（输出残差加回观测图）两种口径的斜率；
  3. 顺带核一个信息论上界：每个模型实际用到了多少个观测点。

用法（在 src/ 下）：
    python ae_skip_sweep.py --tags bl_ae_s1=bl_ae_s1_skipenc \
        --kind ae --gpu 3 --out ../results/ae_skip_sweep.json

    也可以一次传多组，逗号分隔，名字里用 : 分隔 tag 与 skip 口径：
    python ae_skip_sweep.py \
        --tags "AE-none:bl_ae_s1:none,AE-enc:bl_ae_s1_skipenc:enc,..."
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (".", "..", os.path.join(_HERE, ".."), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import RadioMapSeer          # noqa: E402
from baselines import build_baseline       # noqa: E402

COUNTS = [50, 100, 150, 200, 250, 300]


@torch.no_grad()
def evaluate(m, loader, dev, limit=0):
    se = sq = 0.0
    n = 0
    for b in loader:
        obs = b["obs"].to(dev)
        env = b["env"].to(dev)
        t = b["target"].to(dev).float()
        y = m(obs, env).float().clamp(0, 1)
        se += float((y - t).pow(2).sum())
        sq += float(t.pow(2).sum())
        n += t.numel()
        if limit and n >= limit * 256 * 256:
            break
    return se / max(sq, 1e-12), (se / max(n, 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True,
                    help="名字:tag:skip 三元组，逗号分隔")
    ap.add_argument("--kind", default="ae", choices=["ae", "deepae"])
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--setup", default="s1")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="../results/ae_skip_sweep.json")
    ap.add_argument("--merge-fig7", default="",
                    help="把 name 以 -enc 结尾的曲线并入该 fig7 数据文件"
                         "（如 ../results/fig7_obs_sweep.json）")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specs = []
    for s in a.tags.split(","):
        s = s.strip()
        if not s:
            continue
        parts = s.split(":")
        specs.append((parts[0], parts[1], parts[2] if len(parts) > 2 else "none"))

    out = {"setup": a.setup, "kind": a.kind, "counts": COUNTS, "curves": {}}
    if os.path.exists(a.out):
        with open(a.out) as fh:
            out["curves"] = json.load(fh).get("curves", {})

    for name, tag, skip in specs:
        path = os.path.join("runs", tag, "best.pt")
        if not os.path.exists(path):
            print("[skip]", name, tag, "无权重", flush=True)
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = build_baseline(a.kind, ae_skip=skip).to(dev)
        m.load_state_dict(ck["model"])
        m.eval()
        nm, rm = [], []
        for k in COUNTS:
            ds = RadioMapSeer(a.cache, "test", a.setup, n_min=k, n_max=k,
                              seed=2000, cache_gain=True, augment=False,
                              obs_on_signal=False)
            ds.set_epoch(0)
            dl = DataLoader(ds, batch_size=a.bs, shuffle=False,
                            num_workers=a.workers, pin_memory=True)
            nmse, rmse = evaluate(m, dl, dev, a.limit)
            nm.append(nmse)
            rm.append(rmse)
            print("[%-10s] n=%3d NMSE=%.5f" % (name, k, nmse), flush=True)
        out["curves"][name] = {"nmse": nm, "rmse": rm, "tag": tag,
                               "skip": skip, "epoch": ck.get("epoch")}
        del m
        torch.cuda.empty_cache()

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("[saved]", a.out)

    # 可选：把结果并回 Fig.7 的主数据文件。
    # 旧曲线改名保存（AE_noskip），新曲线占用 AE/DeepAE 两个位置，
    # 这样 plot_fig7_ours.py 不用改就能直接画出修好的两条线。
    if a.merge_fig7:
        _merge_fig7(a, out)


def _merge_fig7(a, out):
    """把 --tags 里名字以 -enc 结尾的曲线并入 fig7_obs_sweep.json。"""
    path = a.merge_fig7
    if not os.path.exists(path):
        print("[merge] 找不到", path)
        return
    with open(path) as fh:
        base = json.load(fh)
    base.setdefault("curves", {})
    renamed = []
    for name, v in out["curves"].items():
        if not name.endswith("-enc"):
            continue
        tgt = name[:-4]                      # AE-enc -> AE
        old = base["curves"].get(tgt)
        if old is not None and (tgt + "_noskip") not in base["curves"]:
            base["curves"][tgt + "_noskip"] = old
            renamed.append(tgt + "_noskip")
        base["curves"][tgt] = {"nmse": v["nmse"], "rmse": v["rmse"]}
        print("[merge] %s <- %s (旧的存为 %s_noskip)" % (tgt, name, tgt))
    base["counts"] = out["counts"]
    with open(path, "w") as fh:
        json.dump(base, fh, indent=2)
    print("[merge] 写回", path, "renamed:", renamed)


if __name__ == "__main__":
    main()
