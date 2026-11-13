"""探针：检查精修分支（GCN 残差）到底贡献了多少。

背景：txf_ab_channel_s2 的 test_metrics.json 里 nmse 与 nmse_coarse 到小数点后 6 位
完全相同（0.006270 / 0.006271），训练日志里 val_nmse 与 val_coarse_nmse 也一路相等。
这只有一种可能：fine ≈ coarse，即残差分支输出恒为 0，模型退化成"只做粗估计"。

本脚本直接量三件事，用 12 个 test batch、不落盘、不改任何 checkpoint：
    r_rel = mean|fine-coarse| / mean|target-coarse|   >1 说明精修在往反方向推
    r_abs = mean|fine-coarse|                         绝对幅度
    gain  = 1 - mse(fine) / mse(coarse)               精修的净收益（负=帮倒忙）

用法：
    python probe_residual.py --cache <cache> --setup s2 --gpu 1 \
        --ckpt runs/txf_ab_channel_s2/best.pt [--ckpt ...]
"""
import argparse
import os

import torch

from dataset import build_loaders
from evaluate import build_from_ckpt


@torch.no_grad()
def probe(ckpt, cache, setup, gpu, n_batches=12):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda")
    ck = torch.load(ckpt, map_location="cpu")
    model, arch = build_from_ckpt(ck, device)
    model.to(device).eval()

    _, _, te = build_loaders(cache, setup, batch_size=8, workers=2,
                             seed=0, augment=False, split_mode="city")

    se_fc = se_te = se_ce = 0.0   # |fine-coarse|, |target-coarse|, 计数
    sse_f = sse_c = 0.0          # 平方误差累加（算 gain）
    n = 0
    for i, b in enumerate(te):
        if i >= n_batches:
            break
        obs, env, mask, target = [b[k].to(device, non_blocking=True)
                                  for k in ("obs", "env", "mask", "target")]
        with torch.autocast("cuda", enabled=True):
            if arch == "single":
                coarse = model(torch.cat([obs, env], dim=1))
                fine = coarse
            else:
                coarse, fine, _ = model(obs, env, mask)
        coarse = coarse.float().clamp(0, 1)
        fine = fine.float().clamp(0, 1)
        target = target.float()
        se_fc += (fine - coarse).abs().sum().item()
        se_te += (target - coarse).abs().sum().item()
        se_ce += coarse.numel()
        sse_f += ((fine - target) ** 2).sum().item()
        sse_c += ((coarse - target) ** 2).sum().item()
        n += coarse.numel()

    return {
        "ckpt": ckpt,
        "setup": setup,
        "arch": arch,
        "r_abs": se_fc / max(n, 1),
        "r_rel": se_fc / max(se_te, 1e-9),
        "gain": 1.0 - sse_f / max(sse_c, 1e-12),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--setup", default="s2")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--n-batches", type=int, default=12)
    a = ap.parse_args()

    print(f"{'ckpt':46s} {'|fine-coarse|':>13s} {'相对粗估误差':>12s} {'精修净收益':>11s}")
    for c in a.ckpt:
        r = probe(c, a.cache, a.setup, a.gpu, a.n_batches)
        print(f"{c:46s} {r['r_abs']:13.6f} {r['r_rel']:12.3%} {r['gain']:11.2%}")


if __name__ == "__main__":
    main()
