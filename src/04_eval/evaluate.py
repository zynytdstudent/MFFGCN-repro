"""
完整评估：分区指标 + 定性可视化
================================================================
train.py 只在训练结束时算一个"全图 NMSE"。这个脚本是训练结束后的正规评估，
比 train.py 多做三件事：

    1) 分区统计 —— 把误差拆成三个区域分别看：
         all      全图
         constr   盲区（Setup 2/3 里被挖掉、不允许布点的那块）
         obs      已观测区（有传感器的像素）
       这个拆分是论文的核心分析手段。只看全图 NMSE 会掩盖一个关键问题：
       模型是不是"只会抄观测点、盲区全靠猜"？如果 obs 区误差很小但 constr 区
       误差很大，说明它没学到传播规律，只是在做插值/复制。

    2) 同时报告 coarse 和 fine —— 两者之差就是 GCN 精修段的净收益。

    3) 存定性图 —— 每张图一行 5 联：真值 / 稀疏观测 / 粗估计 / 精修结果 / 误差图
       写报告和 PPT 全靠这些图。

--------------------------------------------------------------------------------
为什么逐样本（batch=1）而不是批量评估？
--------------------------------------------------------------------------------
    因为每个样本的盲区掩码 allowed 都不一样，要在各自的区域内分别累加误差，
    批量化只会让代码更绕。8080 个样本跑一遍在单卡上也就几分钟，
    而且这个脚本只要在训练结束后跑一次，不需要频繁执行。
    （追求速度的场景用 04_eval/eval_all.py，那里是批量化的。）

输出：<out_dir>/metrics_<name>.json  +  <out_dir>/sample_<setup>_<j>.png
"""

import argparse
import json
import os

import numpy as np
import torch

from dataset import RadioMapSeer
from models import MFFGCN, SingleEncoderUNet


def build_from_ckpt(ck, device):
    """根据 checkpoint 里保存的 args 重建完全一致的网络结构。

    这是整个项目里最容易踩坑的地方之一：评估时必须用"产生这个 checkpoint 时
    的那套超参"来建模型，否则 state_dict 的键对不上、或者层宽度不对，
    load_state_dict 会直接报错（或者更糟：用 strict=False 静默错位）。

    所以 train.py 里 torch.save 时特意把 vars(args) 一起存了进去，
    这里就能原样还原：arch / widths / bottleneck / afm_mode / gcn_mode / k /
    residual。任何一个漏掉都会导致结构不匹配。

    返回 (model, arch)：arch 用来决定前向是走 3 通道还是双分支。
    """
    a = ck.get("args") or {}

    arch = a.get("arch", "mffgcn")
    widths = tuple(int(x) for x in str(a.get("widths", "32,64,128,256")).split(","))
    bottleneck = int(a.get("bottleneck", 512))

    if arch == "single":
        model = SingleEncoderUNet(widths=widths, bottleneck=bottleneck)
    else:
        model = MFFGCN(widths=widths, bottleneck=bottleneck,
                       afm_mode=a.get("afm_mode", "softmax"),
                       gcn_mode=a.get("gcn_mode", "dynamic"),
                       k=int(a.get("k", 16)),
                       # 注意这里是 not a.get("no_residual")：
                       # checkpoint 里存的是命令行参数名（no_residual），
                       # 模型要的是正向开关（residual），取反
                       residual=not a.get("no_residual", False))
    model.load_state_dict(ck["model"])
    return model.to(device), arch


def masked_stats(pred, target, region, eps=1e-12):
    """在指定区域内算 (NMSE, RMSE)。

    region 是 bool 图 (1,H,W)，先把 pred 和 target 都在该区域外清零，
    再算误差。因为清零后区域外的贡献为 0，直接全图求和等价于只在区域内求和。

    这是工具函数，主流程里用的是下面的累加写法（更快），保留它便于交互式分析。
    """
    p = pred * region
    t = target * region
    se = float(((p - t) ** 2).sum())
    sq = float((t ** 2).sum())
    # 区域内的元素个数 = 区域内像素数 × 样本数
    n = float(region.sum()) * target.shape[0]
    return se / max(sq, eps), (se / max(n, 1)) ** 0.5


@torch.no_grad()
def run(ckpt, cache, setup, split="test", out_dir="eval_out", gpu="0",
        max_samples=None, save_figs=8, obs_on_signal=False, name=None,
        obs_snr=None, obs_scale=1.0):
    """主评估流程。

    参数说明里最需要注意的两个：
      obs_on_signal : 必须和训练时保持一致！它决定"传感器布在有信号处还是全图"，
                      两边不一致的话，评的就不是同一个任务。
      obs_snr / obs_scale : 鲁棒性扫描用（加噪声 / 改观测预算），
                      默认 None / 1.0 表示"干净的原设定"。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model, arch = build_from_ckpt(ck, device)
    model.eval()

    # seed 固定成 2024 而不是沿用训练时的 seed：评估要用"独立的一套观测点抽取"，
    # 避免评估集恰好和训练时见过的那套观测位置重合
    ds = RadioMapSeer(cache, split, setup, seed=2024,
                      obs_on_signal=obs_on_signal,
                      obs_snr=obs_snr, obs_scale=obs_scale)
    n = len(ds) if not max_samples else min(max_samples, len(ds))
    print(f"[info] evaluating {n} samples from split={split} setup={setup}")

    os.makedirs(out_dir, exist_ok=True)

    # 累加器结构：{区域名: [平方误差和, 真值平方和]}
    # 先用 "all" 占位，constr / obs 遇到时才动态创建
    acc = {"all": [0.0, 0.0]}       # fine 的累加
    acc_c = {"all": [0.0, 0.0]}     # coarse 的累加
    n_img = 0
    figs = []                       # 缓存前 save_figs 张图，最后统一画

    for i in range(n):
        b = ds[i]
        # ds[i] 返回的已经是 (1,H,W)，这里再 [None] 补一个 batch 维 → (1,1,H,W)
        obs = b["obs"][None].to(device)
        env = b["env"][None].to(device)
        mask = b["mask"][None].to(device)
        target = b["target"][None].to(device)
        allowed = b["allowed"][None].to(device)     # 1=可观测区域

        # ---- 前向 ----
        if arch == "single":
            # 单编码器：把 obs(1) + env(2) 拼成 3 通道
            y = model(torch.cat([obs, env], dim=1)).clamp(0, 1)
            coarse, fine = y, y                     # 没有两段结构，两者相同
        else:
            coarse, fine, residual = model(obs, env, mask)
            # 输出必须夹到物理合法范围 [0,1]
            coarse = coarse.clamp(0, 1)
            fine = fine.clamp(0, 1)

        # ---- 定义三个统计区域 ----
        regions = {
            "all": torch.ones_like(mask),      # 全图
            "constr": 1.0 - allowed,           # 盲区 = 不允许布点的地方
            "obs": mask,                       # 已观测像素
        }
        for k, reg in regions.items():
            if reg.sum() == 0:
                # Setup 1 没有盲区，1-allowed 全是 0，直接跳过避免 0/0
                continue
            if k not in acc:
                # 第一次遇到该区域时才创建累加槽
                acc[k] = [0.0, 0.0]
                acc_c[k] = [0.0, 0.0]
            # 同一套区域分别喂给 fine 和 coarse 两个累加器
            for store, pred in ((acc, fine), (acc_c, coarse)):
                p = pred * reg                 # 区域外清零
                t = target * reg
                store[k][0] += float(((p - t) ** 2).sum())
                store[k][1] += float((t ** 2).sum())

        n_img += 1

        # ---- 缓存定性图（只留前 save_figs 张）----
        if len(figs) < save_figs:
            figs.append((
                target[0, 0].cpu().numpy(),          # 真值
                obs[0, 0].cpu().numpy(),             # 稀疏观测
                coarse[0, 0].cpu().numpy(),          # U-Net 粗估计
                fine[0, 0].cpu().numpy(),            # 最终结果
                (fine - target)[0, 0].cpu().numpy(), # 误差图
                allowed[0, 0].cpu().numpy(),         # 用来画盲区轮廓
            ))
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{n}", flush=True)

    def pack(store):
        """把累加结果换算成 NMSE / RMSE。

        ⚠ 口径说明（看数字时要注意）：
          nmse  = 区域内的 se 之和 / 区域内的真值平方和   → 严格正确，可跨区域比较
          rmse  = sqrt(区域内 se 之和 / (样本数 × 256 × 256))
                  分母用的是"全图像素数"，而不是"该区域的像素数"。
                  所以 constr_rmse 和 obs_rmse 的绝对值会偏小（被区域外那部分
                  分母稀释了），只在同区域之间比较才有效。
                  想跨区域比大小，请统一看 nmse。
        """
        out = {}
        for k, (se, sq) in store.items():
            out[f"{k}_nmse"] = se / max(sq, 1e-12)
            out[f"{k}_rmse"] = (se / max(n_img * 256 * 256, 1)) ** 0.5
        return out

    res = {"n_samples": n_img, "setup": setup, "ckpt": ckpt,
           "name": name or setup,
           "best_epoch": ck.get("epoch"),           # 用的是第几轮的权重
           "fine": pack(acc), "coarse": pack(acc_c)}
    print(json.dumps(res, indent=2))
    with open(os.path.join(out_dir, f"metrics_{res['name']}.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    # ==================================================================
    # 定性图：5 联图
    # ==================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")           # 无显示环境（服务器/集群）必须用 Agg
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(0)  # 本处未使用，保留以便将来随机抽样
        for j, (t, o, c, f, e, a) in enumerate(figs):
            fig, ax = plt.subplots(1, 5, figsize=(20, 4.2))
            for k, (img, title, cmap, vmin, vmax) in enumerate([
                (t, "Ground truth", "viridis", 0, 1),
                (o, "Sparse observations", "viridis", 0, 1),
                (c, "Coarse estimate", "viridis", 0, 1),
                (f, "MFFGCN refined", "viridis", 0, 1),
                # 误差图用发散色标，以 0 为中心、范围 ±0.3，
                # 这样红/蓝能直接表达"高估/低估"，不会被整体偏置带偏
                (e, "Error (refined - GT)", "RdBu_r", -0.3, 0.3),
            ]):
                im = ax[k].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                ax[k].set_title(title, fontsize=11)
                ax[k].axis("off")
                if k == 4:
                    # 在误差图上描出盲区边界，方便看"盲区内的误差是不是更大"
                    yy, xx = np.nonzero(a < 0.5)
                    if len(yy):
                        ax[k].contour(a, levels=[0.5], colors="k", linewidths=1.2)
                plt.colorbar(im, ax=ax[k], fraction=0.046)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"sample_{setup}_{j}.png"), dpi=120)
            plt.close(fig)              # 必须关掉，否则画几十张图会耗尽文件句柄
        print(f"[ok] wrote {len(figs)} figures to {out_dir}")
    except Exception as exc:            # matplotlib 是可选的，缺了不影响指标
        print(f"[warn] figures skipped: {exc}")

    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="runs/<tag>/best.pt")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--setup", default="s1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="只评估前 N 条（调试用）")
    ap.add_argument("--obs-on-signal", action="store_true",
                    help="必须与训练时一致，否则评的不是同一个任务")
    ap.add_argument("--name", default=None,
                    help="输出文件名主干，如 s1 / single_s1；"
                         "下游 compare_all.py 按这个名字找文件")
    ap.add_argument("--save-figs", type=int, default=8,
                    help="输出几张定性图，0 表示不画")
    ap.add_argument("--obs-snr", type=float, default=None,
                    help="鲁棒性扫描：按该 SNR(dB) 给观测加噪声，如 20")
    ap.add_argument("--obs-scale", type=float, default=1.0,
                    help="鲁棒性扫描：观测预算倍率")
    a = ap.parse_args()
    run(a.ckpt, a.cache, a.setup, a.split, a.out, a.gpu, a.max_samples,
        obs_on_signal=a.obs_on_signal, name=a.name, save_figs=a.save_figs,
        obs_snr=a.obs_snr, obs_scale=a.obs_scale)
