"""RME-GAN 复现：先用路径损耗物理模型预训练，再用真实数据做条件 GAN 微调。

论文对该模型只给了两句描述：第一阶段基于 log-distance 路径损耗训练，
第二阶段学习真实传播。这里按这个描述实现：

阶段一（物理）
    从训练集统计"距离到发射机"与"归一化增益"的关系，拟合
        g(d) = a * d^(-n)
    然后让生成器去拟合这张由物理公式生成的伪地图。这一阶段完全不看真值，
    学到的是"离发射机越远信号越弱"这个先验。

阶段二（真实数据 GAN）
    Pix2pix 式条件 GAN：生成器 U-Net，判别器 PatchGAN，损失为
        对抗损失 + lambda_L1 * L1(生成图, 真值)
    与 Pix2pix+ 的区别只在有没有阶段一。

用法：
    python rme_gan_train.py --setup s1 --tag bl_rmegan_s1 --phase both
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from baselines import PatchGAN, Pix2pixGen
from dataset import build_loaders
from losses import MetricAccumulator
from models import count_params


def move(b, device):
    return (b["obs"].to(device, non_blocking=True),
            b["env"].to(device, non_blocking=True),
            b["mask"].to(device, non_blocking=True),
            b["target"].to(device, non_blocking=True))


@torch.no_grad()
def fit_pathloss(loader, n_batches=40, device="cpu"):
    """统计距离-增益关系，最小二乘拟合 g = a * d^(-n)。

    返回 (a, n)：把距离 d（像素）映射到归一化增益。
    """
    ds, gs = [], []
    for i, b in enumerate(loader):
        if i >= n_batches:
            break
        env = b["env"]
        tgt = b["target"]
        tx = env[:, 1].flatten(1).argmax(1)
        r = (tx // 256).float()
        c = (tx % 256).float()
        yy = torch.arange(256).view(1, 256, 1).expand_as(tgt[:, 0])
        xx = torch.arange(256).view(1, 1, 256).expand_as(tgt[:, 0])
        d = torch.sqrt((yy - r[:, None, None]) ** 2
                       + (xx - c[:, None, None]) ** 2) + 1.0
        m = tgt[:, 0] > 1e-3
        if m.sum() < 100:
            continue
        ds.append(d[m].flatten()[::37])
        gs.append(tgt[:, 0][m].flatten()[::37])
    d = torch.cat(ds).double()
    g = torch.cat(gs).double()
    # log g = log a - n log d
    A = torch.stack([torch.ones_like(d), -torch.log(d)], dim=1)
    coef, *_ = torch.linalg.lstsq(A, torch.log(g).unsqueeze(1))
    a = float(torch.exp(coef[0]))
    n = float(coef[1])
    return a, n


def physics_map(env, a, n):
    """按 g = a * d^(-n) 生成伪增益图，并归一化到 0..1。"""
    B = env.shape[0]
    tx = env[:, 1].flatten(1).argmax(1)
    r = (tx // 256).float()
    c = (tx % 256).float()
    yy = torch.arange(256, device=env.device).view(1, 256, 1).expand(B, 256, 256)
    xx = torch.arange(256, device=env.device).view(1, 1, 256).expand(B, 256, 256)
    d = torch.sqrt((yy - r[:, None, None]) ** 2
                   + (xx - c[:, None, None]) ** 2) + 1.0
    g = a * d.pow(-n)
    # 逐样本归一化到 [0,1]，避免绝对量纲影响训练
    g = g.reshape(B, -1)
    lo = g.min(1, keepdim=True).values
    hi = g.max(1, keepdim=True).values
    g = (g - lo) / (hi - lo + 1e-8)
    return g.view(B, 1, 256, 256).float()


@torch.no_grad()
def evaluate(model, loader, device, amp=True, max_batches=None):
    model.eval()
    acc = MetricAccumulator()
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        obs, env, mask, target = move(b, device)
        with autocast("cuda", enabled=amp):
            y = model(obs, env)
        y = y.float().clamp(0, 1)
        acc.update(y, y, target.float())
    model.train()
    return acc.compute()


def run_phase1(model, tr, va, device, args, run_dir, a, n):
    """物理模型预训练：只拟合伪标签，不用真值。"""
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    for ep in range(1, args.phase1_epochs + 1):
        model.train()
        t0 = time.time()
        run, nb = 0.0, 0
        for b in tr:
            obs, env, mask, target = move(b, device)
            pseudo = physics_map(env, a, n)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=not args.no_amp):
                y = model(obs, env)
                loss = F.l1_loss(y, pseudo)
            loss.backward()
            opt.step()
            run += float(loss)
            nb += 1
        vm = evaluate(model, va, device, amp=not args.no_amp,
                      max_batches=args.val_batches)
        print(f"[p1 ep {ep:03d}] l1={run/max(nb,1):.4f} "
              f"val RMSE={vm['rmse']:.5f} ({time.time()-t0:.0f}s)", flush=True)
        # 阶段一只看生成质量，不参与早停选择，保存最后一轮供阶段二初始化
    torch.save({"model": model.state_dict(), "epoch": args.phase1_epochs,
                "physics": {"a": a, "n": n}},
               os.path.join(run_dir, "phase1.pt"))
    return best


def run_phase2(model, disc, tr, va, device, args, run_dir):
    import torch.nn.functional as F
    opt_g = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        disc.train()
        if hasattr(tr.dataset, "set_epoch"):
            tr.dataset.set_epoch(ep)
        t0 = time.time()
        for i, b in enumerate(tr):
            obs, env, mask, target = move(b, device)
            cond = torch.cat([obs, env], dim=1)
            fake = model(obs, env)
            d_real = disc(cond, target)
            d_fake = disc(cond, fake.detach())
            loss_d = 0.5 * (bce(d_real, torch.ones_like(d_real))
                            + bce(d_fake, torch.zeros_like(d_fake)))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            fake = model(obs, env)
            d_out = disc(cond, fake)
            loss_g = bce(d_out, torch.ones_like(d_out)) \
                + args.lambda_l1 * l1(fake, target)
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            if (i + 1) % 100 == 0:
                print(f"  p2 ep{ep} it{i+1}/{len(tr)} g={loss_g.item():.4f} "
                      f"d={loss_d.item():.4f}", flush=True)
        vm = evaluate(model, va, device, amp=not args.no_amp,
                      max_batches=args.val_batches)
        print(f"[p2 ep {ep:03d}] val NMSE={vm['nmse']:.5f} "
              f"RMSE={vm['rmse']:.5f} ({time.time()-t0:.0f}s)", flush=True)
        if vm["nmse"] < best:
            best = vm["nmse"]
            torch.save({"model": model.state_dict(), "epoch": ep, "val": vm,
                        "args": vars(args)}, os.path.join(run_dir, "best.pt"))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--setup", default="s1")
    ap.add_argument("--tag", default="bl_rmegan_s1")
    ap.add_argument("--phase", choices=["1", "2", "both"], default="both")
    ap.add_argument("--phase1-epochs", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-batches", type=int, default=60)
    ap.add_argument("--lambda-l1", type=float, default=100.0)
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)

    run_dir = os.path.join("runs", a.tag)
    os.makedirs(run_dir, exist_ok=True)

    tr, va, te = build_loaders(a.cache, a.setup, batch_size=a.bs,
                               workers=a.workers, seed=a.seed,
                               augment=False, obs_on_signal=False)
    model = Pix2pixGen().to(device)
    disc = PatchGAN().to(device)
    print(f"[info] G {count_params(model)/1e6:.2f} M, "
          f"D {count_params(disc)/1e6:.2f} M", flush=True)

    if a.phase in ("1", "both"):
        print("[info] 阶段一：估计路径损耗参数并预训练", flush=True)
        pa, pn = fit_pathloss(tr)
        print(f"[info] 拟合结果 g = {pa:.4f} * d^(-{pn:.3f})", flush=True)
        run_phase1(model, tr, va, device, a, run_dir, pa, pn)

    if a.phase in ("2", "both"):
        p1 = os.path.join(run_dir, "phase1.pt")
        if os.path.exists(p1):
            sd = torch.load(p1, map_location=device, weights_only=False)
            model.load_state_dict(sd["model"])
            print("[info] 阶段二：加载阶段一权重后做 GAN 微调", flush=True)
        run_phase2(model, disc, tr, va, device, a, run_dir)

    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    tst = evaluate(model, te, device, amp=not a.no_amp)
    tst["best_epoch"] = ck["epoch"]
    tst["setup"] = a.setup
    print("[TEST]", json.dumps(tst, indent=2))
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as fh:
        json.dump(tst, fh, indent=2)


if __name__ == "__main__":
    main()
