"""训练论文表 III 的深度基线（AE / DeepAE / RadioUNet / Pix2pix+ / RME-GAN）。

与 train.py 分开，避免动到正在跑的 MFFGCN 训练。数据管线、损失口径、
评估协议全部复用同一套，保证和 MFFGCN 的数字可比。

    python baseline_train.py --arch ae --setup s1 --tag bl_ae_s1
    python baseline_train.py --arch radiounet --setup s1 --tag bl_wunet_s1
    python baseline_train.py --arch rme_gan --setup s1 --tag bl_rmegan_s1

RadioUNet 与 RME-GAN 都是两阶段：第一阶段跑完自动接着跑第二阶段。
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from baselines import PatchGAN, build_baseline
from dataset import build_loaders
from losses import MetricAccumulator
from models import count_params


def move(b, device):
    return (b["obs"].to(device, non_blocking=True),
            b["env"].to(device, non_blocking=True),
            b["mask"].to(device, non_blocking=True),
            b["target"].to(device, non_blocking=True))


def forward(model, obs, env, arch):
    """统一成 (coarse, fine) 两个输出。"""
    out = model(obs, env)
    if arch == "radiounet":
        return out                       # (hint, y)
    return out, out


@torch.no_grad()
def evaluate(model, loader, device, arch, amp=True, max_batches=None):
    model.eval()
    acc = MetricAccumulator()
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        obs, env, mask, target = move(b, device)
        with autocast("cuda", enabled=amp):
            _, y = forward(model, obs, env, arch)
        acc.update(y.float().clamp(0, 1), y.float().clamp(0, 1),
                   target.float())
    model.train()
    return acc.compute()


def train_stage(model, tr, va, device, args, tag, stage=1, epochs=None):
    epochs = epochs or args.epochs
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[args.decay_epoch], gamma=0.1)
    scaler = GradScaler("cuda", enabled=not args.no_amp)
    mse = nn.MSELoss()

    run_dir = os.path.join("runs", tag)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"train_log_stage{stage}.csv")
    with open(log_path, "w") as fh:
        fh.write("epoch,lr,train_loss,val_nmse,val_rmse,sec\n")

    best = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        if hasattr(tr.dataset, "set_epoch"):
            tr.dataset.set_epoch(ep)
        t0 = time.time()
        run, nb = 0.0, 0
        for b in tr:
            obs, env, mask, target = move(b, device)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=not args.no_amp):
                _, y = forward(model, obs, env, args.arch)
                loss = mse(y, target)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            run += float(loss)
            nb += 1
            if nb % 100 == 0:
                print(f"  s{stage} ep{ep} it{nb}/{len(tr)} "
                      f"loss={run/nb:.5f} ({time.time()-t0:.0f}s)", flush=True)

        vm = evaluate(model, va, device, args.arch, amp=not args.no_amp,
                      max_batches=args.val_batches)
        sched.step()
        dt = time.time() - t0
        print(f"[s{stage} ep {ep:03d}] loss={run/max(nb,1):.5f} "
              f"val NMSE={vm['nmse']:.5f} RMSE={vm['rmse']:.5f} "
              f"({dt:.0f}s)", flush=True)
        with open(log_path, "a") as fh:
            fh.write(f"{ep},{opt.param_groups[0]['lr']:.6g},"
                     f"{run/max(nb,1):.6f},{vm['nmse']:.6f},{vm['rmse']:.6f},"
                     f"{dt:.1f}\n")
        if vm["nmse"] < best:
            best = vm["nmse"]
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "stage": stage, "val": vm, "args": vars(args)},
                       os.path.join(run_dir, "best.pt"))
    torch.save({"model": model.state_dict(), "epoch": epochs, "stage": stage,
                "args": vars(args)}, os.path.join(run_dir, "last.pt"))
    return best


def train_gan(model, disc, tr, va, device, args, tag):
    """Pix2pix 式条件 GAN：L1 重建 + 对抗损失。

    ⚠ 2026-11-09 修 bug：原来这里用 args.lr（和 U-Net 共用一个默认 1e-3），
    而且**完全没有学习率衰减**（--decay-epoch 被忽略）。GAN 的判别器和生成器
    同时用 Adam(betas=0.5) 在 1e-3 下训练会持续震荡：bl_pix2pix_s1 的验证
    NMSE 从 ep34 的 0.0182 一路抖到 ep49 的 0.0292，50 个 epoch 都没收敛，
    最终 0.0111，比同结构的监督 U-Net（0.0058）还差一倍。
    现在拆出独立的 --gan-lr（默认 2e-4，cGAN 的通行设置）并补上 MultiStepLR。
    """
    import torch.nn.functional as F
    lr = args.gan_lr if args.gan_lr > 0 else args.lr
    # TTUR：D 的学习率可以单独压低（--d-lr-ratio）。判别器过强时 BCE_adv 会
    # 掉到 0.45 以下（< ln2），对抗梯度开始与 L1 目标冲突，生成器退化。
    d_lr = lr * getattr(args, "d_lr_ratio", 1.0)
    opt_g = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=d_lr, betas=(0.5, 0.999))
    ls = getattr(args, "label_smooth", 1.0)
    sched_g = torch.optim.lr_scheduler.MultiStepLR(
        opt_g, milestones=[args.decay_epoch], gamma=0.1)
    sched_d = torch.optim.lr_scheduler.MultiStepLR(
        opt_d, milestones=[args.decay_epoch], gamma=0.1)
    scaler_g = GradScaler("cuda", enabled=not args.no_amp)
    scaler_d = GradScaler("cuda", enabled=not args.no_amp)
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    run_dir = os.path.join("runs", tag)
    os.makedirs(run_dir, exist_ok=True)
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
            # ---- discriminator ----
            with autocast("cuda", enabled=not args.no_amp):
                fake = model(obs, env)
                d_real = disc(cond, target)
                d_fake = disc(cond, fake.detach())
                loss_d = 0.5 * (bce(d_real, torch.full_like(d_real, ls))
                                + bce(d_fake, torch.zeros_like(d_fake)))
            opt_d.zero_grad(set_to_none=True)
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()
            # ---- generator ----
            with autocast("cuda", enabled=not args.no_amp):
                fake = model(obs, env)
                d_fake = disc(cond, fake)
                loss_g = bce(d_fake, torch.full_like(d_fake, ls)) \
                    + args.lambda_l1 * l1(fake, target)
            opt_g.zero_grad(set_to_none=True)
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()
            if (i + 1) % 100 == 0:
                print(f"  ep{ep} it{i+1}/{len(tr)} g={loss_g.item():.4f} "
                      f"d={loss_d.item():.4f} ({time.time()-t0:.0f}s)",
                      flush=True)
        vm = evaluate(model, va, device, args.arch, amp=not args.no_amp,
                      max_batches=args.val_batches)
        sched_g.step()
        sched_d.step()
        print(f"[ep {ep:03d}] val NMSE={vm['nmse']:.5f} RMSE={vm['rmse']:.5f} "
              f"lr={opt_g.param_groups[0]['lr']:.2g} "
              f"dlr={opt_d.param_groups[0]['lr']:.2g} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vm["nmse"] < best:
            best = vm["nmse"]
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val": vm, "args": vars(args)},
                       os.path.join(run_dir, "best.pt"))
    torch.save({"model": model.state_dict(), "epoch": args.epochs,
                "args": vars(args)}, os.path.join(run_dir, "last.pt"))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    choices=["ae", "deepae", "radiounet", "pix2pix", "unet",
                             "rme_gan"])
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--setup", default="s1")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--decay-epoch", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-batches", type=int, default=60)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--lambda-l1", type=float, default=100.0)
    ap.add_argument("--ae-skip", default="none", choices=["none", "enc", "obs"],
                    help="AE/DeepAE 的 skip 口径（论文写了有 skip，我们原来没有）")
    ap.add_argument("--ae-ndown", type=int, default=0,
                    help="覆盖 AEBase 的下采样次数（0=用默认；deepae 默认 5 -> skip 在 8x8）")
    ap.add_argument("--gan-lr", type=float, default=2e-4,
                    help="cGAN 的 Adam 学习率；<=0 时退回 --lr。"
                         "不要用 U-Net 的 1e-3，GAN 会震荡不收敛")
    ap.add_argument("--d-lr-ratio", type=float, default=1.0,
                    help="判别器学习率 = GAN 学习率 x 该值（TTUR）。"
                         "D 过强时设 0.5 或 0.25")
    ap.add_argument("--label-smooth", type=float, default=1.0,
                    help="真样本标签值，0.9 为常用标签平滑；1.0 表示不平滑")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--stages", default="1,2",
                    help="for two-stage models, which stages to run")
    ap.add_argument("--obs-on-signal", action="store_true")
    ap.add_argument("--subset-cities", type=int, default=0,
                    help="只用前 N 张地图训练（论文 Fig.7 的城市数扫描）；"
                         "0 表示用全部。注意是按城市数取子集，不是按样本数")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    tag = args.tag or f"bl_{args.arch}_{args.setup}"
    tr, va, te = build_loaders(args.cache, args.setup, batch_size=args.bs,
                               workers=args.workers, seed=args.seed,
                               train_subset=(args.smoke or None),
                               augment=not args.no_augment,
                               obs_on_signal=args.obs_on_signal,
                               subset_cities=(args.subset_cities or None))
    print(f"[info] train={len(tr.dataset)} val={len(va.dataset)} "
          f"test={len(te.dataset)}", flush=True)

    t_start = time.time()
    if args.arch == "rme_gan":
        # 2026-11-12：这条分支以前会默默 build 一个 pix2pix 生成器，训出来的
        # 东西**不是 RME-GAN**（没有第一阶段的路损预训练），却挂着 bl_rmegan_* 的
        # 名字，很容易被当成真基线写进表里。真正的 RME-GAN 在 rme_gan_train.py。
        # 全量扫过所有 best.pt，没有一个是走这里训出来的，所以直接改成报错。
        raise SystemExit(
            "--arch rme_gan 已废弃：这里只会训出第二个 Pix2pix+。\n"
            "真正的 RME-GAN（先路损预训练、再 GAN 微调）请用：\n"
            "    python rme_gan_train.py --phase both --tag bl_rmegan_%s"
            % args.setup)
    if args.arch == "pix2pix":
        model = build_baseline("pix2pix").to(device)
        disc = PatchGAN().to(device)
        print(f"[info] generator {count_params(model)/1e6:.2f} M, "
              f"disc {count_params(disc)/1e6:.2f} M", flush=True)
        train_gan(model, disc, tr, va, device, args, tag)
    elif args.arch == "radiounet":
        model = build_baseline("radiounet").to(device)
        print(f"[info] RadioUNet {count_params(model)/1e6:.2f} M", flush=True)
        stages = [int(s) for s in args.stages.split(",")]
        for st in stages:
            model.stage = st
            if st == 2:
                # 冻结第一级，只训第二级
                for p in model.net1.parameters():
                    p.requires_grad = False
                ck = os.path.join("runs", tag, "best.pt")
                if os.path.exists(ck):
                    sd = torch.load(ck, map_location=device,
                                    weights_only=False)["model"]
                    model.load_state_dict(
                        {k: v for k, v in sd.items()}, strict=False)
                else:
                    print("[warn] stage1 checkpoint 缺失，第二级将用未训练的"
                          "第一级输出", flush=True)
            print(f"[info] === stage {st} ===", flush=True)
            train_stage(model, tr, va, device, args, tag, stage=st)
            # stage 1 结束后把最好权重装回去
            ck = os.path.join("runs", tag, "best.pt")
            if st == 1 and os.path.exists(ck):
                model.load_state_dict(
                    torch.load(ck, map_location=device,
                               weights_only=False)["model"])
    else:
        model = build_baseline(args.arch, ae_skip=args.ae_skip,
                                 ae_ndown=args.ae_ndown).to(device)
        print(f"[info] {args.arch} {count_params(model)/1e6:.2f} M", flush=True)
        train_stage(model, tr, va, device, args, tag)

    # ---- final test ----
    ck_path = os.path.join("runs", tag, "best.pt")
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    if hasattr(model, "stage"):
        model.stage = 2
    tst = evaluate(model, te, device, args.arch, amp=not args.no_amp,
                   max_batches=5 if args.smoke else None)
    tst["best_epoch"] = ck["epoch"]
    tst["setup"] = args.setup
    tst["total_minutes"] = (time.time() - t_start) / 60
    print("[TEST]", json.dumps(tst, indent=2))
    with open(os.path.join("runs", tag, "test_metrics.json"), "w") as fh:
        json.dump(tst, fh, indent=2)


if __name__ == "__main__":
    main()
