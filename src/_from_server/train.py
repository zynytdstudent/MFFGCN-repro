"""Train MFFGCN on RadioMapSeer under Setup 1 / 2 / 3."""

import argparse
import json
import os
import time

import torch
from torch.amp import GradScaler, autocast

from dataset import build_loaders
from losses import MFFGCNLoss, MetricAccumulator
from models import MFFGCN, SingleEncoderUNet, count_params


def move(b, device):
    return (b["obs"].to(device, non_blocking=True),
            b["env"].to(device, non_blocking=True),
            b["mask"].to(device, non_blocking=True),
            b["target"].to(device, non_blocking=True))


def fwd(model, obs, env, mask, arch):
    """Uniform forward signature: returns (coarse, fine, residual)."""
    if arch == "single":
        y = model(torch.cat([obs, env], dim=1))
        return y, y, torch.zeros_like(y)
    return model(obs, env, mask)


@torch.no_grad()
def evaluate(model, loader, device, amp=True, max_batches=None, arch="mffgcn"):
    model.eval()
    acc = MetricAccumulator()
    acc_c = MetricAccumulator()
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        obs, env, mask, target = move(b, device)
        with autocast("cuda", enabled=amp):
            coarse, fine, _ = fwd(model, obs, env, mask, arch)
        coarse = coarse.float().clamp(0, 1)
        fine = fine.float().clamp(0, 1)
        target = target.float()
        acc.update(coarse, fine, target)
        acc_c.update(coarse, coarse, target)
        if max_batches and i + 1 >= max_batches:
            break
    out = acc.compute()
    c = acc_c.compute()
    out["coarse_nmse"] = c["nmse"]
    out["coarse_rmse"] = c["rmse"]
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--setup", default="s1",
                    choices=["s1", "s2", "s3", "s4", "s5"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--decay-epoch", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0,
                    help="if >0, train on the first N samples only")
    ap.add_argument("--val-batches", type=int, default=60,
                    help="batches used for the per-epoch validation check")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--l1", type=float, default=0.1)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--l3", type=float, default=1e-4)
    ap.add_argument("--widths", default="32,64,128,256",
                    help="encoder channel widths, comma separated")
    ap.add_argument("--bottleneck", type=int, default=512)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--afm-mode", default="softmax",
                    choices=["softmax", "sigmoid", "spatial", "concat", "add", "cross"],
                    help="fusion module: softmax/sigmoid/spatial gate, concat, add, cross-attention")
    ap.add_argument("--gcn-mode", default="dynamic",
                    choices=["dynamic", "static", "none"],
                    help="dynamic semantic graph, fixed grid graph, or no GCN")
    ap.add_argument("--no-residual", action="store_true",
                    help="ablation w/o Residual: graph branch outputs the map directly")
    ap.add_argument("--k", type=int, default=16,
                    help="neighbours per node in the refinement graph "
                         "(sensitivity sweep)")
    ap.add_argument("--arch", default="mffgcn", choices=["mffgcn", "single"])
    ap.add_argument("--split-mode", default="city", choices=["city", "pair"],
                    help="'pair' splits samples instead of maps (diagnostic)")
    ap.add_argument("--obs-on-signal", action="store_true",
                    help="sample sensors only where signal exists")
    ap.add_argument("--fixed-obs", action="store_true",
                    help="keep the same observation set every epoch")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    tag = args.tag or f"{args.setup}_seed{args.seed}"
    run_dir = os.path.join("runs", tag)
    os.makedirs(run_dir, exist_ok=True)

    tr, va, te = build_loaders(args.cache, args.setup, batch_size=args.bs,
                               workers=args.workers, seed=args.seed,
                               train_subset=(args.smoke or None),
                               augment=not args.no_augment,
                               obs_on_signal=args.obs_on_signal,
                               split_mode=args.split_mode)
    print(f"[info] train={len(tr.dataset)} val={len(va.dataset)} "
          f"test={len(te.dataset)} batches/epoch={len(tr)}")

    widths = tuple(int(x) for x in args.widths.split(","))
    if args.arch == "single":
        model = SingleEncoderUNet(widths=widths,
                                  bottleneck=args.bottleneck).to(device)
    else:
        model = MFFGCN(widths=widths, bottleneck=args.bottleneck,
                       afm_mode=args.afm_mode, gcn_mode=args.gcn_mode,
                       k=args.k, residual=not args.no_residual).to(device)
    print(f"[info] widths={widths} bottleneck={args.bottleneck} "
          f"afm={args.afm_mode} gcn={args.gcn_mode} k={args.k} "
          f"residual={not args.no_residual} "
          f"params={count_params(model)/1e6:.2f} M "
          f"aug={not args.no_augment}")

    crit = MFFGCNLoss(args.l1, args.l2, args.l3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[args.decay_epoch], gamma=0.1)
    scaler = GradScaler("cuda", enabled=not args.no_amp)

    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w") as fh:
        fh.write("epoch,lr,train_loss,mse_coarse,mse_fine,val_nmse,val_rmse,"
                 "val_coarse_nmse,val_coarse_rmse,sec\n")

    best = float("inf")
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        if hasattr(tr.dataset, "set_epoch"):
            # re-drawing the observation set each epoch is a form of
            # augmentation; --fixed-obs disables it for a stable comparison
            tr.dataset.set_epoch(0 if args.fixed_obs else ep)
        if hasattr(va.dataset, "set_epoch"):
            va.dataset.set_epoch(0)
        t0 = time.time()
        run_loss = run_c = run_f = 0.0
        nb = 0
        for b in tr:
            obs, env, mask, target = move(b, device)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=not args.no_amp):
                coarse, fine, _ = fwd(model, obs, env, mask, args.arch)
                loss, parts = crit(coarse, fine, target)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()

            run_loss += parts["loss"].item()
            run_c += parts["mse_coarse"].item()
            run_f += parts["mse_fine"].item()
            nb += 1
            if nb % 100 == 0:
                print(f"  ep{ep} it{nb}/{len(tr)} loss={run_loss/nb:.5f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

        lr_now = opt.param_groups[0]["lr"]
        sched.step()

        vm = evaluate(model, va, device, amp=not args.no_amp,
                      max_batches=40 if args.smoke else args.val_batches,
                      arch=args.arch)
        dt = time.time() - t0
        print(f"[ep {ep:03d}] loss={run_loss/max(nb,1):.5f} "
              f"coarse={run_c/max(nb,1):.5f} fine={run_f/max(nb,1):.5f} | "
              f"val NMSE={vm['nmse']:.5f} RMSE={vm['rmse']:.5f} "
              f"({dt:.0f}s)", flush=True)
        with open(log_path, "a") as fh:
            fh.write(f"{ep},{lr_now:.6g},{run_loss/max(nb,1):.6f},"
                     f"{run_c/max(nb,1):.6f},{run_f/max(nb,1):.6f},"
                     f"{vm['nmse']:.6f},{vm['rmse']:.6f},"
                     f"{vm['coarse_nmse']:.6f},{vm['coarse_rmse']:.6f},{dt:.1f}\n")

        if vm["nmse"] < best:
            best = vm["nmse"]
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val": vm, "args": vars(args)},
                       os.path.join(run_dir, "best.pt"))

    torch.save({"model": model.state_dict(), "epoch": args.epochs,
                "args": vars(args)}, os.path.join(run_dir, "last.pt"))

    # ---- final test on the held-out cities ------------------------------
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    tst = evaluate(model, te, device, amp=not args.no_amp,
                   max_batches=20 if args.smoke else None, arch=args.arch)
    tst["best_epoch"] = ck["epoch"]
    tst["total_minutes"] = (time.time() - t_start) / 60
    tst["setup"] = args.setup
    print("[TEST]", json.dumps(tst, indent=2))
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as fh:
        json.dump(tst, fh, indent=2)
    print(f"[done] run_dir={run_dir}")


if __name__ == "__main__":
    main()
