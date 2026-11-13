"""Is the 13x train/test MSE gap real overfitting or an eval-mode artefact?

Reports, for a trained checkpoint:
  * global NMSE / RMSE on train, val, test
  * train-set error measured in train mode (batch-norm batch stats) vs eval mode
  * error split into observed pixels vs unobserved pixels
  * two trivial baselines (predict 0 / predict mean of observations)
"""
import argparse, json, torch
import numpy as np
from torch.amp import autocast
from dataset import build_loaders
from models import MFFGCN, SingleEncoderUNet
from losses import MetricAccumulator
from train import fwd, move


@torch.no_grad()
def run(model, loader, device, arch, max_batches=None, train_mode=False,
        split_by_mask=False):
    model.train() if train_mode else model.eval()
    acc = MetricAccumulator()
    extra = {}
    if split_by_mask:
        se_obs = se_un = sq_obs = sq_un = 0.0
        n_obs = n_un = 0
        se_zero = 0.0
        se_obsm = 0.0
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        obs, env, mask, target = move(b, device)
        with autocast("cuda", enabled=True):
            coarse, fine, _ = fwd(model, obs, env, mask, arch)
        fine = fine.float().clamp(0, 1)
        target = target.float()
        acc.update(coarse, fine, target)
        if split_by_mask:
            m = mask > 0.5
            d2 = (fine - target) ** 2
            se_obs += d2[m].sum().item(); n_obs += m.sum().item()
            se_un += d2[~m].sum().item(); n_un += (~m).sum().item()
            sq_obs += target[m].pow(2).sum().item()
            sq_un += target[~m].pow(2).sum().item()
            se_zero += target.pow(2).sum().item()
            # baseline: fill the whole map with the mean of the observed values
            mm = obs.sum(dim=(1, 2, 3), keepdim=True) / mask.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1)
            se_obsm += (mm - target).pow(2).sum().item()
    out = acc.compute()
    if split_by_mask:
        out["rmse_obs"] = (se_obs / max(n_obs, 1)) ** 0.5
        out["rmse_unobs"] = (se_un / max(n_un, 1)) ** 0.5
        out["nmse_obs"] = se_obs / max(sq_obs, 1e-12)
        out["nmse_unobs"] = se_un / max(sq_un, 1e-12)
        out["pix_obs"] = n_obs / max(n_obs + n_un, 1)
        out["baseline_zero_nmse"] = 1.0
        out["baseline_obsm_nmse"] = se_obsm / max(sq_obs + sq_un, 1e-12)
        out["energy_obs"] = sq_obs / max(sq_obs + sq_un, 1e-12)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/calib_single")
    ap.add_argument("--arch", default="single", choices=["single", "mffgcn"])
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batches", type=int, default=80)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    tr, va, te = build_loaders(args.cache, "s1", batch_size=32, workers=8,
                               seed=0, augment=False, obs_on_signal=False,
                               split_mode="city")
    ck = torch.load(f"{args.run}/best.pt", map_location=device, weights_only=False)
    a = ck["args"]
    widths = tuple(int(x) for x in a["widths"].split(","))
    if args.arch == "single":
        model = SingleEncoderUNet(widths=widths, bottleneck=a["bottleneck"]).to(device)
    else:
        model = MFFGCN(widths=widths, bottleneck=a["bottleneck"],
                       afm_mode=a["afm_mode"]).to(device)
    model.load_state_dict(ck["model"])
    res = {"run": args.run, "best_epoch": ck["epoch"], "no_augment": a.get("no_augment")}
    res["train_eval"] = run(model, tr, device, args.arch, max_batches=args.batches)
    res["train_bn_mode"] = run(model, tr, device, args.arch, max_batches=20, train_mode=True)
    res["test"] = run(model, te, device, args.arch, split_by_mask=True)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
