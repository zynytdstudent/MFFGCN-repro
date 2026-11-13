"""Full evaluation: global / observation-constrained / observed-region metrics
plus qualitative visualisations."""

import argparse
import json
import os

import numpy as np
import torch

from dataset import RadioMapSeer
from models import MFFGCN, SingleEncoderUNet


def build_from_ckpt(ck, device):
    """Rebuild the exact architecture that produced this checkpoint."""
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
                       residual=not a.get("no_residual", False))
    model.load_state_dict(ck["model"])
    return model.to(device), arch


def masked_stats(pred, target, region, eps=1e-12):
    """region: bool array (1,H,W). Returns (nmse, rmse)."""
    p = pred * region
    t = target * region
    se = float(((p - t) ** 2).sum())
    sq = float((t ** 2).sum())
    n = float(region.sum()) * target.shape[0]
    return se / max(sq, eps), (se / max(n, 1)) ** 0.5


@torch.no_grad()
def run(ckpt, cache, setup, split="test", out_dir="eval_out", gpu="0",
        max_samples=None, save_figs=8, obs_on_signal=False, name=None,
        obs_snr=None, obs_scale=1.0):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model, arch = build_from_ckpt(ck, device)
    model.eval()

    ds = RadioMapSeer(cache, split, setup, seed=2024,
                      obs_on_signal=obs_on_signal,
                      obs_snr=obs_snr, obs_scale=obs_scale)
    n = len(ds) if not max_samples else min(max_samples, len(ds))
    print(f"[info] evaluating {n} samples from split={split} setup={setup}")

    os.makedirs(out_dir, exist_ok=True)
    acc = {"all": [0.0, 0.0]}
    acc_c = {"all": [0.0, 0.0]}
    n_img = 0
    figs = []

    for i in range(n):
        b = ds[i]
        obs = b["obs"][None].to(device)
        env = b["env"][None].to(device)
        mask = b["mask"][None].to(device)
        target = b["target"][None].to(device)
        allowed = b["allowed"][None].to(device)

        if arch == "single":
            y = model(torch.cat([obs, env], dim=1)).clamp(0, 1)
            coarse, fine = y, y
        else:
            coarse, fine, residual = model(obs, env, mask)
            coarse = coarse.clamp(0, 1)
            fine = fine.clamp(0, 1)
        regions = {
            "all": torch.ones_like(mask),
            "constr": 1.0 - allowed,
            "obs": mask,
        }
        for k, reg in regions.items():
            if reg.sum() == 0:
                continue                    # e.g. no constrained region in Setup 1
            if k not in acc:
                acc[k] = [0.0, 0.0]
                acc_c[k] = [0.0, 0.0]
            for store, pred in ((acc, fine), (acc_c, coarse)):
                p = pred * reg
                t = target * reg
                store[k][0] += float(((p - t) ** 2).sum())
                store[k][1] += float((t ** 2).sum())
        n_img += 1

        if len(figs) < save_figs:
            figs.append((
                target[0, 0].cpu().numpy(),
                obs[0, 0].cpu().numpy(),
                coarse[0, 0].cpu().numpy(),
                fine[0, 0].cpu().numpy(),
                (fine - target)[0, 0].cpu().numpy(),
                allowed[0, 0].cpu().numpy(),
            ))
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{n}", flush=True)

    def pack(store):
        out = {}
        for k, (se, sq) in store.items():
            out[f"{k}_nmse"] = se / max(sq, 1e-12)
            out[f"{k}_rmse"] = (se / max(n_img * 256 * 256, 1)) ** 0.5
        return out

    res = {"n_samples": n_img, "setup": setup, "ckpt": ckpt,
           "name": name or setup,
           "best_epoch": ck.get("epoch"), "fine": pack(acc), "coarse": pack(acc_c)}
    print(json.dumps(res, indent=2))
    with open(os.path.join(out_dir, f"metrics_{res['name']}.json"), "w") as fh:
        json.dump(res, fh, indent=2)

    # ---- qualitative figures -------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(0)
        for j, (t, o, c, f, e, a) in enumerate(figs):
            fig, ax = plt.subplots(1, 5, figsize=(20, 4.2))
            for k, (img, title, cmap, vmin, vmax) in enumerate([
                (t, "Ground truth", "viridis", 0, 1),
                (o, "Sparse observations", "viridis", 0, 1),
                (c, "Coarse estimate", "viridis", 0, 1),
                (f, "MFFGCN refined", "viridis", 0, 1),
                (e, "Error (refined - GT)", "RdBu_r", -0.3, 0.3),
            ]):
                im = ax[k].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                ax[k].set_title(title, fontsize=11)
                ax[k].axis("off")
                if k == 4:
                    # outline the constrained region
                    yy, xx = np.nonzero(a < 0.5)
                    if len(yy):
                        ax[k].contour(a, levels=[0.5], colors="k", linewidths=1.2)
                plt.colorbar(im, ax=ax[k], fraction=0.046)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"sample_{setup}_{j}.png"), dpi=120)
            plt.close(fig)
        print(f"[ok] wrote {len(figs)} figures to {out_dir}")
    except Exception as exc:      # matplotlib optional
        print(f"[warn] figures skipped: {exc}")

    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--setup", default="s1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--obs-on-signal", action="store_true",
                    help="must match the policy used at training time")
    ap.add_argument("--name", default=None,
                    help="stem of metrics_<name>.json, e.g. s1 or single_s1")
    ap.add_argument("--save-figs", type=int, default=8,
                    help="how many qualitative figures to write (0 = none)")
    ap.add_argument("--obs-snr", type=float, default=None,
                    help="robustness sweep: add measurement noise at this SNR "
                         "in dB (e.g. 20). Omit for clean observations.")
    ap.add_argument("--obs-scale", type=float, default=1.0,
                    help="robustness sweep: multiply the sensor budget")
    a = ap.parse_args()
    run(a.ckpt, a.cache, a.setup, a.split, a.out, a.gpu, a.max_samples,
        obs_on_signal=a.obs_on_signal, name=a.name, save_figs=a.save_figs,
        obs_snr=a.obs_snr, obs_scale=a.obs_scale)
