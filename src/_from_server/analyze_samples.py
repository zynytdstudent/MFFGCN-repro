"""Quantitative read-out for the three qualitative sample figures."""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from dataset import RadioMapSeer
from models import MFFGCN

DEV = "cuda:0"


def stats(pred, gt, tx, region=None):
    out = {}
    m = np.ones_like(gt, bool) if region is None else region
    out["rmse"] = float(np.sqrt(((pred - gt)[m] ** 2).mean()))
    out["nmse"] = float(((pred - gt)[m] ** 2).sum() / max((gt[m] ** 2).sum(), 1e-12))
    out["bias"] = float((pred - gt)[m].mean())
    out["corr"] = float(np.corrcoef(pred[m].ravel(), gt[m].ravel())[0, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--idx", type=int, default=0)
    a = ap.parse_args()

    for setup, tag in [("s1", "FINAL_s1"), ("s2", "FINAL_s2"), ("s3", "FINAL_s3")]:
        ck = torch.load(f"runs/{tag}/best.pt", map_location=DEV, weights_only=False)
        model = MFFGCN().to(DEV)
        model.load_state_dict(ck["model"])
        model.eval()

        ds = RadioMapSeer(a.cache, "test", setup, seed=2024, obs_on_signal=True)
        b = ds[a.idx]
        obs = b["obs"][None].to(DEV)
        env = b["env"][None].to(DEV)
        mask = b["mask"][None].to(DEV)
        with torch.no_grad():
            coarse, fine, _ = model(obs, env, mask)

        f = fine.clamp(0, 1)[0, 0].cpu().numpy()
        c = coarse.clamp(0, 1)[0, 0].cpu().numpy()
        gt = b["target"][0].numpy()
        mk = b["mask"][0].numpy()
        n_obs = int(mk.sum())

        r, cc = np.unravel_index(gt.argmax(), gt.shape)
        yy, xx = np.mgrid[0:256, 0:256]
        dist = np.sqrt((yy - r) ** 2 + (xx - cc) ** 2)
        near = dist <= 10
        se = (f - gt) ** 2
        near_frac = float(se[near].sum() / max(se.sum(), 1e-12))
        near_area = float(near.mean())

        print(f"--- {setup} (best epoch {ck['epoch']}) sample #{a.idx} ---")
        print(f"  observations      : {n_obs} px  ({n_obs / 65536 * 100:.2f}% of map)")
        print(f"  GT peak           : {gt.max():.3f} at ({r},{cc})")
        print(f"  pred at GT peak   : {f[r, cc]:.3f}   (peak of pred {f.max():.3f})")
        print(f"  corr(pred, GT)    : {np.corrcoef(f.ravel(), gt.ravel())[0, 1]:.3f}")
        print(f"  10px ring around Tx: {near_area * 100:.1f}% of area, "
              f"{near_frac * 100:.1f}% of squared error")
        print(f"  whole map         : rmse {stats(f, gt, (r, cc))['rmse']:.4f} "
              f"bias {stats(f, gt, (r, cc))['bias']:+.4f}")
        print(f"  obs pixels fit    : rmse {np.sqrt(((f - gt)[mk > 0] ** 2).mean()):.4f}")
        if setup != "s1":
            allowed = b["allowed"][0].numpy()
            inside = allowed < 0.5
            print(f"  restricted region : {inside.mean() * 100:.1f}% of area, "
                  f"rmse inside {stats(f, gt, inside)['rmse']:.4f}, "
                  f"outside {stats(f, gt, ~inside)['rmse']:.4f}")
            print(f"  nmse inside       : {stats(f, gt, inside)['nmse']:.4f}")
        print(f"  coarse vs refined : rmse {stats(c, gt, (r, cc))['rmse']:.4f} -> "
              f"{stats(f, gt, (r, cc))['rmse']:.4f}")


if __name__ == "__main__":
    main()
