"""Paper Fig.11 replication: estimation + deviation of 6 methods x 3 setups.

Layout follows the paper exactly:
    3 rows x 2 columns of method blocks, each block = 2 rows (Estimation /
    Deviation) x 3 columns (Setup 1 / 2 / 3), red rectangle marks the
    observation-constrained region.

Method order is the paper's caption: (a) MFFGCN (b) DeepAE (c) U-Net
(d) RadioUNet (e) Pixpix+ (f) RME-GAN.  Inference only, one test map per
setup, no training.
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle

from dataset import RadioMapSeer
from evaluate import build_from_ckpt
from baselines import build_baseline

METHODS = [
    ("MFFGCN", "txf_mffgcn", "mffgcn"),
    ("DeepAE", "bl_deepae", "deepae"),
    ("U-Net", "txf_single", "single"),
    ("RadioUNet", "bl_radiounet", "radiounet"),
    ("Pix2pix+", "bl_pix2pix", "pix2pix"),
    ("RME-GAN", "bl_rmegan", "pix2pix"),
]
SETUPS = ["s1", "s2", "s3"]


def load(tag, kind, device):
    ck = torch.load("runs/%s/best.pt" % tag, map_location=device, weights_only=False)
    arch = (ck.get("args") or {}).get("arch") or kind
    if kind in ("mffgcn", "single"):
        m, _ = build_from_ckpt(ck, device)
    else:
        m = build_baseline(kind)
        m.load_state_dict(ck["model"])
        m = m.to(device)
        if kind == "radiounet":
            m.stage = 2                     # both stages, as in the paper
    m.eval()
    return m, arch


@torch.no_grad()
def predict(model, arch, obs, env, mask):
    if arch == "mffgcn":
        _, y, _ = model(obs, env, mask)
    elif arch == "single":
        y = model(torch.cat([obs, env], dim=1))
    else:
        out = model(obs, env)
        y = out[-1] if isinstance(out, (tuple, list)) else out
    return y.clamp(0, 1)


def blind_box(allowed):
    """Bounding box of the observation-constrained region, in pixel coords."""
    m = allowed.detach().cpu().numpy()
    m = m.reshape(m.shape[-2], m.shape[-1])
    blind = m < 0.5
    if blind.sum() < 8:
        return None
    ys, xs = np.nonzero(blind)
    return xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="../data/cache")
    ap.add_argument("--idx", type=int, default=12, help="test index used for all setups")
    ap.add_argument("--out", default="/tmp/pfig11")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--dev-vmax", type=float, default=0.12)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    datasets = {s: RadioMapSeer(args.cache, "test", s, seed=args.seed) for s in SETUPS}
    idx = args.idx
    if idx < 0:                              # auto-pick the densest building map
        best, bestv = 0, -1.0
        for i in range(40):
            v = float((datasets["s1"][i]["env"][0] > 0.15).float().mean())
            if v > bestv:
                bestv, best = v, i
        idx = best
        print("[pick] densest sample idx=%d building=%.1f%%" % (idx, 100 * bestv), flush=True)
    sample = {s: datasets[s][idx] for s in SETUPS}
    for s in SETUPS:
        t = sample[s]
        print("[data] %s idx=%d obs=%.4f blind=%.1f%%" %
              (s, idx, float((t["obs"] > 0).float().mean()),
               100 * float((t["allowed"].reshape(-1) < 0.5).float().mean())), flush=True)

    panels = {}                              # panels[(name, setup)] = (est, dev)
    for name, tag, kind in METHODS:
        for s in SETUPS:
            m, arch = load("%s_%s" % (tag, s), kind, device)
            print("[load] %-10s %-18s arch=%s" % (name, "%s_%s" % (tag, s), arch), flush=True)
            b = sample[s]
            obs = b["obs"][None].to(device)
            env = b["env"][None].to(device)
            mask = b["mask"][None].to(device)
            tgt = b["target"][None].to(device)
            y = predict(m, arch, obs, env, mask)
            est = y[0, 0].detach().cpu().numpy()
            dev = (y - tgt)[0, 0].abs().detach().cpu().numpy()
            panels[(name, s)] = (est, dev)
            del m
            torch.cuda.empty_cache()

    gt = sample["s1"]["target"][0, 0].detach().cpu().numpy()

    fig = plt.figure(figsize=(9.8, 10.4), dpi=190)
    outer = gridspec.GridSpec(3, 2, figure=fig, hspace=0.30, wspace=0.10)
    tags = "abcdef"
    for k, (name, tag, kind) in enumerate(METHODS):
        r, c = divmod(k, 2)
        inner = outer[r, c].subgridspec(2, 3, hspace=0.06, wspace=0.05)
        for j, s in enumerate(SETUPS):
            est, dev = panels[(name, s)]
            box = blind_box(sample[s]["allowed"])
            ax_e = fig.add_subplot(inner[0, j])
            ax_d = fig.add_subplot(inner[1, j])
            ax_e.imshow(est, cmap="jet", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax_d.imshow(dev, cmap="Reds", vmin=0.0, vmax=args.dev_vmax, interpolation="nearest")
            for ax in (ax_e, ax_d):
                ax.set_xticks([])
                ax.set_yticks([])
                if box is not None:
                    # the constrained band sits flush with a map edge, so pull
                    # the frame 2 px inward or the outer edges fall off-canvas
                    x0 = min(box[0] + 2, 253)
                    y0 = min(box[1] + 2, 253)
                    w = max(3, min(box[2] - 4, 255 - x0))
                    h = max(3, min(box[3] - 4, 255 - y0))
                    ax.add_patch(Rectangle((x0, y0), w, h,
                                           fill=False, ec="#c00000", lw=1.4))
            if r == 0:
                ax_e.set_title("Setup %d" % (j + 1), fontsize=9.5, pad=3)
            if j == 0:
                ax_e.set_ylabel("Estimation", fontsize=8.5, labelpad=2)
                ax_d.set_ylabel("Deviation", fontsize=8.5, labelpad=2)
                ax_e.yaxis.set_label_position("left")
                ax_d.yaxis.set_label_position("left")
        # method name + panel letter, centred under the block
        fig.text(outer[r, c].get_position(fig).x0 + 0.5 * outer[r, c].get_position(fig).width,
                 outer[r, c].get_position(fig).y0 - 0.012,
                 "(%s) %s" % (tags[k], name), ha="center", va="top", fontsize=10)

    out = os.path.join(args.out, "fig11.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[out]", out, flush=True)


if __name__ == "__main__":
    main()
