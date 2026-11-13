"""Shape / memory / loss sanity check that does not need the dataset."""

import time

import torch

from losses import MFFGCNLoss, MetricAccumulator, ms_ssim
from models import MFFGCN, count_params


def main(bs=8, devices=(0,)):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = MFFGCN().to(dev)
    print(f"params: {count_params(model)/1e6:.2f} M")

    obs = torch.rand(bs, 1, 256, 256, device=dev) * (torch.rand(bs, 1, 256, 256, device=dev) > 0.99)
    env = torch.rand(bs, 2, 256, 256, device=dev)
    mask = (obs > 0).float()
    target = torch.rand(bs, 1, 256, 256, device=dev)

    t0 = time.time()
    coarse, fine, res = model(obs, env, mask)
    print("shapes:", tuple(coarse.shape), tuple(fine.shape), tuple(res.shape))
    assert coarse.shape == target.shape == fine.shape == res.shape

    crit = MFFGCNLoss()
    loss, parts = crit(coarse, fine, target)
    print({k: float(v) for k, v in parts.items()})
    loss.backward()
    print("backward ok")

    s = ms_ssim(target, target)
    print(f"ms_ssim(x,x) = {s.item():.4f} (expect ~1.0)")
    s2 = ms_ssim(target, torch.rand_like(target))
    print(f"ms_ssim(x,noise) = {s2.item():.4f} (expect << 1)")

    if dev.type == "cuda":
        print(f"peak mem bs={bs}: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print(f"forward+backward {time.time()-t0:.2f}s")


if __name__ == "__main__":
    import sys
    main(bs=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
