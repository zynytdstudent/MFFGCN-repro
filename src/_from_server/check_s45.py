"""Verify the Setup 4 / Setup 5 rewrite.

1) regression: Setups 1-3 must produce byte-identical observation masks
   before and after the edit (the backup dataset.py.bak.txfix_1104 is loaded
   side by side);
2) Setup 4 : observations = 1%-10% of all 65536 pixels;
3) Setup 5 : two halves sampled at 1% and 10% density.
"""
import importlib.util
from importlib.machinery import SourceFileLoader

import numpy as np


def load(name, path):
    loader = SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m


CACHE = "../data/cache"
new = load("ds_new", "dataset.py")
old = load("ds_old", "dataset.py.bak.txfix_1104")

print("=== regression check: setups 1-3, obs pixel counts (5 samples each) ===")
for setup in ["s1", "s2", "s3"]:
    res = {}
    for tag, mod in [("old", old), ("new", new)]:
        ds = mod.RadioMapSeer(CACHE, "val", setup, seed=7)
        ds.set_epoch(0)
        n = [int(ds[i]["mask"].sum().item()) for i in range(5)]
        res[tag] = n
    ok = res["old"] == res["new"]
    print(f"  {setup}: old={res['old']} new={res['new']}  -> {'SAME' if ok else 'DIFFERENT!'}")

print("\n=== setup 4: uniform random density in [1%, 10%] ===")
ds = new.RadioMapSeer(CACHE, "val", "s4", seed=7)
n4 = [int(ds[i]["mask"].sum().item()) for i in range(40)]
print(f"  n_obs min={min(n4)} max={max(n4)} mean={np.mean(n4):.0f} "
      f"(expected 655..6554, mean ~3605)")

print("\n=== setup 5: two halves at 1% / 10% density ===")
for i in range(4):
    m = ds.__class__(CACHE, "val", "s5", seed=7)
    m.set_epoch(0)
    mask = m[i]["mask"][0].numpy()
    top, bot = mask[:128].sum(), mask[128:].sum()
    left, right = mask[:, :128].sum(), mask[:, 128:].sum()
    half = int(128 * 256)
    print(f"  sample {i}: top={int(top):5d} ({top/half:5.1%}) bot={int(bot):5d} ({bot/half:5.1%})"
          f" | left={int(left):5d} ({left/half:5.1%}) right={int(right):5d} ({right/half:5.1%})")

print("\n=== setup 4/5 shape sanity ===")
for s in ["s4", "s5"]:
    d = new.RadioMapSeer(CACHE, "val", s, seed=11)
    b = d[3]
    print(f"  {s}: obs{tuple(b['obs'].shape)} env{tuple(b['env'].shape)} "
          f"target{tuple(b['target'].shape)} allowed_sum={int(b['allowed'].sum())}")
