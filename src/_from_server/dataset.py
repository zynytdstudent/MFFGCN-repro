"""RadioMapSeer dataset with the five uneven-sampling setups of the paper."""

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

IMG = 256


# --------------------------------------------------------------------------
# paper preprocessing:  [-127,-47] dB -> [0,1] -> drop <0.2 -> renormalise
# --------------------------------------------------------------------------
def preprocess_gain(raw_uint8):
    x = raw_uint8.astype(np.float32) / 255.0
    x[x < 0.2] = 0.0
    x = (x - 0.2) / 0.8
    return np.clip(x, 0.0, 1.0, out=x)


def restricted_area_map(frac, rng):
    """
    Rectangular observation-constrained region flush with one map edge.
    frac = share of the total map area.  Returns bool mask, True = observable.
    """
    allowed = np.ones((IMG, IMG), dtype=bool)
    side = int(rng.integers(4))
    if frac <= 0:
        return allowed
    if side in (0, 1):                      # top / bottom
        h = int(round(IMG * frac))
        h = max(1, min(IMG - 1, h))
        if side == 0:
            allowed[:h, :] = False
        else:
            allowed[-h:, :] = False
    else:                                   # left / right
        w = int(round(IMG * frac))
        w = max(1, min(IMG - 1, w))
        if side == 2:
            allowed[:, :w] = False
        else:
            allowed[:, -w:] = False
    return allowed


def two_side_densities(ratio_a, ratio_b, rng):
    """
    Paper Setup 5: split the radio map into two halves along a random axis,
    then sample ratio_a of the pixels from one half and ratio_b from the other.

    Note the semantics: the paper samples at a *ratio* (1% and 10% of the
    pixels), it does not black out a 1% / 10% *area*.  The pair of halves
    (vertical or horizontal) and which half gets which ratio are both random,
    so the model cannot learn a canonical asymmetry.

    Returns [(mask_half_1, ratio_1), (mask_half_2, ratio_2)].
    """
    half = IMG // 2
    vertical = bool(rng.integers(2))          # True -> top / bottom halves
    flip = bool(rng.integers(2))
    ra, rb = (ratio_b, ratio_a) if flip else (ratio_a, ratio_b)
    m1 = np.zeros((IMG, IMG), dtype=bool)
    m2 = np.zeros((IMG, IMG), dtype=bool)
    if vertical:
        m1[:half, :] = True
        m2[half:, :] = True
    else:
        m1[:, :half] = True
        m2[:, half:] = True
    return [(m1, ra), (m2, rb)]


class RadioMapSeer(Dataset):
    """
    setup:
        's1'  all samples uniformly random over the whole map
        's2'  25% rectangular constrained region, samples on the rest
        's3'  constrained region uniform in [10%, 50%] of the map
        's4'  every pixel kept, sampled at a uniform-random RATIO in
              [1%, 10%] of all pixels (paper Setup 4 / RME-GAN's original
              protocol: "sample the radio map at a random ratio from 1% to
              10% in each region"), i.e. 655-6554 observations
        's5'  map split into two random halves, one sampled at 1% and the
              other at 10% density (paper Setup 5)
    """

    def __init__(self, cache_dir, split="train", setup="s1",
                 n_min=10, n_max=300, seed=0, img=IMG,
                 cache_gain=True, augment=False, obs_on_signal=False,
                 split_mode="city", obs_snr=None, obs_scale=1.0):
        self.cache_dir = cache_dir
        self.setup = setup.lower()
        self.n_min, self.n_max = n_min, n_max
        self.img = img
        self.seed = seed
        self.augment = augment
        self.obs_on_signal = obs_on_signal
        # --- robustness knobs, used only by the evaluation sweep ----------
        # obs_snr   : measurement noise, SNR in dB relative to the RMS of the
        #             observed values. None / 0 disables it.
        # obs_scale : multiplier on the sensor budget, so 2.0 doubles the
        #             number of observations without touching the setup.
        self.obs_snr = obs_snr
        self.obs_scale = float(obs_scale)

        idx_path = os.path.join(cache_dir, "index.csv")
        rows = []
        with open(idx_path) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r["idx"]), int(r["city_id"]), int(r["tx_id"])))

        if split_mode == "pair":
            # diagnostic only: split individual (map, Tx) pairs instead of whole
            # maps, so the test maps were seen during training
            perm = np.random.default_rng(12345).permutation(len(rows))
            n = len(perm)
            a, b = int(0.8 * n), int(0.9 * n)
            part = {"train": perm[:a], "val": perm[a:b],
                    "test": perm[b:]}.get(split, perm)
            self.rows = [rows[i] for i in part]
        else:
            cities = sorted({r[1] for r in rows})
            if split == "train":
                keep = set(cities[:500])
            elif split == "val":
                keep = set(cities[500:600])
            elif split == "test":
                keep = set(cities[600:])
            else:
                keep = set(cities)
            self.rows = [r for r in rows if r[1] in keep]

        self.gain = np.load(os.path.join(cache_dir, "gain_dpm.npy"), mmap_mode="r")
        self.buildings = np.load(os.path.join(cache_dir, "buildings.npy"), mmap_mode="r")
        self.tx = np.load(os.path.join(cache_dir, "tx_coords.npy"))
        bld_city = np.load(os.path.join(cache_dir, "buildings_city.npy"))
        self.city_to_bidx = {int(c): i for i, c in enumerate(bld_city)}

        self._cache = {} if cache_gain else None
        # bumped every epoch so that the observation set is re-drawn
        self.epoch = 0

    def set_epoch(self, ep):
        self.epoch = int(ep)

    def __len__(self):
        return len(self.rows)

    def _load_gain(self, i):
        if self._cache is None:
            return self.gain[self.rows[i][0]]
        if i not in self._cache:
            if len(self._cache) > 4000:
                self._cache.clear()
            self._cache[i] = np.array(self.gain[self.rows[i][0]])
        return self._cache[i]

    def __getitem__(self, i):
        idx, city, txid = self.rows[i]
        rng = np.random.default_rng([self.seed + i * 7919, self.epoch * 104729 + txid])

        raw = self._load_gain(i)
        target = preprocess_gain(raw)

        # ---- observation-constrained region -----------------------------
        # `regions` is None for Setups 1-3: one allowed area and a total
        # budget drawn from [n_min, n_max].  For Setups 4-5 it is a list of
        # (area mask, sampling ratio): the paper specifies a *density* there,
        # not a blocked area.
        allowed = np.ones((IMG, IMG), dtype=bool)
        regions = None
        if self.setup == "s1":
            pass
        elif self.setup == "s2":
            allowed = restricted_area_map(0.25, rng)
        elif self.setup == "s3":
            frac = float(rng.uniform(0.10, 0.50))
            allowed = restricted_area_map(frac, rng)
        elif self.setup == "s4":
            # "sample the radio map at a random ratio from 1% to 10%"
            regions = [(allowed, float(rng.uniform(0.01, 0.10)))]
        elif self.setup == "s5":
            regions = two_side_densities(0.01, 0.10, rng)
        else:
            raise ValueError(f"unknown setup {self.setup}")

        # ---- sparse observations ----------------------------------------
        obs_flat = np.zeros(IMG * IMG, dtype=np.float32)
        tgt_flat = target.ravel()
        if regions is None:
            if self.obs_on_signal:
                cand = np.flatnonzero((tgt_flat > 0) & allowed.ravel())
                if cand.size < self.n_min:
                    cand = np.flatnonzero(allowed.ravel())
            else:
                cand = np.flatnonzero(allowed.ravel())
            n_obs = int(rng.integers(self.n_min, self.n_max + 1))
            if self.obs_scale != 1.0:
                n_obs = max(1, int(round(n_obs * self.obs_scale)))
            n_obs = min(n_obs, cand.size)
            obs_flat[rng.choice(cand, size=n_obs, replace=False)] = 1.0
        else:
            for area, ratio in regions:
                cand = np.flatnonzero(area.ravel())
                if self.obs_on_signal:
                    sig = np.flatnonzero(area.ravel() & (tgt_flat > 0))
                    if sig.size:
                        cand = sig
                n = int(round(ratio * int(area.sum())))
                if self.obs_scale != 1.0:
                    n = max(1, int(round(n * self.obs_scale)))
                n = max(1, min(n, cand.size))
                obs_flat[rng.choice(cand, size=n, replace=False)] = 1.0
        obs_mask = obs_flat.reshape(IMG, IMG)

        # ---- environment map --------------------------------------------
        bld = self.buildings[self.city_to_bidx[city]].astype(np.float32) / 255.0
        tx_map = np.zeros((IMG, IMG), dtype=np.float32)
        # tx_coords.npy is indexed by the GLOBAL row id (idx), not by the
        # position inside this split.  Using the local index here silently
        # gave val / test a random transmitter location while train (which
        # starts at global row 0) happened to be correct.
        r, c = int(self.tx[idx][0]), int(self.tx[idx][1])
        tx_map[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = 1.0

        # ---- symmetric augmentation -------------------------------------
        # City layouts carry no canonical orientation, so flips / transpose
        # are label-preserving. Everything geometric is transformed together.
        if self.augment:
            k = int(rng.integers(0, 4))
            if k:
                def tf(x):
                    if k == 1:
                        return np.ascontiguousarray(x[:, ::-1])
                    if k == 2:
                        return np.ascontiguousarray(x[::-1, :])
                    return np.ascontiguousarray(x.T)
                target = tf(target)
                obs_mask = tf(obs_mask)
                allowed = tf(allowed)
                bld = tf(bld)
                tx_map = tf(tx_map)

        obs = target * obs_mask

        # ---- measurement noise (robustness sweep only) -------------------
        if self.obs_snr:
            vals = obs[obs_mask > 0]
            if vals.size:
                rms = float(np.sqrt(np.mean(vals.astype(np.float64) ** 2)))
                sigma = max(rms, 1e-6) / (10.0 ** (float(self.obs_snr) / 20.0))
                noise = rng.normal(0.0, sigma, obs.shape).astype(np.float32)
                obs = np.clip(obs + noise * obs_mask, 0.0, 1.0)

        env = np.stack([bld, tx_map], axis=0)

        return {
            "obs": torch.from_numpy(obs[None]),
            "env": torch.from_numpy(env),
            "mask": torch.from_numpy(obs_mask[None]),
            "target": torch.from_numpy(target[None]),
            "allowed": torch.from_numpy(allowed.astype(np.float32)[None]),
            "index": torch.tensor(idx, dtype=torch.long),
        }


def build_loaders(cache_dir, setup, batch_size=32, workers=8, seed=0,
                  train_subset=None, augment=True, obs_on_signal=False,
                  split_mode="city"):
    from torch.utils.data import DataLoader

    tr = RadioMapSeer(cache_dir, "train", setup, seed=seed, cache_gain=False,
                      augment=augment, obs_on_signal=obs_on_signal,
                      split_mode=split_mode)
    # val / test MUST use the same sensor-placement policy as training,
    # otherwise early stopping selects a checkpoint for a different task.
    va = RadioMapSeer(cache_dir, "val", setup, seed=seed + 1000,
                      cache_gain=False, obs_on_signal=obs_on_signal,
                      split_mode=split_mode)
    te = RadioMapSeer(cache_dir, "test", setup, seed=seed + 2000,
                      cache_gain=False, obs_on_signal=obs_on_signal,
                      split_mode=split_mode)
    if train_subset:
        tr.rows = tr.rows[:train_subset]

    mk = lambda ds, sh: DataLoader(
        ds, batch_size=batch_size, shuffle=sh, num_workers=workers,
        pin_memory=True, drop_last=sh, persistent_workers=False,
        prefetch_factor=4 if workers > 0 else None)
    return mk(tr, True), mk(va, False), mk(te, False)
