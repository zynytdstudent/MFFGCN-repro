"""Patch dataset.py with three evaluation-only knobs used by paper Fig.9/10/14.

  setup="sq"      one square observable window, side / distance given in pixels
  impulse=<p>     salt & pepper noise on the observed pixels, probability p

Defaults keep the old behaviour, so training is unaffected.
"""
import sys

p = "dataset.py"
s = open(p, encoding="utf-8").read()
orig = s

# 1) __init__ signature
old = 'split_mode="city", obs_snr=None, obs_scale=1.0):'
assert old in s, "init signature not found"
s = s.replace(old, 'split_mode="city", obs_snr=None, obs_scale=1.0,\n'
                   '                 region_side=None, region_dist=None, impulse=None):')

# 2) store the knobs
old = "        self.obs_scale = float(obs_scale)\n"
assert old in s, "obs_scale assignment not found"
s = s.replace(old, old +
              "        # --- paper Fig.9 / Fig.10 / Fig.14 knobs (eval only) -------------\n"
              "        # region_side : side of the square observable window, pixels\n"
              "        # region_dist : its distance from the map centre, pixels\n"
              "        # impulse     : salt & pepper noise probability on observed px\n"
              "        self.region_side = region_side\n"
              "        self.region_dist = region_dist\n"
              "        self.impulse = impulse\n", 1)

# 3) new setup branch
old = '''        elif self.setup == "s5":
            regions = two_side_densities(0.01, 0.10, rng)'''
assert old in s, "setup s5 branch not found"
s = s.replace(old, old + '''
        elif self.setup == "sq":
            # paper Fig.9 / Fig.10: a single square observable window whose
            # side length and offset from the map centre are given explicitly.
            allowed = square_window(self.region_side, self.region_dist)''')

# 4) helper
anchor = "def two_side_densities("
assert anchor in s
helper = '''def square_window(side_px, dist_px, img=IMG):
    """Square 'restricted region' used by paper Fig.9 / Fig.10.

    side_px : side of the observable square, in pixels
    dist_px : distance from the map centre to the centre of the square
    Everything outside the square is unobservable (True = observable).
    """
    allowed = np.zeros((img, img), dtype=bool)
    s = max(4, min(img, int(round(float(side_px)))))
    cy = img // 2
    cx = min(img - s // 2 - 1, max(s // 2, int(round(img / 2.0 + float(dist_px)))))
    y0 = max(0, min(img - s, cy - s // 2))
    x0 = max(0, min(img - s, cx - s // 2))
    allowed[y0:y0 + s, x0:x0 + s] = True
    return allowed


'''
s = s.replace(anchor, helper + anchor, 1)

# 5) impulse noise, injected right before the env stack
old = "        env = np.stack([bld, tx_map], axis=0)"
assert old in s, "env stack not found"
s = s.replace(old, '''        # ---- impulse (salt & pepper) noise, paper Fig.14 ------------------
        if self.impulse:
            prob = float(self.impulse)
            hit = (rng.random(obs.shape) < prob) & obs_mask
            if hit.any():
                salt = rng.random(obs.shape) < 0.5
                obs = np.where(hit & salt, 1.0,
                               np.where(hit, 0.0, obs)).astype(np.float32)

''' + old, 1)

open(p, "w", encoding="utf-8").write(s)
print("patched dataset.py: %d -> %d bytes" % (len(orig), len(s)))
