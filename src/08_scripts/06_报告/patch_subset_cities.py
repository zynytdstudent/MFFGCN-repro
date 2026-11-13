"""Add a --subset-cities knob so paper Fig.7 (NMSE vs number of samples) can be
reproduced: train on the first N maps (cities) of the training split, keeping
every transmitter of those maps.

Touches dataset.py / train.py / baseline_train.py.  Default 0 = full data.
"""
import io

changed = []


def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert old in s, "%s: anchor not found -> %s" % (path, old[:60])
        s = s.replace(old, new, 1)
    open(path, "w", encoding="utf-8").write(s)
    changed.append(path)


# ---------------- dataset.py ----------------
patch("dataset.py", [
    ('''def build_loaders(cache_dir, setup, batch_size=32, workers=8, seed=0,
                  train_subset=None, augment=True, obs_on_signal=False,
                  split_mode="city"):''',
     '''def build_loaders(cache_dir, setup, batch_size=32, workers=8, seed=0,
                  train_subset=None, augment=True, obs_on_signal=False,
                  split_mode="city", subset_cities=None):'''),
    ('''    if train_subset:
        tr.rows = tr.rows[:train_subset]''',
     '''    if train_subset:
        tr.rows = tr.rows[:train_subset]
    if subset_cities:
        # paper Fig.7 sweeps the number of training MAPS, not the number of
        # transmitter positions: keep the first N cities with all their rows.
        keep = sorted({r[1] for r in tr.rows})[: int(subset_cities)]
        tr.rows = [r for r in tr.rows if r[1] in set(keep)]'''),
])

# ---------------- train.py ----------------
patch("train.py", [
    ('''    ap.add_argument("--fixed-obs", action="store_true",
                    help="keep the same observation set every epoch")''',
     '''    ap.add_argument("--fixed-obs", action="store_true",
                    help="keep the same observation set every epoch")
    ap.add_argument("--subset-cities", type=int, default=0,
                    help="train on the first N maps only (paper Fig.7 sweep)")'''),
    ('''                               obs_on_signal=args.obs_on_signal,
                               split_mode=args.split_mode)''',
     '''                               obs_on_signal=args.obs_on_signal,
                               split_mode=args.split_mode,
                               subset_cities=(args.subset_cities or None))'''),
])

# ---------------- baseline_train.py ----------------
patch("baseline_train.py", [
    ('''    ap.add_argument("--obs-on-signal", action="store_true")''',
     '''    ap.add_argument("--obs-on-signal", action="store_true")
    ap.add_argument("--subset-cities", type=int, default=0,
                    help="train on the first N maps only (paper Fig.7 sweep)")'''),
    ('''                               train_subset=(args.smoke or None),
                               augment=not args.no_augment,
                               obs_on_signal=args.obs_on_signal)''',
     '''                               train_subset=(args.smoke or None),
                               augment=not args.no_augment,
                               obs_on_signal=args.obs_on_signal,
                               subset_cities=(args.subset_cities or None))'''),
])

print("patched:", ", ".join(changed))
