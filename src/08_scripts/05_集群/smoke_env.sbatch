#!/bin/bash
#SBATCH -J smoke_env
#SBATCH -p NVIDIAA100-PCIE-40GB
#SBATCH -A zy_221126030197
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 00:20:00
#SBATCH -o /public/home/zy_221126030197/mffgcn/logs/slurm-%x-%j.out
set -u
cd "$HOME/mffgcn/src" || exit 1
export OMP_NUM_THREADS=4
PY="$HOME/miniconda3/envs/mffgcn/bin/python"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY - <<'PYEOF'
import sys, torch
sys.path.insert(0, ".")
from dataset import RadioMapSeer
from baselines import build_baseline
from torch.utils.data import DataLoader
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
ds = RadioMapSeer("../data/cache", "test", "s1", n_min=50, n_max=50,
                  seed=7, cache_gain=True, augment=False, obs_on_signal=False)
ds.set_epoch(0)
dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
b = next(iter(dl))
print("batch", {k: tuple(v.shape) for k, v in b.items()})
m = build_baseline("ae", ae_skip="obs").cuda().eval()
with torch.no_grad():
    y = m(b["obs"].cuda(), b["env"].cuda())
print("ae skip=obs out", tuple(y.shape), float(y.mean()))
print("SMOKE_OK")
PYEOF
echo "rc=$?"
