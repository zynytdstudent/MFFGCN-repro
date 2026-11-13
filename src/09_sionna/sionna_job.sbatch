#!/bin/bash
#SBATCH -J sionna
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /public/home/zy_221126030197/work/mffgcn/logs/slurm-%x-%j.out
#SBATCH -e /public/home/zy_221126030197/work/mffgcn/logs/slurm-%x-%j.err
#
# Sionna RT 作业模板（集群 ivsn / Slurm）
#
# 环境：~/venvs/sionna（独立 venv，python 3.10 + sionna-rt 2.0.0 + mitsuba 3.7.1）
#       与训练环境 mffgcn 完全隔离，互不影响。
#
# 用法：
#   sbatch -p NVIDIAA100-PCIE-40GB scripts/sionna_job.sbatch your_rt_script.py
#
# 注意：
#   1) 必需驱动 >= R535（OptiX 8）。当前集群计算节点为 515.43.04，
#      在升级前 GPU 后端会直接报 "Could not initialize OptiX"。
#   2) 不要试试在登录节点 import sionna.rt —— 登录节点无 GPU 且无 libLLVM，
#      会报 libLLVM.so could not be found，属正常现象。
#
set -u

VENV=$HOME/venvs/sionna
PY=$VENV/bin/python

if [ ! -x "$PY" ]; then
    echo "[fail] 未找到 $PY" >&2
    exit 1
fi

SCRIPT=${1:?用法: sbatch sionna_job.sbatch <your_rt_script.py>}
shift || true

echo "[sionna] node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "[sionna] python=$PY"
"$PY" -c "import sionna.rt, mitsuba; print('[sionna] sionna-rt', mitsuba.__version__)" || exit 1

"$PY" "$SCRIPT" "$@"
