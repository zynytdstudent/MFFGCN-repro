#!/usr/bin/env bash
# ==========================================================================
# gen_fix_jobs.sh —— 把本轮"修差距"的任务丢到集群（Slurm，tc6000）
#
# 为什么用集群：zjut148 的 4 张卡全被占（pix2pix_s1_fix / λ0.5 / λ1.0 / abch2s2），
# 而 ivsn 的 A100 分区 21 张卡只要有一张空出来就能立刻接活。
#
# 任务分两类：
#   A. AE / DeepAE 的 skip 口径探针（论文写了有 skip，我们原来没有）
#      判定标准：Fig.7 里 AE/DeepAE 两条线必须变陡，AE→0.04、DeepAE→0.015
#   B. Setup 3 的两条 cGAN 重训（Fig.9/10/14/15/16 全部用 s3 权重，
#      现在 RME-GAN s3 = 0.0221 vs 论文 0.0055，差 4 倍）
#
# 用法：bash gen_fix_jobs.sh [--dry]
# 每个作业独占一张 A100-40G，参数与 zjut148 上的 run_fix_gan.sh 逐字一致。
# ==========================================================================
set -u
ROOT="$HOME/mffgcn"
PY="$HOME/miniconda3/envs/mffgcn/bin/python"
PART="NVIDIAA100-PCIE-40GB"
ACCT="zy_221126030197"
DRY=0
[ "${1:-}" = "--dry" ] && DRY=1

mkdir -p "$ROOT/sbatch" "$ROOT/logs" "$ROOT/runs"

submit () {
    local tag=$1 kind=$2 setup=$3 extra=$4 tlim=${5:-04:00:00}
    local f="$ROOT/sbatch/$tag.sbatch"

    {
        echo '#!/bin/bash'
        echo "#SBATCH -J $tag"
        echo "#SBATCH -p $PART"
        echo "#SBATCH -A $ACCT"
        echo "#SBATCH -N 1"
        echo "#SBATCH -n 1"
        echo "#SBATCH --gres=gpu:1"
        echo "#SBATCH -c 8"
        echo "#SBATCH -t $tlim"
        echo "#SBATCH -o $ROOT/logs/slurm-%x-%j.out"
        echo
        echo "cd $ROOT/src || exit 1"
        echo "export OMP_NUM_THREADS=8"
        echo "export MKL_NUM_THREADS=8"
        if [ "$kind" = "rme_gan" ]; then
            echo "CUDA_VISIBLE_DEVICES=0 $PY rme_gan_train.py \\"
            echo "    --cache ../data/cache --setup $setup --tag $tag \\"
            echo "    --phase both --phase1-epochs 20 --epochs 50 --decay-epoch 40 \\"
            echo "    --bs 32 --lr 2e-4 --workers 8 --gpu 0 --val-batches 60 $extra"
        else
            echo "CUDA_VISIBLE_DEVICES=0 $PY baseline_train.py \\"
            echo "    --arch $kind --setup $setup --tag $tag --epochs 50 \\"
            echo "    --decay-epoch 40 --bs 32 --lr 1e-3 --workers 8 --gpu 0 \\"
            echo "    --val-batches 60 $extra"
        fi
        echo
        echo "rc=\$?"
        echo "echo \"[\$(date '+%F %T')] $tag exit=\$rc\""
    } > "$f"

    if [ "$DRY" = "1" ]; then
        echo "[dry] $tag -> $f"
        return
    fi
    if [ -f "$ROOT/src/runs/$tag/test_metrics.json" ]; then
        echo "[skip] $tag 已完成"
        return
    fi
    sbatch "$f" | sed "s/^/[$tag] /"
}

# ---- A. AE / DeepAE skip 口径 ----
submit "ae_s1_skipobs"     ae     s1 "--ae-skip obs"
submit "ae_s1_skipenc"     ae     s1 "--ae-skip enc"
submit "deepae_s1_skipobs" deepae s1 "--ae-skip obs"

# ---- B. Setup 3 的 cGAN 重训 ----
submit "bl_pix2pix_s3_fix" pix2pix s3 "--gan-lr 2e-4" 05:00:00
submit "bl_rmegan_s3_fix"  rme_gan s3 ""              05:00:00
