#!/bin/bash
# Full-fidelity run: paper settings, no augmentation, 50 epochs, lr decay at 40.
#   GPU0 : single-encoder control (fast, tells us if 50 epochs is the missing piece)
#   GPU1 : MFFGCN Setup 1
#   GPU2 : MFFGCN Setup 2
set -u
ROOT=/public2/czz/ZhangYu/project/mffgcn
cd "$ROOT/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh

COMMON="--cache ../data/cache --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
        --workers 8 --val-batches 60 --no-augment"

CUDA_VISIBLE_DEVICES=0 python train.py $COMMON --gpu 0 --tag ctrl_single50 \
    --arch single > ../logs/ctrl_single50.log 2>&1 &
p1=$!
sleep 4
CUDA_VISIBLE_DEVICES=1 python train.py $COMMON --gpu 1 --tag s1_50 \
    --setup s1 > ../logs/s1_50.log 2>&1 &
p2=$!
sleep 4
CUDA_VISIBLE_DEVICES=2 python train.py $COMMON --gpu 2 --tag s2_50 \
    --setup s2 > ../logs/s2_50.log 2>&1 &
p3=$!

echo "started: $p1 $p2 $p3 at $(date '+%F %T')"
wait
echo "[$(date '+%F %T')] ALL DONE"
