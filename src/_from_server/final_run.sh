#!/bin/bash
# FINAL RUN: three sampling setups, 50 epochs, lr decay at epoch 40.
# Sensors are placed where signal exists (--obs-on-signal); this deviates from
# the paper's unstated sampling policy but consistently outperformed whole-map
# uniform sampling in the controlled comparison.
set -u
ROOT=/public2/czz/ZhangYu/project/mffgcn
cd "$ROOT/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh

COMMON="--cache ../data/cache --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
        --workers 8 --val-batches 60 --no-augment --obs-on-signal"

CUDA_VISIBLE_DEVICES=0 python train.py $COMMON --setup s1 --gpu 0 --tag FINAL_s1 \
    > ../logs/FINAL_s1.log 2>&1 &
sleep 4
CUDA_VISIBLE_DEVICES=1 python train.py $COMMON --setup s2 --gpu 1 --tag FINAL_s2 \
    > ../logs/FINAL_s2.log 2>&1 &
sleep 4
CUDA_VISIBLE_DEVICES=2 python train.py $COMMON --setup s3 --gpu 2 --tag FINAL_s3 \
    > ../logs/FINAL_s3.log 2>&1 &

echo "[$(date '+%F %T')] three setups launched"
wait
echo "[$(date '+%F %T')] FINAL TRAINING COMPLETE"
for s in s1 s2 s3; do
    echo "===== $s ====="
    tail -3 "../logs/FINAL_$s.log"
done
