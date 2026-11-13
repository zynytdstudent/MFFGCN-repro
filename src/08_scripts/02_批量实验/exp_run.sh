#!/bin/bash
# Controlled experiments to diagnose the generalisation gap.
#   A : baseline widths + softmax AFM + augmentation
#   B : baseline widths + sigmoid AFM + augmentation
#   C : narrow widths + softmax AFM + augmentation
set -u
ROOT=/public2/czz/ZhangYu/project/mffgcn
cd "$ROOT/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh

COMMON="--cache ../data/cache --setup s1 --epochs 6 --decay-epoch 99 \
        --bs 32 --lr 1e-3 --workers 8 --val-batches 60"

CUDA_VISIBLE_DEVICES=0 python train.py $COMMON --gpu 0 --tag expA \
    > ../logs/expA.log 2>&1 &
p1=$!
sleep 4
CUDA_VISIBLE_DEVICES=1 python train.py $COMMON --gpu 1 --tag expB \
    --afm-mode sigmoid > ../logs/expB.log 2>&1 &
p2=$!
sleep 4
CUDA_VISIBLE_DEVICES=2 python train.py $COMMON --gpu 2 --tag expC \
    --widths 24,48,96,192 --bottleneck 384 > ../logs/expC.log 2>&1 &
p3=$!

echo "A=$p1 B=$p2 C=$p3"
wait $p1 $p2 $p3
echo "[$(date '+%F %T')] experiments finished"
for t in A B C; do
    echo "===== $t ====="
    grep '^\[ep' "../logs/exp$t.log" | tail -3
    grep '^\[info\] widths' "../logs/exp$t.log"
done
