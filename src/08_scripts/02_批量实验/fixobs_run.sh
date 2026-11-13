#!/bin/bash
# Does re-drawing the observation set every epoch destabilise training?
#   GPU0 : MFFGCN, observations FIXED across epochs
#   GPU1 : single encoder, observations FIXED (10 epochs, fast signal)
set -u
ROOT=/public2/czz/ZhangYu/project/mffgcn
cd "$ROOT/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh

CUDA_VISIBLE_DEVICES=0 python train.py --cache ../data/cache --setup s1 \
    --epochs 6 --decay-epoch 99 --bs 32 --lr 1e-3 --workers 8 --val-batches 60 \
    --no-augment --fixed-obs --gpu 0 --tag fix_mffgcn > ../logs/fix_mffgcn.log 2>&1 &
p1=$!
sleep 4
CUDA_VISIBLE_DEVICES=1 python train.py --cache ../data/cache --setup s1 \
    --epochs 10 --decay-epoch 99 --bs 32 --lr 1e-3 --workers 8 --val-batches 60 \
    --no-augment --fixed-obs --arch single --gpu 1 --tag fix_single \
    > ../logs/fix_single.log 2>&1 &
p2=$!

echo "started $p1 $p2"
wait $p1 $p2
echo "[$(date '+%F %T')] done"
