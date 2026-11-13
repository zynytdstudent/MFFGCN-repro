#!/bin/bash
# Launch Setup 1/2/3 training in parallel on GPU 0/1/2.
set -u
ROOT=/public2/czz/ZhangYu/project/mffgcn
cd "$ROOT/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh

echo "[$(date '+%F %T')] launching 3 setups"
i=0
pids=()
for s in s1 s2 s3; do
    CUDA_VISIBLE_DEVICES=$i python train.py \
        --cache ../data/cache --setup "$s" \
        --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
        --gpu "$i" --workers 8 --val-batches 60 \
        --tag "${s}_full" > "../logs/train_${s}.log" 2>&1 &
    pids+=($!)
    echo "  $s -> GPU$i pid=${pids[-1]}"
    i=$((i+1))
    sleep 5
done

echo "[$(date '+%F %T')] all launched, waiting"
for p in "${pids[@]}"; do wait "$p"; done
echo "[$(date '+%F %T')] ALL TRAINING FINISHED"
for s in s1 s2 s3; do
    echo "===== $s ====="
    tail -6 "../logs/train_${s}.log"
done
