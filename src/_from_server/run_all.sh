#!/bin/bash
# Prepare cache + launch the three setups in parallel on GPU 0/1/2.
set -u
ROOT=/public2/czz/ZhangYu/project/mffgcn
SRC=$ROOT/src
CACHE=$ROOT/data/cache
LOGS=$ROOT/logs
mkdir -p "$CACHE" "$LOGS"

source /public2/czz/ZhangYu/activate_zhangyu.sh

cd "$SRC" || exit 1

echo "[$(date '+%F %T')] ==== STEP 1: preprocessing ==="
if [ -f "$CACHE/index.csv" ]; then
    echo "cache already present, skip"
else
    python prepare_data.py --root "$ROOT/data/raw" --out "$CACHE" --sim DPM || exit 1
fi

echo "[$(date '+%F %T')] ==== STEP 2: smoke test ===="
python train.py --cache "$CACHE" --setup s1 --epochs 2 --smoke 3000 \
       --gpu 0 --workers 6 --tag smoke_s1 > "$LOGS/smoke_s1.log" 2>&1
tail -12 "$LOGS/smoke_s1.log"

echo "[$(date '+%F %T')] ==== STEP 3: full training x3 in parallel ===="
pids=()
i=0
for s in s1 s2 s3; do
    gpu=$i
    i=$((i+1))
    echo "launch $s on GPU$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python train.py \
        --cache "$CACHE" --setup "$s" --epochs 50 --decay-epoch 40 \
        --bs 32 --lr 1e-3 --gpu "$gpu" --workers 10 \
        --tag "${s}_full" > "$LOGS/train_${s}.log" 2>&1 &
    pids+=($!)
    sleep 3
done

echo "waiting for: ${pids[*]}"
for p in "${pids[@]}"; do wait "$p"; done

echo "[$(date '+%F %T')] ==== ALL TRAINING DONE ===="
for s in s1 s2 s3; do
    echo "--- $s ---"
    tail -20 "$LOGS/train_${s}.log"
done
