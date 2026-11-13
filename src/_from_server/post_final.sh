#!/bin/bash
# Waits for FINAL training to finish, then evaluates all three setups on the
# test split (matching the training-time sampling policy) and on a slice of the
# training split so the generalisation gap can be quantified.
set -u
ROOT=/public2/czz/ZhangYu/project/mffgcn
cd "$ROOT/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh

echo "[$(date '+%F %T')] watcher started"

# ---- 1. wait for the training session to disappear -------------------------
for i in $(seq 1 240); do
    if ! tmux has-session -t final 2>/dev/null; then
        echo "[$(date '+%F %T')] tmux session 'final' is gone"
        break
    fi
    sleep 60
done

for s in s1 s2 s3; do
    echo "----- $s last lines -----"
    tail -4 "../logs/FINAL_$s.log"
done

# ---- 2. test-split evaluation, one GPU per setup ---------------------------
mkdir -p ../eval_FINAL
for s in s1 s2 s3; do
    case $s in
        s1) G=0 ;; s2) G=1 ;; s3) G=2 ;;
    esac
    python evaluate.py --ckpt "runs/FINAL_$s/best.pt" --cache ../data/cache \
        --setup "$s" --split test --out ../eval_FINAL --gpu "$G" \
        --obs-on-signal > "../logs/eval_$s.log" 2>&1 &
    sleep 3
done
wait
echo "[$(date '+%F %T')] test evaluation finished"
for s in s1 s2 s3; do
    echo "== $s =="
    tail -6 "../logs/eval_$s.log"
done

# ---- 3. same model on training maps: how much is memorisation? -------------
mkdir -p ../eval_FINAL/gap
python evaluate.py --ckpt runs/FINAL_s1/best.pt --cache ../data/cache \
    --setup s1 --split train --out ../eval_FINAL/gap --gpu 0 \
    --obs-on-signal --max-samples 800 > ../logs/gap_train.log 2>&1
echo "--- train-split slice (s1 model) ---"
tail -6 ../logs/gap_train.log

# ---- 4. comparison tables --------------------------------------------------
python compare.py --runs ../eval_FINAL --out ../eval_FINAL/reports \
    > ../logs/compare.log 2>&1
echo "--- compare ---"
cat ../logs/compare.log
echo "[$(date '+%F %T')] ALL DONE"
