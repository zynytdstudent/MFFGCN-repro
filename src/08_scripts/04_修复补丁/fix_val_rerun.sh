#!/usr/bin/env bash
# Re-run the three main setups after fixing the validation-set mismatch.
#
# Before the fix, build_loaders() gave the training split on-signal sensors but
# left val/test on uniform placement, so early stopping was selecting a
# checkpoint for the wrong task: 13 of 29 runs stopped at epoch 1.
#
# Nothing else changes, so the difference against FINAL_s1/s2/s3 isolates the
# effect of the validation policy alone.
set -u
cd "$(dirname "$0")/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1

mkdir -p ../logs ../eval_ALL
LOG=../logs/fixval.log

# grab the first card that is genuinely idle; the box is shared with other users
pick_gpu () {
    for i in 0 1 2 3; do
        read -r mem util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu \
            --format=csv,noheader,nounits -i "$i" | tr -d ' ')
        if [ "${mem:-99999}" -lt 3000 ] && [ "${util:-100}" -lt 15 ]; then
            echo "$i"; return 0
        fi
    done
    return 1
}

echo "[$(date '+%F %T')] fixval: waiting for bs15 session to finish" | tee -a $LOG
while tmux has-session -t bs15 2>/dev/null; do
    sleep 180
done
echo "[$(date '+%F %T')] fixval: starting 3 main setups" | tee -a $LOG

for s in s1 s2 s3; do
    tag=fixval_$s
    if [ -f "runs/$tag/best.pt" ]; then
        echo "[$(date '+%F %T')] SKIP $tag" | tee -a $LOG
    else
        while ! gpu=$(pick_gpu); do
            echo "[$(date '+%F %T')] no idle GPU, retry in 5 min" | tee -a $LOG
            sleep 300
        done
        echo "[$(date '+%F %T')] START $tag on gpu$gpu" | tee -a $LOG
        CUDA_VISIBLE_DEVICES=$gpu python train.py --cache ../data/cache \
            --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
            --workers 8 --val-batches 60 --no-augment --obs-on-signal \
            --setup "$s" --gpu "$gpu" --tag "$tag" > "../logs/$tag.log" 2>&1
        echo "[$(date '+%F %T')] DONE $tag $(tail -1 ../logs/$tag.log)" | tee -a $LOG
    fi
    if [ -f "runs/$tag/best.pt" ] && [ ! -f "../eval_ALL/metrics_$tag.json" ]; then
        gpu=$(pick_gpu || echo 0)
        echo "[$(date '+%F %T')] EVAL $tag on gpu$gpu" | tee -a $LOG
        CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --ckpt "runs/$tag/best.pt" \
            --cache ../data/cache --setup "$s" --split test \
            --out ../eval_ALL --gpu "$gpu" --name "$tag" --save-figs 0 \
            --obs-on-signal >> $LOG 2>&1
    fi
done

echo "[$(date '+%F %T')] fixval finished" | tee -a $LOG
