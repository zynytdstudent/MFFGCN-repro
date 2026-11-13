#!/usr/bin/env bash
# Sensitivity sweep on the graph neighbourhood size k.
#
# The paper's default is k = 16, which is what every run so far used. This
# script trains k = 8 and k = 32 on Setup 1 and nothing else changes, so the
# three points 8 / 16 / 32 give one sensitivity curve with the default marked.
set -u
cd "$(dirname "$0")/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1

LOG=../logs/sensitivity.log
mkdir -p ../logs ../eval_ALL

# Same free-memory rule as fix_val_rerun.sh: shared box, dead tenants hold a
# few GB on otherwise idle cards, so test free memory instead of raw usage.
pick_gpu () {
    for i in 0 1 2 3; do
        IFS=',' read -r used total util < <(nvidia-smi \
            --query-gpu=memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits -i "$i" | tr -d ' ')
        free=$(( ${total:-0} - ${used:-99999} ))
        if [ "$free" -gt 20000 ] && [ "${util:-100}" -lt 30 ]; then
            echo "$i"; return 0
        fi
    done
    return 1
}

echo "[$(date '+%F %T')] sensitivity: waiting for fixval" | tee -a $LOG
while tmux has-session -t fixval 2>/dev/null; do
    sleep 180
done

for k in 8 32; do
    tag=sens_k${k}_s1
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
            --setup s1 --gpu "$gpu" --tag "$tag" --k "$k" \
            > "../logs/$tag.log" 2>&1
        echo "[$(date '+%F %T')] DONE $tag $(tail -1 ../logs/$tag.log)" | tee -a $LOG
    fi
    if [ -f "runs/$tag/best.pt" ] && [ ! -f "../eval_ALL/metrics_$tag.json" ]; then
        gpu=$(pick_gpu || echo 0)
        echo "[$(date '+%F %T')] EVAL $tag on gpu$gpu" | tee -a $LOG
        CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --ckpt "runs/$tag/best.pt" \
            --cache ../data/cache --setup s1 --split test \
            --out ../eval_ALL --gpu "$gpu" --name "$tag" --save-figs 0 \
            --obs-on-signal >> $LOG 2>&1
    fi
done

echo "[$(date '+%F %T')] sensitivity finished" | tee -a $LOG
