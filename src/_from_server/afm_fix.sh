#!/usr/bin/env bash
# Re-run the AFM-spatial ablation under the corrected validation policy.
#
# afm_spatial_s1 / afm_spatial_s2 were trained while build_loaders() still gave
# val/test uniform sensor placement, so early stopping locked onto epoch 2 and
# the reported numbers (val NMSE 0.223 / 0.205) describe the wrong task.
# Everything else is copied verbatim from followup_afm.sh; only the validation
# split changes, so the delta against the single-encoder run isolates the AFM
# gate itself.
#
# Chained after fixval -> sensitivity: the whole box shares four cards, so each
# stage waits for the previous tmux session to disappear instead of racing it
# for the same free card.
set -u
cd "$(dirname "$0")/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1

mkdir -p ../logs ../eval_ALL
LOG=../logs/afmfix.log

# Pick a card by FREE MEMORY; dead tenants on this shared box hold a few GB on
# otherwise idle cards, so "memory.used < 3000" never matches anything.
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

echo "[$(date '+%F %T')] afmfix: waiting for fixval + sensitivity" | tee -a $LOG
while tmux has-session -t fixval 2>/dev/null || \
      tmux has-session -t sensitivity 2>/dev/null; do
    sleep 180
done
echo "[$(date '+%F %T')] afmfix: starting AFM-spatial reruns" | tee -a $LOG

COMMON="--cache ../data/cache --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
        --workers 8 --val-batches 60 --no-augment --obs-on-signal \
        --afm-mode spatial"

for s in s1 s2; do
    tag=fixval_afm_$s
    if [ -f "runs/$tag/best.pt" ]; then
        echo "[$(date '+%F %T')] SKIP $tag" | tee -a $LOG
    else
        while ! gpu=$(pick_gpu); do
            echo "[$(date '+%F %T')] no idle GPU, retry in 5 min" | tee -a $LOG
            sleep 300
        done
        echo "[$(date '+%F %T')] START $tag on gpu$gpu" | tee -a $LOG
        CUDA_VISIBLE_DEVICES=$gpu python train.py $COMMON \
            --setup "$s" --gpu "$gpu" --tag "$tag" > "../logs/$tag.log" 2>&1
        if [ ! -f "runs/$tag/best.pt" ]; then
            echo "[$(date '+%F %T')] FAIL $tag (see logs/$tag.log)" | tee -a $LOG
            continue
        fi
        echo "[$(date '+%F %T')] DONE $tag $(tail -1 ../logs/$tag.log)" | tee -a $LOG
    fi
    if [ ! -f "../eval_ALL/metrics_$tag.json" ]; then
        gpu=$(pick_gpu || echo 0)
        echo "[$(date '+%F %T')] EVAL $tag on gpu$gpu" | tee -a $LOG
        CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --ckpt "runs/$tag/best.pt" \
            --cache ../data/cache --setup "$s" --split test \
            --out ../eval_ALL --gpu "$gpu" --name "$tag" --save-figs 0 \
            --obs-on-signal >> $LOG 2>&1
    fi
done

echo "[$(date '+%F %T')] afmfix finished" | tee -a $LOG
