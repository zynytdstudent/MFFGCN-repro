#!/bin/bash
# Follow-up after the 26-run queue: test whether the AFM gate is the reason the
# full model underperforms the single-encoder variant.
#
# Diagnosis from eval_ALL: Single Encoder 0.0417 vs full MFFGCN 0.0770 on S1.
# The paper's eq.(4) gate is softmax over the CHANNEL axis; with C=512 every
# channel ends up with a mean weight of 1/512, so the environment branch is
# scaled down by two orders of magnitude before fusion. 'spatial' normalises
# over H*W instead and keeps the gate at unit magnitude.
#
# Waits for tmux session all_experiments to disappear, then runs spatial on
# S1 and S2 (the two setups where we have full-model + single-encoder numbers).
set -u
ROOT=/public2/czz/ZhangYu/project/mffgcn
cd "$ROOT/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh

echo "[$(date '+%F %T')] followup: GPU0 is free -> starting AFM spatial runs" | tee -a ../logs/followup.log

COMMON="--cache ../data/cache --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
        --workers 8 --val-batches 60 --no-augment --obs-on-signal"

for s in s1 s2; do
    tag="afm_spatial_$s"
    name="spatial_$s"
    if [ -d "runs/$tag" ]; then
        echo "[$(date '+%F %T')] SKIP  $tag (run dir exists, already trained or running)" | tee -a ../logs/followup.log
    else
        echo "[$(date '+%F %T')] START $tag" | tee -a ../logs/followup.log
        CUDA_VISIBLE_DEVICES=0 python train.py $COMMON --setup "$s" \
            --gpu 0 --tag "$tag" --afm-mode spatial > "../logs/$tag.log" 2>&1
        if [ ! -f "runs/$tag/best.pt" ]; then
            echo "[$(date '+%F %T')] FAIL  $tag (see logs/$tag.log)" | tee -a ../logs/followup.log
            continue
        fi
        echo "[$(date '+%F %T')] DONE  $tag $(tail -1 ../logs/$tag.log)" | tee -a ../logs/followup.log
    fi
    python evaluate.py --ckpt "runs/$tag/best.pt" --cache ../data/cache \
        --setup "$s" --split test --out ../eval_ALL --gpu 0 \
        --obs-on-signal --name "$name" --save-figs 0 \
        > "../logs/eval_$name.log" 2>&1
    echo "[$(date '+%F %T')] EVAL  $name" | tee -a ../logs/followup.log
done

echo "[$(date '+%F %T')] followup finished" | tee -a ../logs/followup.log
