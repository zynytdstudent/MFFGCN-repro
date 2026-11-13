#!/usr/bin/env bash
# Faithfulness check against Algorithm 1 of the paper.
#
# Algorithm 1 line 4 fixes  T_max = 50, B = 15, alpha = 0.001.
# Every run so far used B = 32, so B is the one explicit hyper-parameter we
# never matched. This script re-runs Setup 1 with B = 15 and changes nothing
# else, so the comparison is single-variable:
#
#   FINAL_s1   B=32  augment=off  obs=on-signal  amp=on   -> 0.0770
#   bs15_s1    B=15  augment=off  obs=on-signal  amp=on   -> ?
#
# It waits for the AFM check (tmux session afm_run) to finish before starting,
# so it never competes for the GPU.
set -u
cd "$(dirname "$0")/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1

mkdir -p ../logs ../eval_ALL

echo "[$(date '+%F %T')] bs15: waiting for afm_run to finish" | tee -a ../logs/paper_bs15.log
while tmux has-session -t afm_run 2>/dev/null; do
    sleep 120
done
echo "[$(date '+%F %T')] bs15: GPU0 free -> starting" | tee -a ../logs/paper_bs15.log

tag=bs15_s1
if [ -f "runs/$tag/best.pt" ]; then
    echo "[$(date '+%F %T')] SKIP $tag" | tee -a ../logs/paper_bs15.log
else
    echo "[$(date '+%F %T')] START $tag" | tee -a ../logs/paper_bs15.log
    CUDA_VISIBLE_DEVICES=0 python train.py --cache ../data/cache \
        --epochs 50 --decay-epoch 40 --bs 15 --lr 1e-3 \
        --workers 8 --val-batches 60 --no-augment --obs-on-signal \
        --setup s1 --gpu 0 --tag "$tag" > ../logs/$tag.log 2>&1
fi

if [ -f "runs/$tag/best.pt" ] && [ ! -f "../eval_ALL/metrics_bs15_s1.json" ]; then
    echo "[$(date '+%F %T')] EVAL $tag" | tee -a ../logs/paper_bs15.log
    CUDA_VISIBLE_DEVICES=0 python evaluate.py --ckpt "runs/$tag/best.pt" \
        --cache ../data/cache --setup s1 --split test \
        --out ../eval_ALL --gpu 0 --name bs15_s1 --save-figs 0 --obs-on-signal \
        >> ../logs/paper_bs15.log 2>&1
fi
echo "[$(date '+%F %T')] bs15 finished" | tee -a ../logs/paper_bs15.log
