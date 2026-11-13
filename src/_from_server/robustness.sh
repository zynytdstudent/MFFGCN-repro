#!/usr/bin/env bash
# Robustness sweep. No training: the same Setup 1 checkpoint is evaluated under
# degraded inputs.
#
#   noise  : Gaussian measurement noise at SNR 30 / 20 / 10 dB
#   budget : sensor count multiplied by 0.5 / 2 / 4
#
# The observation mask is derived from the same seed for every configuration,
# so only the degradation differs between runs.
set -u
cd "$(dirname "$0")/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1

CKPT=runs/FINAL_s1/best.pt
OUT=../eval_ALL
LOG=../logs/robustness.log
GPU=${1:-0}

mkdir -p "$OUT"

ev () {   # name, extra flags
    local name=$1; shift
    if [ -f "$OUT/metrics_$name.json" ]; then
        echo "[$(date '+%F %T')] SKIP $name" | tee -a $LOG; return
    fi
    echo "[$(date '+%F %T')] EVAL $name  ($*)" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=$GPU python evaluate.py --ckpt "$CKPT" \
        --cache ../data/cache --setup s1 --split test --out "$OUT" \
        --gpu "$GPU" --name "$name" --save-figs 0 --obs-on-signal "$@" \
        > "/tmp/eval_$name.log" 2>&1
    echo "[$(date '+%F %T')] DONE $name" | tee -a $LOG
}

ev rob_clean
for s in 30 20 10; do
    ev "rob_snr$s" --obs-snr "$s"
done
ev rob_bud0p5 --obs-scale 0.5
ev rob_bud2   --obs-scale 2.0
ev rob_bud4   --obs-scale 4.0

echo "[$(date '+%F %T')] robustness sweep finished" | tee -a $LOG
