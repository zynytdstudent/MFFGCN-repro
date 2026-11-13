#!/bin/bash
# ==========================================================================
# lambda_sweep.sh —— 论文表 VII（λ 敏感性）补跑
#
# 论文原文（results/paper_text.txt 里可查）：
#   "We choose λ1 from the levels of 0.1, 0.5 and 1. When λ1 is 0.5,
#    the MFFGCN achieves the best performance ... We choose from three
#    levels, 0.00001, 0.0001 and 0.001, and MFFGCN achieves the best
#    performance when λ3 is 0.0001."
#
# 我们已有 (λ1=0.1, λ3=1e-4) 的主模型，所以还要补 4 个点：
#   λ1=0.5 / λ1=1.0 / λ3=1e-5 / λ3=1e-3
# 每个点是 Setup 1、50 epoch、decay-40、seed 0 —— 与 txf_mffgcn_s1 完全同口径，
# 这样表 VII 内部可直接横比，也能和主模型对上。
#
# 服务器四张卡长期被别人占着，所以这里做成"抢空闲卡"的调度器：
#   每 2 分钟查一次，找 util<50% 且空闲显存>20GB 的卡；最多同时跑 2 个。
# 不会 kill 任何人的进程，也不会碰 autopilot5 / abch2s2 的会话。
#
# 用法：bash lambda_sweep.sh [最多并发数，默认 2]
# ==========================================================================
set -u
MAXJ="${1:-2}"
cd "$(dirname "$0")" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1
cd src || exit 1
LOG="../logs/lambda_sweep.log"
mkdir -p ../logs

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

pick_gpu() {
    # 输出最空闲且满足条件的卡号；没有就输出空
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null | awk -F', *' '
        { idx=$1; util=$2+0; used=$3+0; tot=$4+0;
          if (util < 50 && (tot-used) > 20000) print idx, tot-used }' \
        | sort -k2 -nr | head -1 | cut -d' ' -f1
}

TAGS=("l1_0.5" "l1_1.0" "l3_1e-5" "l3_1e-3")
ARGS=("--l1 0.5" "--l1 1.0" "--l3 1e-5" "--l3 1e-3")

declare -A PID2TAG

running() {
    local n=0
    for p in "${!PID2TAG[@]}"; do
        if kill -0 "$p" 2>/dev/null; then n=$((n+1)); fi
    done
    echo "$n"
}

say "===== lambda sweep start (max_parallel=$MAXJ) ====="
for i in "${!TAGS[@]}"; do
    TAG="txf_lambda_${TAGS[$i]}"
    if [ -f "runs/${TAG}/test_metrics.json" ]; then
        say "SKIP $TAG (already done)"; continue
    fi
    # 等一个空槽
    while [ "$(running)" -ge "$MAXJ" ]; do
        for p in "${!PID2TAG[@]}"; do
            kill -0 "$p" 2>/dev/null || unset "PID2TAG[$p]"
        done
        sleep 120
    done
    while :; do
        GPU="$(pick_gpu)"
        [ -n "${GPU:-}" ] && break
        say "no free GPU, wait 300s"
        sleep 300
    done
    say "START $TAG on gpu${GPU}  (${ARGS[$i]})"
    CUDA_VISIBLE_DEVICES="$GPU" python train.py \
        --cache ../data/cache --setup s1 \
        --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
        --workers 6 --val-batches 60 --no-augment --seed 0 \
        --widths 32,64,128,256 --bottleneck 512 --k 16 \
        --gpu "$GPU" --arch mffgcn --afm-mode softmax ${ARGS[$i]} \
        --tag "$TAG" > "../logs/${TAG}.log" 2>&1 &
    PID2TAG[$!]="$TAG"
    sleep 90     # 等这个任务把利用率顶上去，避免下一轮又挑到同一张卡
done
wait
say "===== lambda sweep ALL DONE ====="
