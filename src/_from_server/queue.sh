#!/usr/bin/env bash
# ==========================================================================
# 多卡共享工作队列（work stealing）
#
# 用法： ./queue.sh <gpu> <spec...>
#        spec = <tagbase>:<arch>:<afm-mode>:<seed>
#   例： ./queue.sh 2 txf_mffgcn_s1:mffgcn:softmax:1 txf_single_s1:single:softmax:1
#
# 每个 spec 用 mkdir runs/.claim_<tag> 原子抢占，抢不到就跳到下一个，
# 所以在多张卡上并行起多个 worker 不会重复跑同一个 tag。
# 列表按"耗时长的排前面"（mffgcn 在前、single 在后），最小化总完成时间。
# seed=0 的 tag 不带后缀，与 rerun_fix.sh 保持一致。
# ==========================================================================
set -u
if [ $# -lt 2 ]; then
    echo "usage: $0 <gpu> <tagbase:arch:afm:seed ...>" >&2
    exit 2
fi
GPU="$1"; shift

cd "$(dirname "$0")/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1

LOG=../logs/queue_gpu${GPU}.log
mkdir -p ../logs
echo "=== worker gpu$GPU start $(date '+%F %T') with $# specs ===" | tee -a "$LOG"

for spec in "$@"; do
    IFS=':' read -r base arch afm seed <<< "$spec"
    if [ "$seed" = "0" ]; then tag="$base"; else tag="${base}_seed${seed}"; fi

    if [ -f "runs/$tag/best.pt" ]; then
        echo "[$(date '+%T')] SKIP $tag (best.pt 已存在)" | tee -a "$LOG"
        continue
    fi

    claim="runs/.claim_$tag"
    if ! mkdir "$claim" 2>/dev/null; then
        echo "[$(date '+%T')] SKIP $tag (已被其它 worker 抢占)" | tee -a "$LOG"
        continue
    fi

    echo "[$(date '+%F %T')] START $tag gpu$GPU arch=$arch afm=$afm seed=$seed" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU python train.py --cache ../data/cache \
        --setup s1 --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
        --workers 8 --val-batches 60 --no-augment --seed "$seed" \
        --widths 32,64,128,256 --bottleneck 512 --k 16 \
        --gpu "$GPU" --arch "$arch" --afm-mode "$afm" \
        --tag "$tag" > "../logs/$tag.log" 2>&1
    rc=$?

    if [ -f "runs/$tag/test_metrics.json" ]; then
        echo "[$(date '+%F %T')] DONE $tag -> $(tr -d '\n ' < runs/$tag/test_metrics.json | cut -c1-200)" | tee -a "$LOG"
    else
        echo "[$(date '+%F %T')] FAIL $tag rc=$rc :: $(tail -3 ../logs/$tag.log | tr '\n' ' ')" | tee -a "$LOG"
    fi
    rmdir "$claim" 2>/dev/null || true
done

echo "=== worker gpu$GPU finished $(date '+%F %T') ===" | tee -a "$LOG"
