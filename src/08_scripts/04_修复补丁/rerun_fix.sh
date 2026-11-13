#!/usr/bin/env bash
# ==========================================================================
# 修复 tx 索引错位后的重跑（2026-11-04）
#
# 根因：dataset.py 用"split 内局部下标 i"去索引全局 tx_coords.npy，
#       train 因为正好从全局行 0 开始所以碰巧正确，val/test 拿到的是
#       随机的发射机位置（平均偏 76 / 86 px）。
#       后果：所有历史 run 的 val / test 数字全部作废。
# 修复：tx -> tx[idx]，并让返回的 index 字段也用全局 idx。
#
# 本轮秩序（全部 setup=s1，与论文 Table III Setup 1 对齐）：
#   1) txf_single_s1   arch=single                 对照论文 U-Net 0.0182
#   2) txf_mffgcn_s1   arch=mffgcn softmax          对照论文 MFFGCN 0.0150
#   3) txf_sigmoid_s1  arch=mffgcn sigmoid          AFM 通道衰减假设
#   4) txf_spatial_s1  arch=mffgcn spatial          空间维度归一化门控
# 论文明确写 "No further augmentation or transformation is applied"，
# 所以 --no-augment 是**正确**的协议，不是偷懒。
# ==========================================================================
set -u
cd "$(dirname "$0")/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1

LOG=../logs/rerun_fix.log
mkdir -p ../logs

pick_gpu () {
    # 第一轮：真正空闲（显存够 + 利用率低）
    # 第二轮：只要显存够就上（和别人时分共享，彼此都会变慢但不会卡死）
    for pass in 1 2; do
        for i in 0 1 2 3; do
            IFS=',' read -r used total util < <( \
                nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
                --format=csv,noheader,nounits -i "$i" | tr -d ' ')
            free=$(( ${total:-0} - ${used:-99999} ))
            if [ "$pass" = "1" ]; then
                [ "$free" -gt 15000 ] && [ "${util:-100}" -lt 40 ] && { echo "$i"; return 0; }
            else
                [ "$free" -gt 12000 ] && { echo "$i"; return 0; }
            fi
        done
    done
    return 1
}

run_one () {
    local tag="$1"; shift
    if [ -f "runs/$tag/best.pt" ]; then
        echo "[$(date '+%F %T')] SKIP $tag (已存在)" | tee -a "$LOG"
        return 0
    fi
    local gpu=""
    while [ -z "$gpu" ]; do
        gpu=$(pick_gpu) || true
        if [ -z "$gpu" ]; then
            echo "[$(date '+%F %T')] no idle GPU, retry in 5 min" | tee -a "$LOG"
            sleep 300
        fi
    done
    echo "[$(date '+%F %T')] START $tag on gpu$gpu : $*" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$gpu python train.py --cache ../data/cache \
        --setup s1 --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
        --workers 8 --val-batches 60 --no-augment --seed 0 \
        --widths 32,64,128,256 --bottleneck 512 --k 16 \
        --gpu "$gpu" --tag "$tag" "$@" > "../logs/$tag.log" 2>&1
    echo "[$(date '+%F %T')] DONE $tag  $(tail -1 ../logs/$tag.log)" | tee -a "$LOG"
    if [ -f "runs/$tag/test_metrics.json" ]; then
        echo "[$(date '+%F %T')] $tag -> $(tr -d '\n ' < runs/$tag/test_metrics.json | cut -c1-240)" | tee -a "$LOG"
    fi
}

echo "=========== rerun_fix start $(date '+%F %T') ===========" | tee -a "$LOG"
run_one txf_single_s1  --arch single
run_one txf_mffgcn_s1  --arch mffgcn --afm-mode softmax
run_one txf_sigmoid_s1 --arch mffgcn --afm-mode sigmoid
run_one txf_spatial_s1 --arch mffgcn --afm-mode spatial
echo "=========== rerun_fix finished $(date '+%F %T') ===========" | tee -a "$LOG"
