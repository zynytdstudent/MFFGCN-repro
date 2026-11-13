#!/usr/bin/env bash
# ==========================================================================
# 协议校准实验 (calibration) —— 目的不是刷分，是把"管线错"和"模型错"分开
#
# 与之前所有 run 的唯一区别：**不带 --obs-on-signal**
#   之前 32 个 run 全部是 obs_on_signal=true（传感器只落在信号非零像素）
#   论文原文是 "sampled ... randomly distributed over the entire radio map"
#   本脚本恢复论文协议：整张 256x256 图均匀随机撒点
#
# 三个对照（全部 setup=s1，与论文 Table III Setup 1 对齐）：
#   1) calib_single   arch=single          最朴素 concat-UNet，等价论文 U-Net 行
#                     -> 论文 U-Net RMSE 0.0182；若我们接近，说明管线对了
#   2) calib_mffgcn   arch=mffgcn softmax  完整模型，论文口径
#                     -> 论文 MFFGCN RMSE 0.0150
#   3) calib_sigmoid  arch=mffgcn sigmoid  验证 AFM 通道 softmax 衰减假设
#                     -> 论文式(4)未说死 softmax 维度，通道 softmax 会把
#                        环境分支压到 1/C（C=512），coarse 因此不如 concat
# ==========================================================================
set -u
cd "$(dirname "$0")/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1

LOG=../logs/calib.log
mkdir -p ../logs

pick_gpu () {
    for i in 0 1 2 3; do
        IFS=',' read -r used total util < <( \
            nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits -i "$i" | tr -d ' ')
        free=$(( ${total:-0} - ${used:-99999} ))
        if [ "$free" -gt 20000 ] && [ "${util:-100}" -lt 30 ]; then
            echo "$i"; return 0
        fi
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
        echo "[$(date '+%F %T')] $tag -> $(tr -d '\n ' < runs/$tag/test_metrics.json | cut -c1-220)" | tee -a "$LOG"
    fi
}

echo "=========== calib start $(date '+%F %T') ===========" | tee -a "$LOG"
run_one calib_single  --arch single
run_one calib_mffgcn  --arch mffgcn --afm-mode softmax
run_one calib_sigmoid --arch mffgcn --afm-mode sigmoid
echo "=========== calib finished $(date '+%F %T') ===========" | tee -a "$LOG"
