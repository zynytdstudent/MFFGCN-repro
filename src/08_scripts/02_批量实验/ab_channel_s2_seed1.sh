#!/bin/bash
# ==========================================================================
# ab_channel_s2_seed1.sh —— 复核 channel(SE) 在 Setup 2 下的"精修分支退化"
#
# 背景：txf_ab_channel_s2 的 test_metrics.json 里 nmse 与 nmse_coarse 到小数第 6 位
# 完全相同（0.006270 / 0.006271），train_log.csv 的 val_nmse 与 val_coarse_nmse
# 也一路相等。probe_residual.py 实测该权重的 |fine-coarse| 只有 0.000059
# （主模型 s2 是 0.005416），即 GCN 精修分支输出恒为 0，退化成纯粗估计。
#
# 这个 run 只换种子（seed 1），其余超参与 ab_channel.sh 完全一致，用来区分：
#   - 若 seed1 也退化  -> SE 在 25% 盲区下结构性地压制了精修（架构性质，可写进正文）
#   - 若 seed1 正常    -> 上一格是随机退化，需要报区间而不是单点
#
# 它不参与进度统计（tag 带 _seed1 后缀，progress.py 的正则不匹配），
# 也不覆盖 txf_ab_channel_s2 的官方数字。
#
# 用法：./ab_channel_s2_seed1.sh <gpu>
# ==========================================================================
set -u
cd "$(dirname "$0")" || exit 1
GPU="${1:?用法: ab_channel_s2_seed1.sh <gpu>}"
TAG="txf_ab_channel_s2_seed1"
LOG="logs/${TAG}.log"

source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1
cd src || exit 1

if [ -f "runs/${TAG}/test_metrics.json" ]; then
    echo "[$(date '+%F %T')] SKIP ${TAG} already done"
    exit 0
fi

echo "[$(date '+%F %T')] START ${TAG} on gpu${GPU}"
CUDA_VISIBLE_DEVICES="$GPU" python train.py \
    --cache ../data/cache --setup s2 \
    --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
    --workers 6 --val-batches 60 --no-augment --seed 1 \
    --widths 32,64,128,256 --bottleneck 512 --k 16 \
    --gpu "$GPU" --arch mffgcn --afm-mode se \
    --tag "$TAG" > "../${LOG}" 2>&1
rc=$?
echo "[$(date '+%F %T')] DONE ${TAG} exit=${rc}"
tail -5 "../${LOG}"
