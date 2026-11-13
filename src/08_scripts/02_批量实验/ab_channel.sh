#!/bin/bash
# ==========================================================================
# ab_channel.sh —— 补论文表 V 的 "Channel Attention" 三格
#
# 为什么之前一直缺这三格：论文表 V 的 Channel Attention 行是
# Squeeze-and-Excitation，而我们的 models.py 只有
# softmax / sigmoid / spatial / concat / add / cross 六种模式，
# 从来没有 SE 实现。旧脚本里的 ab_channel_s1 其实写的是
# `--afm-mode sigmoid`（见 run_experiments.sh:36），既不是 SE，又是
# 坐标 bug 修复前的口径。
#
# 现在 models.py 增加了 mode="se"（别名 "channel"），这个脚本就是补上的
# 调度入口。超参逐项照抄 autopilot4.sh 的 mffgcn 分支，保证与
# txf_ab_add_* / txf_ab_concat_* 完全同口径。
#
# 用法：./ab_channel.sh <s1|s2|s3> <gpu>
# ==========================================================================
set -u
cd "$(dirname "$0")" || exit 1
SETUP="${1:?用法: ab_channel.sh <s1|s2|s3> <gpu>}"
GPU="${2:?用法: ab_channel.sh <s1|s2|s3> <gpu>}"
TAG="txf_ab_channel_${SETUP}"
LOG="logs/${TAG}.log"

source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1
cd src || exit 1

# 幂等：已有 test_metrics.json 就直接跳过（与 autopilot 的完成判据一致）
if [ -f "runs/${TAG}/test_metrics.json" ]; then
    echo "[$(date '+%F %T')] SKIP ${TAG} already done"
    exit 0
fi

echo "[$(date '+%F %T')] START ${TAG} on gpu${GPU}"
CUDA_VISIBLE_DEVICES="$GPU" python train.py \
    --cache ../data/cache --setup "$SETUP" \
    --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
    --workers 8 --val-batches 60 --no-augment --seed 0 \
    --widths 32,64,128,256 --bottleneck 512 --k 16 \
    --gpu "$GPU" --arch mffgcn --afm-mode se \
    --tag "$TAG" > "../${LOG}" 2>&1
rc=$?
echo "[$(date '+%F %T')] DONE ${TAG} exit=${rc}"
tail -5 "../${LOG}"
