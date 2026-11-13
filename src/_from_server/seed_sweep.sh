#!/usr/bin/env bash
# ==========================================================================
# 多种子补跑（论文要求 5 个随机种子报 mean ± std）
#
# 用法： ./seed_sweep.sh <gpu> <tagbase> <arch> <afm-mode> <seed...>
# 例：   ./seed_sweep.sh 2 txf_mffgcn_s1  mffgcn softmax 1 2 3 4
#        ./seed_sweep.sh 3 txf_single_s1   single  softmax 1 2 3 4
#
# 与 rerun_fix.sh 的唯一区别：--seed 可变，其余超参逐项与论文对齐。
# 幂等：runs/<tag>/best.pt 存在即跳过，中断后重跑同一命令即可续。
# ==========================================================================
set -u
if [ $# -lt 5 ]; then
    echo "usage: $0 <gpu> <tagbase> <arch> <afm-mode> <seed...>" >&2
    exit 2
fi
GPU="$1"; BASE="$2"; ARCH="$3"; AFM="$4"; shift 4
SEEDS="$*"

cd "$(dirname "$0")/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1

LOG=../logs/seed_sweep.log
mkdir -p ../logs

echo "=========== seed_sweep $BASE on gpu$GPU start $(date '+%F %T') seeds=$SEEDS =====" | tee -a "$LOG"

for s in $SEEDS; do
    tag="${BASE}_seed${s}"
    if [ -f "runs/$tag/best.pt" ]; then
        echo "[$(date '+%F %T')] SKIP $tag (已存在)" | tee -a "$LOG"
        continue
    fi
    echo "[$(date '+%F %T')] START $tag on gpu$GPU (arch=$ARCH afm=$AFM seed=$s)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU python train.py --cache ../data/cache \
        --setup s1 --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
        --workers 8 --val-batches 60 --no-augment --seed "$s" \
        --widths 32,64,128,256 --bottleneck 512 --k 16 \
        --gpu "$GPU" --arch "$ARCH" --afm-mode "$AFM" \
        --tag "$tag" > "../logs/$tag.log" 2>&1
    rc=$?
    if [ -f "runs/$tag/test_metrics.json" ]; then
        echo "[$(date '+%F %T')] DONE $tag -> $(tr -d '\n ' < runs/$tag/test_metrics.json | cut -c1-200)" | tee -a "$LOG"
    else
        echo "[$(date '+%F %T')] FAIL $tag rc=$rc $(tail -2 ../logs/$tag.log | tr '\n' ' ')" | tee -a "$LOG"
    fi
done

echo "=========== seed_sweep $BASE finished $(date '+%F %T') =====" | tee -a "$LOG"
