#!/bin/bash
# Diagnostic round 2
#   D1 : single-encoder early-fusion U-Net (data-pipeline sanity baseline)
#   D2 : MFFGCN with a deeper encoder (6 downsampling stages, 4x4 bottleneck)
set -u
ROOT=/public2/czz/ZhangYu/project/mffgcn
cd "$ROOT/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh

COMMON="--cache ../data/cache --setup s1 --epochs 5 --decay-epoch 99 \
        --bs 32 --lr 1e-3 --workers 8 --val-batches 60"

CUDA_VISIBLE_DEVICES=0 python train.py $COMMON --gpu 0 --tag D1_single \
    --arch single > ../logs/D1.log 2>&1 &
p1=$!
sleep 4
CUDA_VISIBLE_DEVICES=1 python train.py $COMMON --gpu 1 --tag D2_deep \
    --widths 32,64,128,192,256,384 --bottleneck 512 > ../logs/D2.log 2>&1 &
p2=$!

echo "D1=$p1 D2=$p2"
wait $p1 $p2
echo "[$(date '+%F %T')] done"
for t in D1 D2; do
    echo "===== $t ====="
    grep '^\[info\] ' "../logs/$t.log" | head -3
    grep '^\[ep' "../logs/$t.log"
done
