#!/bin/bash
# Wait for a GPU to genuinely free up, then train the 5-layer MFFGCN used by
# paper Fig.13 (propagation steps 1-5).  Runs alongside autopilot5, so it
# takes the same .gpulock_<gpu> mutex instead of forcing a slot.
cd /public2/czz/ZhangYu/project/mffgcn/src || exit 1
set +u
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1

TAG=txf_gcn5_s1
NEED=18000

[ -f "runs/$TAG/test_metrics.json" ] && { echo "already done"; exit 0; }

while true; do
    for g in 0 1 2 3; do
        free=$(nvidia-smi --id=$g --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
        [ -n "$free" ] || continue
        [ "$free" -ge "$NEED" ] || continue
        sleep 60
        free2=$(nvidia-smi --id=$g --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
        [ -n "$free2" ] && [ "$free2" -ge "$NEED" ] || continue
        if mkdir "runs/.gpulock_$g" 2>/dev/null; then
            echo $$ > "runs/.gpulock_$g/pid"
            echo "[$(date '+%F %T')] gcn5 starts on gpu$g (free ${free2}M)"
            CUDA_VISIBLE_DEVICES=$g python train.py --cache ../data/cache \
                --setup s1 --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
                --workers 8 --val-batches 60 --no-augment --seed 0 \
                --widths 32,64,128,256 --bottleneck 512 --k 16 \
                --gpu "$g" --arch mffgcn --afm-mode softmax --gcn-layers 5 \
                --tag "$TAG" > "../logs/$TAG.log" 2>&1
            rc=$?
            rm -rf "runs/.gpulock_$g"
            echo "[$(date '+%F %T')] gcn5 finished rc=$rc"
            exit 0
        fi
    done
    sleep 120
done
