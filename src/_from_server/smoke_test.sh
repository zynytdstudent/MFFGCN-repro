#!/bin/bash
# One-epoch smoke for every new code path: Setup 4/5 and the eight ablation
# switches. Fails loudly instead of silently starting a 30 hour queue.
set -u
cd /public2/czz/ZhangYu/project/mffgcn/src || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null
GPU=${1:-2}
OUT=/tmp/smoke_mffgcn
mkdir -p "$OUT"
FAIL=0

smoke() {
    local setup=$1 name=$2; shift 2
    local extra="$*"
    echo "=== $name (setup=$setup) $extra"
    CUDA_VISIBLE_DEVICES=$GPU python train.py --cache ../data/cache --epochs 1 \
        --decay-epoch 40 --bs 8 --lr 1e-3 --workers 2 --val-batches 2 \
        --no-augment --obs-on-signal --setup "$setup" --gpu "$GPU" \
        --tag "smoke_$name" --smoke 320 $extra > "$OUT/$name.train.log" 2>&1
    if [ $? -ne 0 ] || [ ! -f "runs/smoke_$name/best.pt" ]; then
        echo "    TRAIN FAIL"; tail -5 "$OUT/$name.train.log"; FAIL=$((FAIL+1)); return
    fi
    python evaluate.py --ckpt "runs/smoke_$name/best.pt" --cache ../data/cache \
        --setup "$setup" --split test --out "$OUT" --gpu "$GPU" \
        --obs-on-signal --name "smoke_$name" --max-samples 8 --save-figs 1 \
        > "$OUT/$name.eval.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "    EVAL FAIL"; tail -5 "$OUT/$name.eval.log"; FAIL=$((FAIL+1)); return
    fi
    echo "    ok  $(grep -m1 '"all_nmse"' "$OUT/metrics_smoke_$name.json")"
}

smoke s4 s4
smoke s5 s5
smoke s1 single --arch single
smoke s1 add --afm-mode add
smoke s1 concat --afm-mode concat
smoke s1 channel --afm-mode sigmoid
smoke s1 cross --afm-mode cross
smoke s1 nogcn --gcn-mode none
smoke s1 static --gcn-mode static
smoke s1 nores --no-residual

echo "=== failures: $FAIL"
