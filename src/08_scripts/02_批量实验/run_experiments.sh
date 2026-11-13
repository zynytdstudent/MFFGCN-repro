#!/bin/bash
# Completes the experiments the paper reports but the first pass skipped:
#   * Setup 4 / 5  (extreme sparsity: 1-10% restricted, two sides 1% vs 10%)
#   * Table V ablation, 8 variants x 3 setups
#
# 26 training runs. Four workers, two per GPU (0 and 2). Each run trains with
# exactly the flags of the FINAL runs (50 epochs, decay at 40, no augmentation,
# sensors on signal) so the numbers stay comparable, then evaluates on the test
# split into ../eval_ALL/metrics_<name>.json.
#
# Launch:  tmux new -d -s all_experiments 'bash run_experiments.sh'
# Monitor: tmux attach -t all_experiments   /   tail -f ../logs/queue.log
set -u
ROOT=/public2/czz/ZhangYu/project/mffgcn
cd "$ROOT/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh
mkdir -p ../logs ../eval_ALL/reports

# pull the three finished setups into the same folder so the final report can
# print all five columns side by side
for s in s1 s2 s3; do
    if [ ! -f "../eval_ALL/metrics_$s.json" ] && [ -f "../eval_FINAL/metrics_$s.json" ]; then
        cp "../eval_FINAL/metrics_$s.json" "../eval_ALL/"
    fi
done

COMMON="--cache ../data/cache --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
        --workers 8 --val-batches 60 --no-augment --obs-on-signal"

JOBS=$(cat <<'EOF'
FINAL_s4|s4|s4|
FINAL_s5|s5|s5|
ab_single_s1|s1|single_s1|--arch single
ab_add_s1|s1|add_s1|--afm-mode add
ab_concat_s1|s1|concat_s1|--afm-mode concat
ab_channel_s1|s1|channel_s1|--afm-mode sigmoid
ab_cross_s1|s1|cross_s1|--afm-mode cross
ab_nogcn_s1|s1|nogcn_s1|--gcn-mode none
ab_static_s1|s1|static_s1|--gcn-mode static
ab_nores_s1|s1|nores_s1|--no-residual
ab_single_s2|s2|single_s2|--arch single
ab_add_s2|s2|add_s2|--afm-mode add
ab_concat_s2|s2|concat_s2|--afm-mode concat
ab_channel_s2|s2|channel_s2|--afm-mode sigmoid
ab_cross_s2|s2|cross_s2|--afm-mode cross
ab_nogcn_s2|s2|nogcn_s2|--gcn-mode none
ab_static_s2|s2|static_s2|--gcn-mode static
ab_nores_s2|s2|nores_s2|--no-residual
ab_single_s3|s3|single_s3|--arch single
ab_add_s3|s3|add_s3|--afm-mode add
ab_concat_s3|s3|concat_s3|--afm-mode concat
ab_channel_s3|s3|channel_s3|--afm-mode sigmoid
ab_cross_s3|s3|cross_s3|--afm-mode cross
ab_nogcn_s3|s3|nogcn_s3|--gcn-mode none
ab_static_s3|s3|static_s3|--gcn-mode static
ab_nores_s3|s3|nores_s3|--no-residual
EOF
)

TOTAL=$(echo "$JOBS" | grep -c .)
echo "[$(date '+%F %T')] queue built: $TOTAL runs" | tee -a ../logs/queue.log

run_one() {
    local gpu=$1 tag=$2 setup=$3 name=$4; shift 4
    local extra="$*"
    if [ -f "runs/$tag/best.pt" ]; then
        echo "[$(date '+%F %T')] gpu$gpu SKIP $tag (checkpoint exists)" | tee -a ../logs/queue.log
    else
        echo "[$(date '+%F %T')] gpu$gpu START $tag  setup=$setup  $extra" | tee -a ../logs/queue.log
        CUDA_VISIBLE_DEVICES=$gpu python train.py $COMMON --setup "$setup" \
            --gpu "$gpu" --tag "$tag" $extra > "../logs/$tag.log" 2>&1
        if [ ! -f "runs/$tag/best.pt" ]; then
            echo "[$(date '+%F %T')] gpu$gpu FAIL  $tag (no checkpoint, see logs/$tag.log)" | tee -a ../logs/queue.log
            return
        fi
        echo "[$(date '+%F %T')] gpu$gpu DONE  $tag $(tail -1 ../logs/$tag.log)" | tee -a ../logs/queue.log
    fi
    if [ ! -f "../eval_ALL/metrics_$name.json" ]; then
        python evaluate.py --ckpt "runs/$tag/best.pt" --cache ../data/cache \
            --setup "$setup" --split test --out ../eval_ALL --gpu "$gpu" \
            --obs-on-signal --name "$name" --save-figs 0 \
            > "../logs/eval_$name.log" 2>&1
        echo "[$(date '+%F %T')] gpu$gpu EVAL  $name" | tee -a ../logs/queue.log
    fi
}

worker() {
    local idx=$1 gpu=$2
    echo "$JOBS" | awk -v k="$idx" 'NR % 3 == k' | while IFS='|' read -r tag setup name extra; do
        [ -z "$tag" ] && continue
        run_one "$gpu" "$tag" "$setup" "$name" $extra
    done
    echo "[$(date '+%F %T')] gpu$gpu worker finished" | tee -a ../logs/queue.log
}

# one process per GPU: sharing a card nearly doubled the epoch time and made
# the queue slower overall
worker 0 0 &
worker 1 1 &
worker 2 2 &
wait

echo "[$(date '+%F %T')] ALL TRAINING FINISHED -> building report" | tee -a ../logs/queue.log
python compare_all.py --dir ../eval_ALL --out ../eval_ALL/reports \
    > ../logs/compare_all.log 2>&1
echo "[$(date '+%F %T')] report: ../eval_ALL/reports/comparison_all.md" | tee -a ../logs/queue.log
