#!/bin/bash
# 用统一口径重跑消融实验（均匀场景）。
#
# 为什么重跑：原来的 24 个消融 run 训练时把观测点只放在"有信号"的像素上
# （--obs-on-signal），而论文口径是在整张图上按密度均匀撒点。主线 txf_* 用的
# 是论文口径，所以旧消融表和新主表不能放在一起比。
#
# 这里只重跑均匀场景（s1）：8 个变体足够看出趋势，三个场景全跑要多花 3 倍
# 时间，实习期内做不完。
#
# 训练配置与主线完全一致：50 epoch、bs 32、lr 1e-3、第 40 轮衰减、无数据增强。
#
# 用法：./rerun_ablation.sh 0    （0 = 用哪张卡）

cd "$(dirname "$0")" || exit 1
GPU="${1:-0}"
LOG=logs/ablation_rerun.log
exec >> "$LOG" 2>&1
echo "=== 消融重跑开始 $(date) on gpu$GPU ==="
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1
cd src || exit 1

run_one () {
    local tag="$1"; shift
    if [ -f "runs/$tag/test_metrics.json" ]; then
        echo "[$(date +%H:%M:%S)] SKIP $tag （已完成）"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] START $tag $*"
    CUDA_VISIBLE_DEVICES="$GPU" python train.py \
        --cache ../data/cache --setup s1 \
        --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
        --workers 8 --val-batches 60 --no-augment --seed 0 \
        --tag "$tag" "$@"
    echo "[$(date +%H:%M:%S)] DONE $tag exit=$?"
}

# 轻的排前面，重的（mffgcn 全量）排后面
run_one ab2_single_s1  --arch single
run_one ab2_channel_s1 --afm-mode sigmoid
run_one ab2_concat_s1  --afm-mode concat
run_one ab2_add_s1     --afm-mode add
run_one ab2_nogcn_s1   --gcn-mode none
run_one ab2_static_s1  --gcn-mode static
run_one ab2_nores_s1   --no-residual
run_one ab2_cross_s1   --afm-mode cross

echo "=== 消融重跑结束 $(date) ==="
