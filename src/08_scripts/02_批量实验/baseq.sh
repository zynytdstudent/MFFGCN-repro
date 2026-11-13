#!/bin/bash
# 主线 single(seed 0) 跑完后接管 gpu0，按优先级依次训练表 III 的深度基线。
#
# 为什么要等：gpu0 现在跑的是主线 single，直接抢会拖慢主表那张最关键的图。
# 等它出 [TEST] 行（训练结束）再收卡。
#
# 顺序：AE -> DeepAE -> RadioUNet(两阶段)。两个 cGAN 单独排，成本最高。
cd "$(dirname "$0")" || exit 1
LOG=logs/baseline_queue.log
: > "$LOG"
exec >> "$LOG" 2>&1

echo "[$(date +%H:%M:%S)] guard: 等待主线 single 结束"
# 等 test_metrics.json 落地，而不是只等 [TEST] 打印——避免在写文件前一刻
# 把 rerunfix 杀掉，丢掉主线的测试指标
while [ ! -f src/runs/txf_single_s1/test_metrics.json ]; do
    sleep 30
done

echo "[$(date +%H:%M:%S)] 主线 single 已完成，释放 gpu0"
tmux kill-session -t rerunfix 2>/dev/null
sleep 15

source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1
cd src || exit 1

run_one () {
    local arch="$1" tag="$2" setup="$3"
    if [ -f "runs/$tag/best.pt" ] && [ -f "runs/$tag/test_metrics.json" ]; then
        echo "[$(date +%H:%M:%S)] SKIP $tag (已完成)"
        return 0
    fi
    echo "[$(date +%H:%M:%S)] START $arch $setup -> $tag"
    CUDA_VISIBLE_DEVICES=0 python baseline_train.py \
        --arch "$arch" --setup "$setup" --tag "$tag" \
        --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
        --workers 8 --gpu 0
    echo "[$(date +%H:%M:%S)] DONE $arch exit=$?"
}

run_one ae        bl_ae_s1        s1
run_one deepae    bl_deepae_s1    s1
run_one radiounet bl_radiounet_s1 s1

echo "[$(date +%H:%M:%S)] 表 III 深度基线全部完成"
