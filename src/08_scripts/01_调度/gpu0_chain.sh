#!/usr/bin/env bash
# ==========================================================================
# gpu0 单链调度（2026-11-04 16:50 重构）
#
# 为什么有这个脚本：上一版把"等主线结束"写进 baseq.sh，里面用
# tmux kill-session 收卡，结果把 rerun_fix 的后续三个 run 一起杀了，
# 主线只留下 single 就断了。这一版不再跨会话杀进程：gpu0 上所有任务
# 都由本脚本串行跑完，谁也别去 kill 谁。
#
# 顺序（先主线，后基线——主表的核心行比次要基线重要）：
#   1) txf_mffgcn_s1    本文模型，seed 0       对照论文 MFFGCN
#   2) txf_sigmoid_s1   AFM 换 sigmoid 门控     解释"图卷积无收益"的假设
#   3) txf_spatial_s1   AFM 换空间归一化门控    同一假设的第二条证据
#   4) bl_deepae_s1     DeepAE 基线
#   5) bl_radiounet_s1  RadioUNet 基线（两阶段）
#   6) bl_pix2pix_s1    Pix2pix+ 基线
#   7) bl_rmegan_s1     RME-GAN 基线（物理阶段 + GAN 阶段）
#
# 每个 run 都以 test_metrics.json 是否存在为完成判据，可以随便 Ctrl-C
# 或重启，已完成的自动跳过。
# ==========================================================================
set -u
cd "$(dirname "$0")/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1

LOG=../logs/gpu0_chain.log
mkdir -p ../logs

GPU=0

wait_vram () {
    # 等 gpu0 空出至少 15 GB，最多等 2 小时
    local waited=0
    while [ "$waited" -lt 7200 ]; do
        local free
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
        if [ "${free:-0}" -gt 15000 ]; then return 0; fi
        echo "[$(date '+%F %T')] gpu$GPU 显存仅 ${free} MiB，等待"
        sleep 120
        waited=$((waited + 120))
    done
    echo "[$(date '+%F %T')] 等待超时，仍然继续尝试"
}

# ---- 主线：本文模型与门控变体 -------------------------------------------
run_main () {
    local tag="$1" arch="$2" afm="$3"
    if [ -f "runs/$tag/test_metrics.json" ]; then
        echo "[$(date '+%F %T')] SKIP $tag (已完成)" | tee -a "$LOG"
        return 0
    fi
    wait_vram
    echo "[$(date '+%F %T')] START $tag arch=$arch afm=$afm" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU python train.py --cache ../data/cache \
        --setup s1 --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
        --workers 8 --val-batches 60 --no-augment --seed 0 \
        --widths 32,64,128,256 --bottleneck 512 --k 16 \
        --gpu "$GPU" --arch "$arch" --afm-mode "$afm" \
        --tag "$tag" > "../logs/$tag.log" 2>&1
    local rc=$?
    if [ -f "runs/$tag/test_metrics.json" ]; then
        echo "[$(date '+%F %T')] DONE $tag -> $(tr -d '\n ' < "runs/$tag/test_metrics.json" | cut -c1-200)" | tee -a "$LOG"
    else
        echo "[$(date '+%F %T')] FAIL $tag rc=$rc :: $(tail -3 "../logs/$tag.log" | tr '\n' ' ')" | tee -a "$LOG"
    fi
}

# ---- 基线 ---------------------------------------------------------------
run_base () {
    local arch="$1" tag="$2"
    if [ -f "runs/$tag/test_metrics.json" ]; then
        echo "[$(date '+%F %T')] SKIP $tag (已完成)" | tee -a "$LOG"
        return 0
    fi
    wait_vram
    echo "[$(date '+%F %T')] START $tag arch=$arch" | tee -a "$LOG"
    if [ "$arch" = "rme_gan" ]; then
        CUDA_VISIBLE_DEVICES=$GPU python rme_gan_train.py \
            --setup s1 --tag "$tag" --phase both --phase1-epochs 20 \
            --epochs 50 --bs 32 --lr 2e-4 --workers 8 --gpu "$GPU" \
            --val-batches 60 > "../logs/$tag.log" 2>&1
    else
        CUDA_VISIBLE_DEVICES=$GPU python baseline_train.py \
            --arch "$arch" --setup s1 --tag "$tag" \
            --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
            --workers 8 --gpu "$GPU" --val-batches 60 \
            > "../logs/$tag.log" 2>&1
    fi
    local rc=$?
    if [ -f "runs/$tag/test_metrics.json" ]; then
        echo "[$(date '+%F %T')] DONE $tag -> $(tr -d '\n ' < "runs/$tag/test_metrics.json" | cut -c1-200)" | tee -a "$LOG"
    else
        echo "[$(date '+%F %T')] FAIL $tag rc=$rc :: $(tail -3 "../logs/$tag.log" | tr '\n' ' ')" | tee -a "$LOG"
    fi
}

echo "=========== gpu0_chain start $(date '+%F %T') ===========" | tee -a "$LOG"

run_main txf_mffgcn_s1  mffgcn softmax
run_main txf_sigmoid_s1 mffgcn sigmoid
run_main txf_spatial_s1 mffgcn spatial

echo "[$(date '+%F %T')] 主线三个 run 结束，转入基线" | tee -a "$LOG"

run_base deepae    bl_deepae_s1
run_base radiounet bl_radiounet_s1
run_base pix2pix   bl_pix2pix_s1
run_base rme_gan   bl_rmegan_s1

echo "=========== gpu0_chain finished $(date '+%F %T') ===========" | tee -a "$LOG"
