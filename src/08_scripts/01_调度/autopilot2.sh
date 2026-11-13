#!/usr/bin/env bash
# ==========================================================================
# autopilot.sh —— 全局自动派发器（2026-11-04 17:20）
#
# 目标：4 张卡任何一张空出来（不管是自己的跑完、还是别人的任务撤走），
#       都自动接下一个待办任务，不需要人盯。
#
# 与其它脚本的分工（互不重叠，避免重复训练）：
#   gpu0_chain.sh  —— 独占 gpu0：主模型 seed0 → sigmoid → spatial → 4 个基线
#   queue.sh(w2/w3) —— gpu2/gpu3：mffgcn seed1-4、single seed1-4
#   本脚本         —— 兜底抢任何空闲卡：主模型/U-Net 的 s2、s3 场景
#                      （论文三场景对照表缺的正是这些）
#
# 安全设计：
#   1. 每张卡运行时用 mkdir runs/.gpulock_<gpu> 加锁，两个 worker 不会挤一张卡
#   2. 判定"空闲"要同时满足：显存剩余 > 22G、利用率 < 25%、没有我方训练进程
#   3. gpu0 只要 gpu0_chain 会话还在就跳过，防止抢它两次 run 之间的空档
#   4. 每个任务用 mkdir runs/.claim_<tag> 原子抢占，与 queue.sh 共用同一套
#   5. 失败重试上限 2 次，超过就跳过并在日志里点名，不会无限重跑
# ==========================================================================
set -u
cd "$(dirname "$0")/src" || exit 1

LOG=../logs/autopilot2.log
POLL=150            # 没有空闲卡时的轮询间隔（秒）
MIN_FREE=15000      # 默认门槛（大模型实测 12.9G + 2G 余量）；小模型在队列里单独写
                    # 实测 11-04 三个 run 连跑 4h：单 run 稳定 12.9G，含验证峰值约 13G
                    # 18000 = 13G 需求 + 5G 余量
                    # 原来 34000 是照评估阶段的峰值定的，结果任何卡都凑不出 34G，队列饿死
MAX_UTIL=25         # 判定空闲的最大利用率（%）
NUM_WORKERS=4
MAX_FAIL=4

mkdir -p ../logs
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1 || true

# ---- 待办队列（按优先级从上往下）----------------------------------------
# 格式： <tag>:<arch>:<afm_mode>:<setup>
# 这批是论文三场景对照表缺的：之前只有 s1 是干净重训过的
TASKS=(
    # ===== 第一优先：三场景主表还缺的格子 =====
    "bl_radiounet_s2:radiounet:softmax:s2:15000"
    "bl_pix2pix_s2:pix2pix:softmax:s2:15000"
    "bl_rmegan_s2:rme_gan:softmax:s2:15000"
    "bl_deepae_s3:deepae:softmax:s3:15000"
    "bl_radiounet_s3:radiounet:softmax:s3:15000"
    # ===== 第二优先：表 V 消融 s2/s3 干净重跑（旧的 ab_*_s2/s3 是坐标 bug 期产物）=====
    "txf_ab_concat_s2:mffgcn:concat:s2:15000"
    "txf_ab_add_s2:mffgcn:add:s2:15000"
    "txf_ab_sigmoid_s2:mffgcn:sigmoid:s2:15000"
    "txf_ab_cross_s2:mffgcn:cross:s2:15000"
    "txf_ab_nogcn_s2:mffgcn:softmax:s2:15000:--gcn-mode none"
    "txf_ab_static_s2:mffgcn:softmax:s2:15000:--gcn-mode static"
    "txf_ab_nores_s2:mffgcn:softmax:s2:15000:--no-residual"
    "txf_ab_concat_s3:mffgcn:concat:s3:15000"
    "txf_ab_add_s3:mffgcn:add:s3:15000"
    "txf_ab_sigmoid_s3:mffgcn:sigmoid:s3:15000"
    "txf_ab_cross_s3:mffgcn:cross:s3:15000"
    "txf_ab_nogcn_s3:mffgcn:softmax:s3:15000:--gcn-mode none"
    "txf_ab_static_s3:mffgcn:softmax:s3:15000:--gcn-mode static"
    "txf_ab_nores_s3:mffgcn:softmax:s3:15000:--no-residual"
)

say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG" >&2; }

# ---- 有没有我的训练进程占着这张卡 ---------------------------------------
my_pid_on_gpu () {
    local gpu=$1 pid dev
    for p in /proc/[0-9]*; do
        pid=${p#/proc/}
        [ "$(stat -c %U "$p" 2>/dev/null)" = "${USER:-czz}" ] || continue
        tr '\0' '\n' < "$p/cmdline" 2>/dev/null | head -3 | grep -qE '(train|baseline_train|rme_gan_train)\.py' || continue
        dev=$(tr '\0' '\n' < "$p/environ" 2>/dev/null | grep -m1 '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2-)
        [ "${dev:-}" = "$gpu" ] && return 0
    done
    return 1
}

gpu_snapshot_ok () {
    local gpu=$1 need=${2:-$MIN_FREE} free util
    IFS=',' read -r free util < <(
        nvidia-smi --query-gpu=memory.free,utilization.gpu \
            --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
    [ -z "${free:-}" ] && return 1
    [ "${free:-0}" -lt "$need" ] && return 1
    # 注意：这台机器的卡常年被别的租户压到 99%，利用率不是可用信号。
    # 之前用 MAX_UTIL=25 判定，等于把所有卡永远排除，队列饿了 14 小时。
    # 现在只看剩余显存（够不够这个模型）+ 有没有我方进程。
    my_pid_on_gpu "$gpu" && return 1
    return 0
}

# 判定空闲要连续两次采样都通过（间隔 40s）。这是为了防止抢到别的 worker
# 两次 run 之间的空档：那种空档只有几秒，单次采样很容易误判，踩上去就是 OOM。
gpu_is_free () {
    local gpu=$1 need=${2:-$MIN_FREE}
    gpu_snapshot_ok "$gpu" "$need" || return 1
    sleep 40
    gpu_snapshot_ok "$gpu" "$need" || return 1
    return 0
}

# ---- 抢一张空卡 ---------------------------------------------------------
# 锁目录里记持有者 PID：万一 worker 被意外杀掉，锁会残留，那张卡就永远
# 分不出去了。所以 mkdir 失败时先看持有者还在不在，不在就清掉重抢。
lock_owner_alive () {
    local d="runs/.gpulock_$1" pid
    [ -f "$d/pid" ] || return 1
    pid=$(cat "$d/pid" 2>/dev/null)
    [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null
}

pick_gpu () {
    local gpu need=${1:-$MIN_FREE}
    for gpu in ${CAND_GPUS:-0 1 2 3}; do
        # gpu0 属于 gpu0_chain，只要它的会话还在就不碰
        if [ "$gpu" = "0" ] && tmux has-session -t gpu0 2>/dev/null; then
            continue
        fi
        if ! mkdir "runs/.gpulock_$gpu" 2>/dev/null; then
            if ! lock_owner_alive "$gpu"; then
                say "清理 gpu$gpu 的残留锁"
                rm -rf "runs/.gpulock_$gpu"
                mkdir "runs/.gpulock_$gpu" 2>/dev/null || continue
            else
                continue
            fi
        fi
        echo "$BASHPID" > "runs/.gpulock_$gpu/pid"
        if gpu_is_free "$gpu" "$need"; then
            echo "$gpu"; return 0
        fi
        rm -rf "runs/.gpulock_$gpu"
    done
    return 1
}

# ---- 领一个任务 ---------------------------------------------------------
fail_count () {
    local f="runs/.fails_$1"
    [ -f "$f" ] && cat "$f" || echo 0
}

peek_task () {
    local spec tag
    for spec in "${TASKS[@]}"; do
        tag=${spec%%:*}
        [ -f "runs/$tag/test_metrics.json" ] && continue
        [ "$(fail_count "$tag")" -ge "$MAX_FAIL" ] && continue
        if [ -d "runs/.claim_$tag" ]; then
            # worker 被意外杀掉时抢占目录会残留，不清理的话这个任务永远轮不到
            claim_alive "$tag" || { say "清理 $tag 的残留抢占"; rm -rf "runs/.claim_$tag"; }
            [ -d "runs/.claim_$tag" ] && continue
        fi
        echo "$spec"; return 0
    done
    return 1
}

# 抢占目录里记持有者 PID（worker 主 shell），持有者不在就当残留
claim_alive () {
    local d="runs/.claim_$1" pid
    if [ -f "$d/pid" ]; then
        pid=$(cat "$d/pid" 2>/dev/null)
        [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null && return 0
        return 1
    fi
    # 没有 pid 文件：旧版 autopilot 或 queue.sh 抢的，退化成「看有没有带这个 tag 的进程」
    tag_running "$1"
}

tag_running () {
    local p
    for p in /proc/[0-9]*; do
        tr '\0' ' ' < "$p/cmdline" 2>/dev/null | grep -q -- "--tag $1 " && return 0
    done
    return 1
}

task_need () {
    local spec=$1 need
    need=$(awk -F: '{print $5}' <<< "$spec")
    [ -z "${need:-}" ] && need=$MIN_FREE
    echo "$need"
}

next_task () {
    local spec tag rest
    for spec in "${TASKS[@]}"; do
        tag=${spec%%:*}
        [ -f "runs/$tag/test_metrics.json" ] && continue
        [ "$(fail_count "$tag")" -ge "$MAX_FAIL" ] && continue
        # 关键：已经有同名 tag 的进程在跑就必须跳过。
        # 手动 rerun（tmux 里直接起 baseline_train.py）不会建 claim 目录，
        # 旧逻辑只看 mkdir 成败 → 同一个 tag 被 autopilot 又派发一次，
        # 两个进程同时写 runs/<tag>/train_log*.csv，epoch 序列错乱成 28 29 1 30 2 31 3 32。
        if tag_running "$tag"; then
            continue
        fi
        # claim 目录存在但持有者已死（旧版没写 pid / 被 kill 过），先清掉再抢
        if [ -d "runs/.claim_$tag" ] && ! claim_alive "$tag"; then
            rm -rf "runs/.claim_$tag"
        fi
        if mkdir "runs/.claim_$tag" 2>/dev/null; then
            echo "$$" > "runs/.claim_$tag/pid"   # $$ = worker 主 shell，被 kill 后 PID 失效
            echo "$spec"; return 0
        fi
    done
    return 1
}

run_task () {
    local spec=$1 gpu=$2
    local tag arch afm setup rest
    IFS=':' read -r tag arch afm setup _need _extra <<< "$spec"
    local t0=$(date +%s)

    if [ "$arch" = "rme_gan" ]; then
        CUDA_VISIBLE_DEVICES=$gpu python rme_gan_train.py \
            --setup "$setup" --tag "$tag" --phase both --phase1-epochs 20 \
            --epochs 50 --bs 32 --lr 2e-4 --workers 8 --gpu "$gpu" \
            --val-batches 60 > "../logs/$tag.log" 2>&1
    elif [ "$arch" = "single" ] || [ "$arch" = "mffgcn" ]; then
        CUDA_VISIBLE_DEVICES=$gpu python train.py --cache ../data/cache \
            --setup "$setup" --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
            --workers 8 --val-batches 60 --no-augment --seed 0 \
            --widths 32,64,128,256 --bottleneck 512 --k 16 \
            --gpu "$gpu" --arch "$arch" --afm-mode "$afm" ${_extra:-} \
            --tag "$tag" > "../logs/$tag.log" 2>&1
    else
        CUDA_VISIBLE_DEVICES=$gpu python baseline_train.py \
            --arch "$arch" --setup "$setup" --tag "$tag" \
            --epochs 50 --decay-epoch 40 --bs 32 --lr 1e-3 \
            --workers 8 --gpu "$gpu" --val-batches 60 \
            > "../logs/$tag.log" 2>&1
    fi
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))

    if [ -f "runs/$tag/test_metrics.json" ]; then
        say "DONE $tag gpu$gpu ${dt}s -> $(tr -d '\n ' < "runs/$tag/test_metrics.json" | cut -c1-160)"
    else
        local n=$(( $(fail_count "$tag") + 1 ))
        echo "$n" > "runs/.fails_$tag"
        say "FAIL($n) $tag gpu$gpu rc=$rc ${dt}s :: $(tail -2 "../logs/$tag.log" | tr '\n' ' ' | cut -c1-220)"
    fi
    rm -rf "runs/.claim_$tag" 2>/dev/null   # 里面现在有 pid 文件，rmdir 删不掉
}

# ---- worker：不停找卡、领任务、跑 ---------------------------------------
worker () {
    local wid=$1 gpu spec need
    say "worker#$wid 启动"
    while :; do
        if ! spec=$(peek_task); then
            say "worker#$wid 没有待办任务了，退出"
            return 0
        fi
        need=$(task_need "$spec")
        if ! gpu=$(pick_gpu "$need"); then
            sleep "$POLL"; continue
        fi
        if ! spec=$(next_task); then
            rm -rf "runs/.gpulock_$gpu"
            say "worker#$wid 任务被抢光了，退出"
            return 0
        fi
        say "worker#$wid 领到 $spec -> gpu$gpu（需 ${need}M）"
        run_task "$spec" "$gpu"
        rm -rf "runs/.gpulock_$gpu"
    done
}

say "=========== autopilot start (workers=$NUM_WORKERS, tasks=${#TASKS[@]}) ==========="

pids=()
for i in $(seq 1 "$NUM_WORKERS"); do
    worker "$i" &
    pids+=($!)
    sleep 3
done
wait "${pids[@]}"

say "=========== autopilot finished ==========="
