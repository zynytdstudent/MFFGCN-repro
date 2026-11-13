"""给 autopilot 加「按模型大小给门槛」+ 重排队列。2026-11-05"""
import io

P = "/public2/czz/ZhangYu/project/mffgcn/autopilot.sh"
s = io.open(P, encoding="utf-8").read()

# ---- 1. 默认门槛：按大模型（12.9G）定，小模型在队列里单独写 ----
old = "MIN_FREE=18000      # 判定空闲的最小剩余显存（MiB）"
new = "MIN_FREE=15000      # 默认门槛（大模型实测 12.9G + 2G 余量）；小模型在队列里单独写"
assert old in s, "MIN_FREE 行没找到"
s = s.replace(old, new, 1)

# ---- 2. 队列表：加第 5 段（所需显存） ----
i0 = s.index("TASKS=(")
i1 = s.index(")", s.index("\n", i0))
old_tasks = s[i0:i1 + 1]
new_tasks = '''TASKS=(
    # 格式： tag:arch:afm:setup:所需显存(MiB)
    # 实测：mffgcn / deepae / radiounet ~12.9G，single(U-Net) ~5.9G
    "txf_mffgcn_s2:mffgcn:softmax:s2:15000"
    "txf_mffgcn_s3:mffgcn:softmax:s3:15000"
    "bl_deepae_s2:deepae:softmax:s2:15000"
    "bl_radiounet_s2:radiounet:softmax:s2:15000"
    "bl_pix2pix_s2:pix2pix:softmax:s2:15000"
    "bl_rmegan_s2:rme_gan:softmax:s2:15000"
    "bl_deepae_s3:deepae:softmax:s3:15000"
    "bl_radiounet_s3:radiounet:softmax:s3:15000"
)'''
s = s.replace(old_tasks, new_tasks, 1)

# ---- 3. 空闲判定带上所需显存 ----
old = '''gpu_snapshot_ok () {
    local gpu=$1 free util'''
new = '''gpu_snapshot_ok () {
    local gpu=$1 need=${2:-$MIN_FREE} free util'''
assert old in s
s = s.replace(old, new, 1)

old = '''    [ "${free:-0}" -lt "$MIN_FREE" ] && return 1'''
new = '''    [ "${free:-0}" -lt "$need" ] && return 1'''
assert old in s
s = s.replace(old, new, 1)

old = '''gpu_is_free () {
    local gpu=$1
    gpu_snapshot_ok "$gpu" || return 1
    sleep 40
    gpu_snapshot_ok "$gpu" || return 1
    return 0
}'''
new = '''gpu_is_free () {
    local gpu=$1 need=${2:-$MIN_FREE}
    gpu_snapshot_ok "$gpu" "$need" || return 1
    sleep 40
    gpu_snapshot_ok "$gpu" "$need" || return 1
    return 0
}'''
assert old in s
s = s.replace(old, new, 1)

old = '''pick_gpu () {
    local gpu'''
new = '''pick_gpu () {
    local gpu need=${1:-$MIN_FREE}'''
assert old in s
s = s.replace(old, new, 1)

old = '''        if gpu_is_free "$gpu"; then'''
new = '''        if gpu_is_free "$gpu" "$need"; then'''
assert old in s
s = s.replace(old, new, 1)

# ---- 4. 加 peek_task（只挑不占）与 task_need ----
old = '''next_task () {'''
new = '''peek_task () {
    local spec tag
    for spec in "${TASKS[@]}"; do
        tag=${spec%%:*}
        [ -f "runs/$tag/test_metrics.json" ] && continue
        [ "$(fail_count "$tag")" -ge "$MAX_FAIL" ] && continue
        [ -d "runs/.claim_$tag" ] && continue
        echo "$spec"; return 0
    done
    return 1
}

task_need () {
    local spec=$1 need
    need=$(awk -F: '{print $5}' <<< "$spec")
    [ -z "${need:-}" ] && need=$MIN_FREE
    echo "$need"
}

next_task () {'''
assert old in s
s = s.replace(old, new, 1)

# ---- 5. worker：先看任务需要多少显存，再挑卡 ----
old = '''worker () {
    local wid=$1 gpu spec
    say "worker#$wid 启动"
    while :; do
        if ! gpu=$(pick_gpu); then
            sleep "$POLL"; continue
        fi
        if ! spec=$(next_task); then
            rm -rf "runs/.gpulock_$gpu"
            say "worker#$wid 没有待办任务了，退出"
            return 0
        fi
        say "worker#$wid 领到 $spec -> gpu$gpu"
        run_task "$spec" "$gpu"
        rm -rf "runs/.gpulock_$gpu"
    done
}'''
new = '''worker () {
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
}'''
assert old in s
s = s.replace(old, new, 1)

io.open(P, "w", encoding="utf-8").write(s)
print("patched ok")
