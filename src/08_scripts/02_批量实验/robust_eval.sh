#!/usr/bin/env bash
# 鲁棒性实验：纯推理，不训练。
#
#   噪声档   --obs-snr 30 / 20 / 10 dB
#   观测数量 --obs-scale 0.5 / 2.0 / 4.0   （传感器数量倍率）
#
# 口径与论文一致：均匀撒点（不加 --obs-on-signal），权重用坐标 bug 修复后
# 新训练的 runs/txf_mffgcn_s{1,2,3}/best.pt。旧的 FINAL_s1 权重是 bug 期产物，
# 结果对不上，已弃用。
#
# 三个 setup 里哪个权重先出来就先评哪个，s2/s3 明天出权重后会自动补上。
# 每评完一个就释放 GPU 锁，不和训练抢卡。
set -u
ROOT=/public2/czz/ZhangYu/project/mffgcn
cd "$ROOT/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1

OUT=../eval_ROB
LOG=../logs/robust_eval.log
NEED=16000        # 评估本身只需 6-8G，不必等整卡空
MAX_ROUNDS=120          # 120 轮 × 5 分钟 ≈ 10 小时，防死循环
mkdir -p "$OUT" ../logs

say () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG" >&2; }

unlock () { rm -rf "runs/.gpulock_$1"; }

pick_gpu () {
    local g pid free
    for g in 3 2 1 0; do
        if [ "$g" = "0" ] && tmux has-session -t gpu0 2>/dev/null; then
            continue
        fi
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
        [ -z "${free:-}" ] && continue
        [ "$free" -lt "$NEED" ] && continue
        if mkdir "runs/.gpulock_$g" 2>/dev/null; then
            echo $$ > "runs/.gpulock_$g/pid"; echo "$g"; return 0
        fi
        pid=$(cat "runs/.gpulock_$g/pid" 2>/dev/null)
        if [ -z "${pid:-}" ] || ! kill -0 "$pid" 2>/dev/null; then
            rm -rf "runs/.gpulock_$g"
            if mkdir "runs/.gpulock_$g" 2>/dev/null; then
                echo $$ > "runs/.gpulock_$g/pid"; echo "$g"; return 0
            fi
        fi
    done
    return 1
}

# run_cfg <setup> <name> [evaluate.py 额外参数...]
run_cfg () {
    local s=$1 name=$2; shift 2
    local ck="runs/txf_mffgcn_${s}/best.pt"
    local out="$OUT/metrics_${name}.json"
    [ -f "$out" ] && return 0
    [ -f "$ck" ] || return 1

    local g
    while ! g=$(pick_gpu); do sleep 60; done
    say "EVAL $name on gpu$g ($*)"
    CUDA_VISIBLE_DEVICES=$g python evaluate.py --ckpt "$ck" --cache ../data/cache \
        --setup "$s" --split test --out "$OUT" --gpu "$g" --name "$name" \
        --save-figs 0 "$@" > "/tmp/eval_$name.log" 2>&1
    local rc=$?
    unlock "$g"
    if [ $rc -eq 0 ] && [ -f "$out" ]; then
        say "DONE $name"
    else
        say "FAIL $name rc=$rc :: $(tail -1 "/tmp/eval_$name.log" 2>/dev/null | cut -c1-160)"
    fi
    return 0
}

CFGS="clean snr30 snr20 snr10 bud0.5 bud2.0 bud4.0"

round=0
while [ $round -lt $MAX_ROUNDS ]; do
    round=$((round + 1))
    for s in s1 s2 s3; do
        run_cfg "$s" "rob_${s}_clean"
        for d in 30 20 10; do run_cfg "$s" "rob_${s}_snr$d" --obs-snr "$d"; done
        for b in 0.5 2.0 4.0; do run_cfg "$s" "rob_${s}_bud$b" --obs-scale "$b"; done
    done

    missing=0
    for s in s1 s2 s3; do
        for c in $CFGS; do
            [ -f "$OUT/metrics_rob_${s}_${c}.json" ] || missing=$((missing + 1))
        done
    done
    [ $missing -eq 0 ] && { say "ALL ROBUSTNESS DONE (21 个配置全齐)"; exit 0; }
    say "还缺 $missing 个配置（多半是 s2/s3 权重还没出来），5 分钟后再看"
    sleep 300
done

say "达到最大轮次退出，检查 $OUT"
