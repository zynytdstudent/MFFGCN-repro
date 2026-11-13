#!/usr/bin/env bash
# ==========================================================================
# supervisor.sh —— 会话守护（cron 每 10 分钟跑一次）
#
# 为什么需要：autopilot / gpu0_chain / w2 / w3 都跑在 tmux 里，一旦会话
# 因为断连、重启、误杀而消失，整套自动化就静默停摆，而且没人会发现。
#
# 判定规则：
#   会话不存在，且它负责的 tag 里还有没生成 test_metrics.json 的
#   -> 重新拉起；全部完成 -> 什么都不做。
# 三个调度脚本本身是幂等的（靠 test_metrics.json / best.pt 判完成），
# 重复拉起只会空跑一轮然后立刻退出，不会重训已完成的任务。
#
# 注意：本脚本只"拉起"，绝不 kill 任何会话，避免重演 11-04 上午
# baseq.sh 跨会话 kill 把主线三个 run 一起杀掉的事故。
# ==========================================================================
set -u
ROOT=/public2/czz/ZhangYu/project/mffgcn
cd "$ROOT" || exit 1
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH
export TERM=screen-256color

LOG=$ROOT/logs/supervisor.log
say () { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# 这些 tag 里只要有一个还没出 test_metrics.json，就说明还有活干
pending () {
    local t
    for t in "$@"; do
        [ -f "$ROOT/src/runs/$t/test_metrics.json" ] || return 0
    done
    return 1
}

ensure () {
    local name=$1 cmd=$2
    shift 2
    tmux has-session -t "$name" 2>/dev/null && return 0
    if ! pending "$@"; then
        return 0
    fi
    say "会话 $name 不在了且仍有待办，重新拉起"
    tmux new-session -d -s "$name" "$cmd" && say "已拉起 $name"
}

SEED_TAGS=(txf_mffgcn_s1_seed1 txf_mffgcn_s1_seed2 txf_mffgcn_s1_seed3 txf_mffgcn_s1_seed4
           txf_single_s1_seed1 txf_single_s1_seed2 txf_single_s1_seed3 txf_single_s1_seed4)
SPECS='txf_mffgcn_s1:mffgcn:softmax:1 txf_mffgcn_s1:mffgcn:softmax:2 txf_mffgcn_s1:mffgcn:softmax:3 txf_mffgcn_s1:mffgcn:softmax:4 txf_single_s1:single:softmax:1 txf_single_s1:single:softmax:2 txf_single_s1:single:softmax:3 txf_single_s1:single:softmax:4'

AP_TAGS=(txf_mffgcn_s2 txf_single_s2 bl_deepae_s2 bl_radiounet_s2 txf_mffgcn_s3
         txf_single_s3 bl_pix2pix_s2 bl_rmegan_s2 bl_deepae_s3 bl_radiounet_s3)

CHAIN_TAGS=(txf_mffgcn_s1 txf_sigmoid_s1 txf_spatial_s1 bl_deepae_s1
            bl_radiounet_s1 bl_pix2pix_s1 bl_rmegan_s1)

ensure gpu0      "$ROOT/gpu0_chain.sh"          "${CHAIN_TAGS[@]}"
ensure autopilot "$ROOT/autopilot.sh"           "${AP_TAGS[@]}"
ensure w2        "$ROOT/queue.sh 2 $SPECS"      "${SEED_TAGS[@]}"
ensure w3        "$ROOT/queue.sh 3 $SPECS"      "${SEED_TAGS[@]}"

say "check: $(tmux ls 2>/dev/null | tr '\n' ' ')"
