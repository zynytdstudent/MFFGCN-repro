#!/usr/bin/env bash
# ==========================================================================
# status.sh —— 一眼看清全部训练状态
# 用法： ./status.sh
# ==========================================================================
set -u
ROOT=/public2/czz/ZhangYu/project/mffgcn
cd "$ROOT/src" || exit 1

line () { printf '%s\n' "--------------------------------------------------------------"; }

echo "时间: $(date '+%F %T')"
line
echo "【显卡】"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits \
    | awk -F', ' '{printf "  gpu%s  %6d / %d MiB  利用率 %s%%\n", $1, $2, $3, $4}'

echo
echo "【我方训练进程】"
# DataLoader 会 fork 出多个子进程，cmdline 与环境跟主进程一模一样。
# 主进程的父进程是 bash 脚本，子进程的父进程才是 python，据此只留主进程。
rows=$(
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    [ "$(stat -c %U "$p" 2>/dev/null)" = "${USER:-czz}" ] || continue
    tr '\0' '\n' < "$p/cmdline" 2>/dev/null | head -3 \
        | grep -qE '(^|/)(train|baseline_train|rme_gan_train)\.py' || continue
    ppid=$(awk '{print $4}' "$p/stat" 2>/dev/null)
    # 只跳过「父进程也是同一个 python 解释器」的 fork 子进程。
    # 原来判断「父进程 cmdline 里含脚本名就跳过」，会误杀用
    # bash -c "python baseline_train.py ..." 启动的主进程
    # （bl_deepae_s2 正是这样起的，之前从进度表里整条消失）。
    if [ "$(readlink -f "/proc/$ppid/exe" 2>/dev/null)" = "$(readlink -f "$p/exe" 2>/dev/null)" ]; then
        continue
    fi
    dev=$(tr '\0' '\n' < "$p/environ" 2>/dev/null | grep -m1 '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2-)
    tag=$(tr '\0' '\n' < "$p/cmdline" 2>/dev/null | grep -A1 '^--tag$' | tail -1)
    et=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
    echo "${dev:-?}|${tag:-?}|${et:-?}"
  done | sort -u
)
if [ -z "$rows" ]; then echo "  （无）"; else
  echo "$rows" | awk -F'|' '{printf "  gpu%-2s %-26s 已跑 %s\n", $1, $2, $3}'
fi

line
echo "【各 run 进度】"
for d in runs/txf_*/ runs/bl_*/; do
    [ -d "$d" ] || continue
    t=$(basename "$d")
    [ -f "$d/best.pt" ] || continue
    if [ -f "$d/test_metrics.json" ]; then
        m=$(tr -d '\n ' < "$d/test_metrics.json")
        nmse=$(echo "$m" | grep -oE '"nmse":[0-9.eE+-]+' | head -1 | cut -d: -f2)
        rmse=$(echo "$m" | grep -oE '"rmse":[0-9.eE+-]+' | head -1 | cut -d: -f2)
        awk -v t="$t" -v a="${nmse:-0}" -v b="${rmse:-0}" \
            'BEGIN{printf "  %-24s 完成  NMSE=%-9.5f RMSE=%.5f\n", t, a, b}'
    else
        # 优先读 CSV，列位置明确。两类任务的列不同：
        #   主模型 train_log.csv      : ep,lr,train_loss,mse_c,mse_f,val_nmse,val_rmse,...
        #   基线 train_log_stage1/2.csv: ep,lr,train_loss,val_nmse,val_rmse,sec
        lf=""
        for c in "$d/train_log.csv" "$d/train_log_stage2.csv" "$d/train_log_stage1.csv"; do
            [ -s "$c" ] && lf="$c" && break
        done
        ep=""; vrmse=""; spd=""
        if [ -n "$lf" ] && [ "$(wc -l < "$lf")" -gt 1 ]; then
            case "$lf" in
                *stage*) read -r ep _ _ _ vrmse spd <<< "$(tail -1 "$lf" | tr ',' ' ')" ;;
                *)       read -r ep _ _ _ _ _ vrmse _ _ spd <<< "$(tail -1 "$lf" | tr ',' ' ')" ;;
            esac
        else
            last=$(grep -E 'ep ?[0-9]+' "../logs/$t.log" 2>/dev/null | tail -1)
            ep=$(echo "$last" | grep -oE 'ep ?[0-9]+' | head -1 | grep -oE '[0-9]+')
            vrmse=$(echo "$last" | grep -oE 'RMSE=[0-9.]+' | tail -1 | cut -d= -f2)
            spd=$(echo "$last" | grep -oE '\([0-9]+s\)' | tr -d '()')
        fi
        spd=${spd%s}
        printf "  %-24s 进行  ep %-4s val RMSE=%-8s %ss/轮\n" "$t" "${ep:--}" "${vrmse:--}" "${spd:--}"
    fi
done

line
echo "【调度会话】"
tmux ls 2>/dev/null | sed 's/^/  /' || echo "  （无）"

echo
echo "【自动派发】最近 3 条"
tail -3 ../logs/autopilot.log 2>/dev/null | sed 's/^/  /'
printf "  待办 %s 项，已完成 %s 个 run\n" \
    "$(grep -cE '^    "(txf|bl)_' ../autopilot.sh 2>/dev/null)" \
    "$(ls -d runs/*/test_metrics.json 2>/dev/null | wc -l)"
line
