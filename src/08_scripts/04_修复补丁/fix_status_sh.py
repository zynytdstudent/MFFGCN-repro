import io, sys

p = "status.sh"
s = io.open(p, encoding="utf-8").read()
orig = s

old_proc = '''    ppid=$(awk '{print $4}' "$p/stat" 2>/dev/null)
    tr '\\0' ' ' < "/proc/$ppid/cmdline" 2>/dev/null \\
        | grep -qE '(train|baseline_train|rme_gan_train)\\.py' && continue'''

new_proc = '''    ppid=$(awk '{print $4}' "$p/stat" 2>/dev/null)
    # 只跳过「父进程也是同一个 python 解释器」的 fork 子进程。
    # 原来判断「父进程 cmdline 里含脚本名就跳过」，会误杀用
    # bash -c "python baseline_train.py ..." 启动的主进程
    # （bl_deepae_s2 正是这样起的，之前从进度表里整条消失）。
    if [ "$(readlink -f "/proc/$ppid/exe" 2>/dev/null)" = "$(readlink -f "$p/exe" 2>/dev/null)" ]; then
        continue
    fi'''

assert old_proc in s, "进程过滤段未找到"
s = s.replace(old_proc, new_proc, 1)

old_prog = '''    else
        last=$(grep -E '^\\[ep' "../logs/$t.log" 2>/dev/null | tail -1)
        ep=$(echo "$last" | grep -oE 'ep [0-9]+' | head -1)
        vrmse=$(echo "$last" | grep -oE 'RMSE=[0-9.]+' | tail -1 | cut -d= -f2)
        spd=$(echo "$last" | grep -oE '\\([0-9]+s\\)' | tr -d '()')
        printf "  %-24s 进行  %-8s val RMSE=%-8s %s/轮\\n" "$t" "${ep:--}" "${vrmse:--}" "${spd:--}"
    fi'''

new_prog = '''    else
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
            spd=$(echo "$last" | grep -oE '\\([0-9]+s\\)' | tr -d '()')
        fi
        printf "  %-24s 进行  ep %-4s val RMSE=%-8s %s/轮\\n" "$t" "${ep:--}" "${vrmse:--}" "${spd:--}"
    fi'''

assert old_prog in s, "进度段未找到"
s = s.replace(old_prog, new_prog, 1)

assert s != orig
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched status.sh")
