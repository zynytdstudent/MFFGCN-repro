#!/usr/bin/env bash
# ==========================================================================
# cstatus.sh —— 集群（ivsn / Slurm）训练状态一眼看
# 用法： bash scripts/cstatus.sh
#
# 对应服务器那台的 status.sh。集群上任务由 Slurm 调度，
# 所以除了「跑到第几轮」还要看「为什么排不上」。
#
# 2026-11-05 改进：
#   1) 「有空卡」不再只看 GPU，还要看 CPU 核够不够——之前 node7 空着 2 张卡却
#      只剩 1 个核，作业要 4 个，于是 PENDING，脚本却写「有空卡」，误导。
#   2) 排队的作业直接给出原因（缺核 / 缺卡 / 优先级）。
#   3) 历史失败留下的空目录合并成一行，不再刷屏。
# ==========================================================================
set -u
ROOT=/public/home/zy_221126030197/work/mffgcn
cd "$ROOT" || exit 1

MIN_CORES=4      # 我的作业通常申请 4 个核，判定「能不能开跑」按这个算

line () { printf '%s\n' "--------------------------------------------------------------"; }

echo "时间: $(date '+%F %T')  （集群 ivsn）"
line
echo "【我的作业】"
# 多取一列：申请的核数，后面判断排队原因要用
sj=$(squeue -u "$USER" -h -o "%.8i %.14j %.9T %.12R %.10M %.4C" 2>/dev/null)
if [ -z "$sj" ]; then
    echo "  （队列里没有作业）"
else
    printf "  %-8s %-14s %-9s %-12s %-8s %s\n" "作业号" "名称" "状态" "所在节点" "已跑" "核数"
    echo "$sj" | awk '{printf "  %-8s %-14s %-9s %-12s %-8s %s\n", $1, $2, $3, $4, $5, $6}'
fi

line
echo "【各 run 进度】"
found=0
empties=""
for d in src/runs/*/; do
    [ -d "$d" ] || continue
    t=$(basename "$d")
    case "$t" in txf_*|bl_*) ;; *) continue ;; esac
    [ -f "$d/train_log.csv" ] || continue
    # 只有表头、没有数据行 = 早前失败留下的空目录，先攒起来最后合并显示
    if [ "$(wc -l < "$d/train_log.csv" 2>/dev/null || echo 0)" -lt 2 ]; then
        empties="$empties $t"
        continue
    fi
    found=1
    if [ -f "$d/test_metrics.json" ]; then
        rmse=$(grep -oE '"rmse": *[0-9.eE+-]+' "$d/test_metrics.json" | head -1 | grep -oE '[0-9.eE+-]+$')
        awk -v t="$t" -v r="${rmse:-0}" 'BEGIN{printf "  %-24s 完成  RMSE=%.5f\n", t, r}'
    else
        last=$(tail -1 "$d/train_log.csv" 2>/dev/null)
        ep=$(echo "$last" | cut -d, -f1)
        spd=$(echo "$last" | awk -F, '{print $NF}')
        val=$(echo "$last" | awk -F, '{print $7}')
        [ -z "${ep:-}" ] && continue
        awk -v t="$t" -v e="$ep" -v v="${val:-0}" -v s="${spd:-0}" \
            'BEGIN{printf "  %-24s 进行  ep %2d/%d  val RMSE=%.5f  %ds/轮  剩 %.0f 分钟\n", t, e, 50, v, s, (50-e)*s/60}'
    fi
done
[ "$found" = "0" ] && echo "  （src/runs 下还没有 txf_/bl_ 训练记录）"
if [ -n "$empties" ]; then
    n_e=$(echo "$empties" | wc -w)
    short=$(echo "$empties" | tr ' ' '\n' | sed 's/txf_single_s1_seed/seed/' | tr '\n' ' ' | sed 's/ $//')
    echo "  另：$n_e 个历史失败空目录（$short）"
fi

line
echo "【节点 GPU / CPU 占用】"
printf "  %-7s %-8s %-12s %-12s %s\n" "节点" "状态" "空闲GPU" "空闲CPU" "能否开跑(需${MIN_CORES}核)"
any_can=0        # 有没有节点「卡够 + 核够」，能立刻开跑
any_gpu_free=0   # 有没有节点还有空闲卡（哪怕核不够）
for n in node1 node2 node3 node4 node5 node6 node7 node8; do
    info=$(scontrol show node "$n" 2>/dev/null)
    [ -z "$info" ] && continue
    st=$(echo "$info" | grep -oE 'State=[A-Z+]*' | head -1 | cut -d= -f2)
    tot=$(echo "$info" | grep -oE 'Gres=gpu:[^:]*:[0-9]+' | head -1 | grep -oE '[0-9]+$')
    alloc=$(echo "$info" | grep -oE 'AllocTRES=[^ ]*gres/gpu=[0-9]+' | head -1 | grep -oE '[0-9]+$')
    cpuT=$(echo "$info" | grep -oE 'CPUTot=[0-9]+' | head -1 | cut -d= -f2)
    cpuA=$(echo "$info" | grep -oE 'CPUAlloc=[0-9]+' | head -1 | cut -d= -f2)
    cpuI=$(( ${cpuT:-0} - ${cpuA:-0} ))
    free=$(( ${tot:-0} - ${alloc:-0} ))

    if [ "$st" = "DOWN" ] || [ "$st" = "DOWN*" ]; then
        verdict="节点宕机"
        free=0
    elif [ "${free:-0}" -le 0 ]; then
        verdict="卡已占满"
    elif [ "$cpuI" -lt "$MIN_CORES" ]; then
        verdict="有卡但核不够（只空 $cpuI 个）"
        any_gpu_free=1
    else
        verdict="可以开跑"
        any_can=1
        any_gpu_free=1
    fi
    printf "  %-7s %-8s %-12s %-12s %s\n" "$n" "${st:-?}" "${free}/${tot:-?}" "$cpuI" "$verdict"
done

line
echo "【排队诊断】"
pend=$(echo "${sj:-}" | awk '$3=="PENDING"')
if [ -z "$pend" ]; then
    echo "  没有排队中的作业。"
else
    echo "$pend" | while read -r jid jname jstate jreason jtime jcores; do
        [ -z "${jid:-}" ] && continue
        need=${jcores:-$MIN_CORES}
        if [ "$any_can" = "1" ]; then
            echo "  $jid $jname：已有节点卡和核都够（需 $need 核），等前面作业让位即可"
        elif [ "$any_gpu_free" = "1" ]; then
            echo "  $jid $jname：有空闲卡，但 CPU 核不够（需 $need 核，最好的节点空不到这么多）→ 等正在跑的作业收尾"
        else
            echo "  $jid $jname：所有节点 GPU 都已占满 → 等别人释放，或排队优先级靠后"
        fi
    done
fi

line
echo "【最近日志】"
for f in $(ls -t logs/slurm-*.out 2>/dev/null | head -2); do
    echo "  $f"
    tail -2 "$f" 2>/dev/null | sed 's/^/    /'
done
line
echo "提示：完整日志 logs/slurm-<名称>-<作业号>.out ；错误看同名 .err"
echo "      单体进度 src/runs/<tag>/train_log.csv（最后一列是秒/轮）"
