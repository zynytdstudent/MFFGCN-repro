#!/usr/bin/env bash
# ==========================================================================
# depth_sweep.sh —— 盲区深度剖面：三个采样场景 + U-Net 对照
#
# 为什么单独跑：diag_depth.py 只需要推理，不需要训练，因此可以和训练并行。
# 每个 step 都以 results/depth_*.json 是否存在为完成判据，可随时重跑。
# 输出：results/depth_{s1,s2,s3,single_s1}.json + logs/depth_*.log
# ==========================================================================
set -u
cd "$(dirname "$0")/src" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1
mkdir -p ../logs ../results

GPU=${GPU:-0}

run_one () {   # $1=run 目录名  $2=setup  $3=输出名
    local run="$1" setup="$2" name="$3"
    local out="../results/depth_${name}.json"
    local log="../logs/depth_${name}.log"
    if [ -f "$out" ]; then
        echo "[skip] $name 已完成"; return 0
    fi
    if [ ! -f "runs/$run/best.pt" ]; then
        echo "[miss] runs/$run/best.pt 不存在，跳过"; return 0
    fi
    echo "[run ] $name  ($run, $setup) -> $out"
    python diag_depth.py --ckpt "runs/$run/best.pt" --setup "$setup" \
        --gpu "$GPU" --batch-size 8 --workers 4 --name "$name" --out "$out" \
        > "$log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[fail] $name 退出码 $rc，见 $log"; tail -5 "$log"; return 1
    fi
    tail -9 "$log" | sed 's/^/       /'
}

run_one txf_mffgcn_s1    s1 s1
run_one txf_mffgcn_s2    s2 s2
run_one txf_mffgcn_s3    s3 s3
run_one txf_single_s1    s1 single_s1
echo "[done] 深度剖面全部结束"
