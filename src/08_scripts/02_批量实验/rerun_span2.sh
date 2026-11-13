#!/bin/bash
# 用 --span 重跑 Fig.14 / Fig.15 / Fig.16 三组补充实验。
# 与 Fig.9/10 同一个病因：旧结果只取前 200 张图（约 2.5 个城市）。
# 见报告 17。
set -u
GPU="${1:-0}"
cd "$(dirname "$0")" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1

for mode in noise degrade missing; do
  old="../results/supp_${mode}_paper.json"
  [ -f "$old" ] && cp "$old" "../results/supp_${mode}_paper_n200biased.json"
  echo "===== $mode (span, gpu $GPU)  $(date '+%F %T') ====="
  python supp_eval.py --mode "$mode" --setup s3 --n 200 --span \
      --paper-grid --gpu "$GPU" || echo "FAILED $mode"
done

echo "===== replot  $(date '+%F %T') ====="
python plot_fig_compare.py noise degrade missing
echo "ALL DONE $(date '+%F %T')"
