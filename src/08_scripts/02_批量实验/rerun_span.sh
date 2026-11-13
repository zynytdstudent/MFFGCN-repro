#!/bin/bash
# 用 --span（等间隔铺满测试集）重跑 Fig.9 / Fig.10 的补充实验。
#
# 原因：旧结果只取了前 200 张图（约 2.5 个城市），这批图 E[t^2]=0.0448，
# 只有全集 0.0688 的 65%，把所有 NMSE 抬高了 1.19 倍。见报告 17。
#
# 用法：./rerun_span.sh <gpu>
set -u
GPU="${1:-0}"
cd "$(dirname "$0")" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1
# 脚本就放在 src/ 下，supp_eval.py 与绘图脚本都在同一层，results 在上一层
for mode in dist side; do
  old="../results/supp_${mode}_paper.json"
  [ -f "$old" ] && cp "$old" "../results/supp_${mode}_paper_n200biased.json"
  echo "===== $mode  (span, gpu $GPU)  $(date '+%F %T') ====="
  python supp_eval.py --mode "$mode" --setup s3 --n 200 --span \
      --paper-grid --gpu "$GPU" || { echo "FAILED $mode"; continue; }
done

echo "===== replot  $(date '+%F %T') ====="
python plot_fig9_10_decomp.py
echo "ALL DONE $(date '+%F %T')"
