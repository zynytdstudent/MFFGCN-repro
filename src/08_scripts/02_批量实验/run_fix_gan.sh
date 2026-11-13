#!/bin/bash
# ==========================================================================
# run_fix_gan.sh —— 修掉"图与论文对不上"的三个确定性问题（2026-11-09）
#
# 起因：逐条审计代码后发现三处硬伤，都会直接改变图/表的形状。
#
#   ① Fig.7 的自变量搞错了
#      原 paper_fig_7.py 把横轴当成"训练集城市数"（fs_*_c50..c300，42 次训练），
#      论文原文是 "different numbers of observation samples"（每张图的观测点数）。
#      → 本脚本第 1 步用现有权重做纯推理扫描，42 次评估代替 42 次训练。
#
#   ② Pix2pix+ 用错了学习率，且没有任何衰减
#      gpu0_chain.sh 给所有基线统一传 --lr 1e-3，但 GAN 的 Adam(betas=0.5)
#      在 1e-3 下一直震荡：val NMSE ep34=0.0182 → ep49=0.0292，最终 0.0111，
#      比同结构监督 U-Net（0.0058）还差一倍。论文里 Pix2pix+ 是 0.0046。
#      → 本脚本第 2 步用 --gan-lr 2e-4 + 第 40 epoch 衰减重训。
#
#   ③ RME-GAN 阶段二没有学习率衰减
#      rme_gan_train.py 的阶段二只建了 Adam，没有 scheduler。
#      → 本脚本第 3 步补上 decay-epoch 40 重训。
#
# 用法：bash run_fix_gan.sh [gpu]
# 产物：results/fig7_obs_sweep.json、runs/bl_pix2pix_s1_fix、runs/bl_rmegan_s1_fix
# ==========================================================================
set -u
GPU="${1:-0}"
cd "$(dirname "$0")" || exit 1
source /public2/czz/ZhangYu/activate_zhangyu.sh >/dev/null 2>&1
cd src || exit 1
LOG="../logs/run_fix_gan.log"
mkdir -p ../logs ../results

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "===== 0) Fig.7 观测点扫描（纯推理，7 模型 × 6 档） ====="
python fig7_obs_sweep.py --gpu "$GPU" --setup s1 \
    --out ../results/fig7_obs_sweep.json > ../logs/fig7_obs_sweep.log 2>&1
say "fig7 sweep rc=$? -> $(ls -l ../results/fig7_obs_sweep.json 2>/dev/null | awk '{print $5}') bytes"

say "===== 1) Pix2pix+ 重训（gan-lr 2e-4 + decay@40） ====="
if [ -f runs/bl_pix2pix_s1_fix/test_metrics.json ]; then
    say "SKIP pix2pix fix (done)"
else
    python baseline_train.py --arch pix2pix --setup s1 --tag bl_pix2pix_s1_fix \
        --epochs 50 --decay-epoch 40 --gan-lr 2e-4 --bs 32 \
        --workers 8 --gpu "$GPU" --val-batches 60 \
        > ../logs/bl_pix2pix_s1_fix.log 2>&1
    say "pix2pix fix rc=$?"
fi

say "===== 2) RME-GAN 重训（阶段二补衰减） ====="
if [ -f runs/bl_rmegan_s1_fix/test_metrics.json ]; then
    say "SKIP rmegan fix (done)"
else
    python rme_gan_train.py --setup s1 --tag bl_rmegan_s1_fix --phase both \
        --phase1-epochs 20 --epochs 50 --decay-epoch 40 --bs 32 --lr 2e-4 \
        --workers 8 --gpu "$GPU" --val-batches 60 \
        > ../logs/bl_rmegan_s1_fix.log 2>&1
    say "rmegan fix rc=$?"
fi

say "===== ALL DONE ====="
for t in bl_pix2pix_s1_fix bl_rmegan_s1_fix; do
    [ -f "runs/$t/test_metrics.json" ] && \
        say "$t -> $(tr -d '\n ' < "runs/$t/test_metrics.json" | cut -c1-160)"
done
