"""
训练主入口：在 RadioMapSeer 上训练 MFFGCN（或消融变体）
================================================================
用法示例（在 src/ 目录下，或用 -m 方式运行）：

    # 正式训练：Setup 1，50 epoch，单卡
    python 03_train/train.py --cache ../data/cache --setup s1 \
        --epochs 50 --decay-epoch 40 --bs 32 --gpu 0 --tag s1_seed0

    # 冒烟测试：只跑 1200 个样本、2 epoch，用来验证代码不报错
    python 03_train/train.py --cache ../data/cache --setup s1 \
        --epochs 2 --smoke 1200 --bs 16 --tag debug_s1

    # 消融：换成 sigmoid 门控
    python 03_train/train.py ... --afm-mode sigmoid --tag ab_sigmoid_s1

本文件只负责"把数据喂进模型、算损失、反向传播、存 checkpoint"，模型结构和
损失函数分别在 02_models/models.py 与 02_models/losses.py。

--------------------------------------------------------------------------------
一次训练迭代的完整数据流
--------------------------------------------------------------------------------
  DataLoader(batch)
      │  dict: obs / env / mask / target / allowed / index
      ▼
  move() 搬到 GPU
      │  obs (B,1,256,256)  env (B,2,256,256)  mask (B,1,256,256)  target (B,1,256,256)
      ▼
  fwd() 前向
      │  arch=mffgcn → model(obs, env, mask) → (coarse, fine, residual)
      │  arch=single → model(cat([obs,env])) → (y, y, 0)
      ▼
  MFFGCNLoss
      │  L = 0.1·MSE(coarse, target) + 1.0·MSE(fine, target) + 1e-4·(1 - MS-SSIM)
      ▼
  AMP 反向传播（GradScaler 三步走）
      ▼
  优化器更新参数
      ▼
  每 epoch 末：在 val 上算 NMSE，刷新 best.pt
      ▼
  训练结束：载入 best.pt，在 test 上出最终指标 → test_metrics.json

输出目录 runs/<tag>/ 内容是：
    train_log.csv       每个 epoch 一行：loss、各分项、val NMSE/RMSE、耗时
    best.pt             val NMSE 最低的 checkpoint（含 args，便于复现结构）
    last.pt             最后一个 epoch 的 checkpoint
    test_metrics.json   最终测试集指标（后续所有对比分析都读这个文件）
"""

import argparse
import json
import os
import time

import torch
from torch.amp import GradScaler, autocast

from dataset import build_loaders
from losses import MFFGCNLoss, MetricAccumulator
from models import MFFGCN, SingleEncoderUNet, count_params


def move(b, device):
    """把 DataLoader 吐出的一个 batch 搬到指定设备。

    non_blocking=True 配合 DataLoader 的 pin_memory=True 才能生效：
    CPU 锁页内存 → GPU 的拷贝可以和计算异步重叠，省掉一部分等待。
    注意这里只搬训练真正用到的 4 个张量，"allowed" 和 "index" 只在评估/诊断
    时用得到，不必占用显存和带宽。
    """
    return (b["obs"].to(device, non_blocking=True),
            b["env"].to(device, non_blocking=True),
            b["mask"].to(device, non_blocking=True),
            b["target"].to(device, non_blocking=True))


def fwd(model, obs, env, mask, arch):
    """统一两种架构的前向签名，返回 (coarse, fine, residual)。

    单编码器 U-Net 只接受一个 3 通道输入，所以要把 obs(1) 和 env(2) 拼起来；
    它没有"粗/精"两段，所以把同一个输出同时当作 coarse 和 fine，残差补零。
    这样上层训练循环不需要写 if-else 分支，指标口径也保持一致。
    """
    if arch == "single":
        y = model(torch.cat([obs, env], dim=1))
        return y, y, torch.zeros_like(y)
    return model(obs, env, mask)


@torch.no_grad()
def evaluate(model, loader, device, amp=True, max_batches=None, arch="mffgcn"):
    """在验证/测试集上算 NMSE / RMSE，同时单独统计"粗估计"的指标。

    为什么要限制 max_batches？
      每个 epoch 都跑完整验证集（8000 样本）会让训练时间翻倍。默认只跑 60 个
      batch（约 1920 样本），足够反映趋势；训练结束后再用完整的 test 集出最终
      数字。--val-batches 可以调。

    为什么 coarse 和 fine 分开统计？
      coarse_nmse 反映"只有 U-Net"的水平，fine 是"加了 GCN"的水平，两者差值
      就是 GCN 精修段的净贡献，写报告和画图时都要用到。
    """
    model.eval()                         # 关掉 Dropout / 切到 BN 的 running stats
    acc = MetricAccumulator()            # 累计 fine 的误差
    acc_c = MetricAccumulator()          # 累计 coarse 的误差
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        obs, env, mask, target = move(b, device)
        with autocast("cuda", enabled=amp):
            coarse, fine, _ = fwd(model, obs, env, mask, arch)

        # 转回 float32 再算指标：半精度下做平方和容易溢出/丢精度，
        # 而 NMSE 是"平方和之比"，对精度非常敏感
        coarse = coarse.float().clamp(0, 1)     # 模型的输出物理上必须在 [0,1]
        fine = fine.float().clamp(0, 1)
        target = target.float()

        acc.update(coarse, fine, target)        # 记一组 (coarse, fine)
        acc_c.update(coarse, coarse, target)    # coarse 自己和自己比，只为拿它的误差

        if max_batches and i + 1 >= max_batches:
            break

    out = acc.compute()
    c = acc_c.compute()
    # 用 coarse 统计出的值覆盖，保证这两个字段口径正确
    out["coarse_nmse"] = c["nmse"]
    out["coarse_rmse"] = c["rmse"]

    model.train()                        # 恢复训练模式（BN 用 batch 统计量）
    return out


def main():
    # ======================================================================
    # 1) 命令行参数
    # ======================================================================
    ap = argparse.ArgumentParser()

    # ---- 数据 ----
    ap.add_argument("--cache", required=True,
                    help="prepare_data.py 生成的 npy cache 目录")
    ap.add_argument("--setup", default="s1",
                    choices=["s1", "s2", "s3", "s4", "s5"],
                    help="s1 均匀 / s2 25%%盲区 / s3 10-50%%盲区 / s4 1-10%%密度 / s5 双侧异密度")
    ap.add_argument("--workers", type=int, default=8,
                    help="DataLoader 进程数。集群上曾因申请核数不足导致排队，"
                         "改为 -c 4 后恢复正常")

    # ---- 训练超参 ----
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--decay-epoch", type=int, default=40,
                    help="在第几个 epoch 把学习率乘 0.1（默认 50 的 0.8 处）")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0,
                    help="影响：权重初始化、观测点采样、数据顺序")
    ap.add_argument("--no-amp", action="store_true", help="关闭混合精度（排查 NaN 时用）")

    # ---- 损失权重（论文式 10-15）----
    ap.add_argument("--l1", type=float, default=0.1, help="粗估计 MSE 的权重")
    ap.add_argument("--l2", type=float, default=1.0, help="精修结果 MSE 的权重（主损失）")
    ap.add_argument("--l3", type=float, default=1e-4, help="MS-SSIM 项的权重")

    # ---- 结构 ----
    ap.add_argument("--widths", default="32,64,128,256",
                    help="编码器各层通道宽度，逗号分隔")
    ap.add_argument("--bottleneck", type=int, default=512)
    ap.add_argument("--arch", default="mffgcn", choices=["mffgcn", "single"],
                    help="mffgcn=论文模型；single=早期融合单编码器（消融/探针）")

    # ---- 消融开关 ----
    ap.add_argument("--afm-mode", default="softmax",
                    choices=["softmax", "sigmoid", "spatial", "concat", "add",
                             "cross", "se", "channel"],
                    help="融合模块变体；se/channel = 论文表 V 的 Channel Attention")
    ap.add_argument("--gcn-mode", default="dynamic",
                    choices=["dynamic", "static", "none"],
                    help="动态语义图 / 固定网格图 / 不用图卷积")
    ap.add_argument("--no-residual", action="store_true",
                    help="去掉残差：GCN 分支直接输出整张图")
    ap.add_argument("--k", type=int, default=16,
                    help="每个图节点连几个邻居（敏感性分析扫这个）")

    # ---- 数据策略开关 ----
    ap.add_argument("--no-augment", action="store_true",
                    help="关闭翻转/转置增强")
    ap.add_argument("--obs-on-signal", action="store_true",
                    help="只在有信号的像素上布传感器（部分实验的设定）")
    ap.add_argument("--fixed-obs", action="store_true",
                    help="每个 epoch 用同一套观测点（关闭'重抽观测'带来的增强）")
    ap.add_argument("--split-mode", default="city", choices=["city", "pair"],
                    help="city=按城市划分（正确做法）；pair=按样本划分（仅诊断用，"
                         "会让测试城市在训练时见过，指标虚高）")

    # ---- 其他 ----
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tag", default="", help="输出目录名，默认 {setup}_seed{seed}")
    ap.add_argument("--smoke", type=int, default=0,
                    help=">0 时只用前 N 个训练样本、少跑验证，用来快速验证代码")
    ap.add_argument("--val-batches", type=int, default=60,
                    help="每个 epoch 用多少个 batch 做验证")
    args = ap.parse_args()

    # ======================================================================
    # 2) 环境准备
    # ======================================================================
    # 必须在 import torch 之后、创建模型之前设置，否则 CUDA 已经初始化
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # benchmark=True：输入尺寸固定（永远 256x256），让 cuDNN 自动挑最快的算法
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    tag = args.tag or f"{args.setup}_seed{args.seed}"
    run_dir = os.path.join("runs", tag)      # 相对当前工作目录
    os.makedirs(run_dir, exist_ok=True)

    # ======================================================================
    # 3) 数据
    # ======================================================================
    tr, va, te = build_loaders(args.cache, args.setup, batch_size=args.bs,
                               workers=args.workers, seed=args.seed,
                               train_subset=(args.smoke or None),
                               augment=not args.no_augment,
                               obs_on_signal=args.obs_on_signal,
                               split_mode=args.split_mode)
    print(f"[info] train={len(tr.dataset)} val={len(va.dataset)} "
          f"test={len(te.dataset)} batches/epoch={len(tr)}")

    # ======================================================================
    # 4) 模型
    # ======================================================================
    widths = tuple(int(x) for x in args.widths.split(","))
    if args.arch == "single":
        model = SingleEncoderUNet(widths=widths,
                                  bottleneck=args.bottleneck).to(device)
    else:
        model = MFFGCN(widths=widths, bottleneck=args.bottleneck,
                       afm_mode=args.afm_mode, gcn_mode=args.gcn_mode,
                       k=args.k, residual=not args.no_residual).to(device)
    # 这行日志非常有用：参数量突变说明结构改错了，不用等训练完才发现
    print(f"[info] widths={widths} bottleneck={args.bottleneck} "
          f"afm={args.afm_mode} gcn={args.gcn_mode} k={args.k} "
          f"residual={not args.no_residual} "
          f"params={count_params(model)/1e6:.2f} M "
          f"aug={not args.no_augment}")

    # ======================================================================
    # 5) 损失 / 优化器 / 学习率调度 / 混合精度
    # ======================================================================
    crit = MFFGCNLoss(args.l1, args.l2, args.l3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    # 阶梯衰减：默认第 40 epoch 后 lr 从 1e-3 降到 1e-4，
    # 让最后 10 个 epoch 做精细收敛（论文训练协议常见做法）
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[args.decay_epoch], gamma=0.1)
    # AMP 的梯度缩放器：防止 float16 下小梯度下溢成 0
    scaler = GradScaler("cuda", enabled=not args.no_amp)

    # 训练日志表头（列的顺序与后面写入严格对应）
    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w") as fh:
        fh.write("epoch,lr,train_loss,mse_coarse,mse_fine,val_nmse,val_rmse,"
                 "val_coarse_nmse,val_coarse_rmse,sec\n")

    # ======================================================================
    # 6) 训练循环
    # ======================================================================
    best = float("inf")
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()

        # 每个 epoch 换一次观测点采样 —— 这是一种数据增强：
        # 同一张地图在不同 epoch 看到不同的观测位置，逼迫模型学到"位置无关"的
        # 传播规律，而不是死记"第 37 个像素点是多少"。
        # va 固定用 epoch=0，保证验证集每次都一样，指标才有可比性。
        if hasattr(tr.dataset, "set_epoch"):
            tr.dataset.set_epoch(0 if args.fixed_obs else ep)
        if hasattr(va.dataset, "set_epoch"):
            va.dataset.set_epoch(0)

        t0 = time.time()
        run_loss = run_c = run_f = 0.0
        nb = 0

        for b in tr:
            obs, env, mask, target = move(b, device)

            # set_to_none=True 比置零更省：直接把 grad 内存释放，由下次 backward 重建
            opt.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=not args.no_amp):
                coarse, fine, _ = fwd(model, obs, env, mask, args.arch)
                loss, parts = crit(coarse, fine, target)

            # ---- AMP 标准三步，顺序不能错 ----
            scaler.scale(loss).backward()          # ① 放大 loss 再反传，避免梯度下溢
            scaler.unscale_(opt)                   # ② 把梯度还原回真实量级
            # 必须在 unscale_ 之后裁剪：否则裁的是"被放大了 65536 倍"的梯度，
            # 阈值 5.0 形同虚设
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)                       # ③ 内部检查是否有 inf/nan，正常才更新
            scaler.update()                        # ④ 动态调整缩放系数

            # 累计的是"未加权的分项"（parts 里是各 loss 的原始值），便于观察趋势
            run_loss += parts["loss"].item()
            run_c += parts["mse_coarse"].item()
            run_f += parts["mse_fine"].item()
            nb += 1
            if nb % 100 == 0:
                print(f"  ep{ep} it{nb}/{len(tr)} loss={run_loss/nb:.5f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

        # 记录本 epoch 用的学习率（在 sched.step() 之前读，才是真正生效的那个）
        lr_now = opt.param_groups[0]["lr"]
        sched.step()

        # ---- 每个 epoch 末验证 ----
        vm = evaluate(model, va, device, amp=not args.no_amp,
                      max_batches=40 if args.smoke else args.val_batches,
                      arch=args.arch)
        dt = time.time() - t0
        print(f"[ep {ep:03d}] loss={run_loss/max(nb,1):.5f} "
              f"coarse={run_c/max(nb,1):.5f} fine={run_f/max(nb,1):.5f} | "
              f"val NMSE={vm['nmse']:.5f} RMSE={vm['rmse']:.5f} "
              f"({dt:.0f}s)", flush=True)

        with open(log_path, "a") as fh:
            fh.write(f"{ep},{lr_now:.6g},{run_loss/max(nb,1):.6f},"
                     f"{run_c/max(nb,1):.6f},{run_f/max(nb,1):.6f},"
                     f"{vm['nmse']:.6f},{vm['rmse']:.6f},"
                     f"{vm['coarse_nmse']:.6f},{vm['coarse_rmse']:.6f},{dt:.1f}\n")

        # ---- 只保留验证集最好的一次 ----
        # 用 val NMSE 而不是 train loss 选模型，避免"训练集过拟合但测试更差"
        if vm["nmse"] < best:
            best = vm["nmse"]
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val": vm, "args": vars(args)},
                       os.path.join(run_dir, "best.pt"))

    # last.pt 只做备份/断点续训用，正式指标一律取 best.pt
    torch.save({"model": model.state_dict(), "epoch": args.epochs,
                "args": vars(args)}, os.path.join(run_dir, "last.pt"))

    # ======================================================================
    # 7) 最终测试
    # ======================================================================
    # 注意：测试必须用 best.pt，不能用内存里当前（最后一轮）的权重，
    # 否则就和"按 val 选模型"的协议不一致了
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    # 这里 max_batches=None，跑完整测试集（8080 样本）
    tst = evaluate(model, te, device, amp=not args.no_amp,
                   max_batches=20 if args.smoke else None, arch=args.arch)
    tst["best_epoch"] = ck["epoch"]
    tst["total_minutes"] = (time.time() - t_start) / 60
    tst["setup"] = args.setup
    print("[TEST]", json.dumps(tst, indent=2))
    # 下游的 collect_trend.py / agg_seeds.py / compare_all.py 全靠这个文件
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as fh:
        json.dump(tst, fh, indent=2)
    print(f"[done] run_dir={run_dir}")


if __name__ == "__main__":
    main()
