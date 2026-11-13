# -*- coding: utf-8 -*-
"""给 baseline_train.py 的 GAN 循环补上两点稳定化设置（TTUR + 标签平滑）。

为什么
------
现有 cGAN 的对抗平衡是坏的：bl_pix2pix_s1_fix 训练末段

    g=2.85, d=0.24

其中 g = BCE_adv + 100 * L1。按 L1≈0.024 反推，BCE_adv≈0.45，而生成器
完全骗过判别器时这一项应接近 ln2 = 0.693。也就是说 **D 判假图的准确率已经
很高**，对抗梯度不是"温和地引导细节"，而是在跟 L1 目标打架。

后果：同一套权重下
    pix2pix+  (生成器=U-Net)   NMSE 0.00897
    我们表里的 U-Net 基线       NMSE 0.00367     ← 2.4 倍
而论文 Table III（Setup 3）里 Pix2pix+ 0.0059 vs U-Net 0.0081，cGAN 是
**略优于** U-Net 的。方向完全相反，所以问题在训练配方而不是结构。

改动（都用开关控制，默认值保持旧行为以便复现已有的跑）
    1. --d-lr-ratio  判别器学习率 = 生成器学习率 × 该比例（默认 1.0）。
       设 0.5 即 TTUR（two time-scale update rule），让 D 追得慢一点。
    2. --label-smooth  真样本标签从 1.0 降到该值（默认 1.0，即不平滑）。
       0.9 是 GAN 里的常用设置，能压低 D 的过度自信。

用法：
    python patch_gan_stability.py          # 打补丁
    python baseline_train.py --arch pix2pix --d-lr-ratio 0.5 --label-smooth 0.9 \
        --tag bl_pix2pix_s1_ttur
"""
import io

P = "baseline_train.py"

OLD_LR = '''    lr = args.gan_lr if args.gan_lr > 0 else args.lr
    opt_g = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))'''

NEW_LR = '''    lr = args.gan_lr if args.gan_lr > 0 else args.lr
    # TTUR：D 的学习率可以单独压低（--d-lr-ratio）。判别器过强时 BCE_adv 会
    # 掉到 0.45 以下（< ln2），对抗梯度开始与 L1 目标冲突，生成器退化。
    d_lr = lr * getattr(args, "d_lr_ratio", 1.0)
    opt_g = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=d_lr, betas=(0.5, 0.999))
    ls = getattr(args, "label_smooth", 1.0)'''

assert OLD_LR in io.open(P, encoding="utf-8").read()

OLD_BCE = '''                loss_d = 0.5 * (bce(d_real, torch.ones_like(d_real))
                                + bce(d_fake, torch.zeros_like(d_fake)))'''
NEW_BCE = '''                loss_d = 0.5 * (bce(d_real, torch.full_like(d_real, ls))
                                + bce(d_fake, torch.zeros_like(d_fake)))'''

OLD_G = '''                loss_g = bce(d_fake, torch.ones_like(d_fake)) \\
                    + args.lambda_l1 * l1(fake, target)'''
NEW_G = '''                loss_g = bce(d_fake, torch.full_like(d_fake, ls)) \\
                    + args.lambda_l1 * l1(fake, target)'''

OLD_EP = '''        print(f"[ep {ep:03d}] val NMSE={vm['nmse']:.5f} RMSE={vm['rmse']:.5f} "
              f"lr={opt_g.param_groups[0]['lr']:.2g} "
              f"({time.time()-t0:.0f}s)", flush=True)'''
NEW_EP = '''        print(f"[ep {ep:03d}] val NMSE={vm['nmse']:.5f} RMSE={vm['rmse']:.5f} "
              f"lr={opt_g.param_groups[0]['lr']:.2g} "
              f"dlr={opt_d.param_groups[0]['lr']:.2g} "
              f"({time.time()-t0:.0f}s)", flush=True)'''

OLD_ARG = '''    ap.add_argument("--smoke", type=int, default=0)'''
NEW_ARG = '''    ap.add_argument("--d-lr-ratio", type=float, default=1.0,
                    help="判别器学习率 = GAN 学习率 x 该值（TTUR）。"
                         "D 过强时设 0.5 或 0.25")
    ap.add_argument("--label-smooth", type=float, default=1.0,
                    help="真样本标签值，0.9 为常用标签平滑；1.0 表示不平滑")
    ap.add_argument("--smoke", type=int, default=0)'''


def main():
    s = io.open(P, encoding="utf-8").read()
    n = 0
    for old, new, tag in ((OLD_LR, NEW_LR, "lr"),
                          (OLD_BCE, NEW_BCE, "loss_d"),
                          (OLD_G, NEW_G, "loss_g"),
                          (OLD_EP, NEW_EP, "log"),
                          (OLD_ARG, NEW_ARG, "args")):
        if old in s:
            s = s.replace(old, new)
            n += 1
        else:
            print("[warn] 没找到:", tag)
    io.open(P, "w", encoding="utf-8").write(s)
    print("[patched] %d 处" % n)


if __name__ == "__main__":
    main()
