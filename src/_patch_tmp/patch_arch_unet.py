# -*- coding: utf-8 -*-
"""给 baselines.build_baseline 与 baseline_train 加上 unet 架构（2026-11-13）。

目的：把 pix2pix 的生成器单独拿出来做纯 L1 监督训练，用来隔离
"对抗损失" 与 "GAN 学习率 2e-4" 这两个因素——否则无法判断 cGAN 比
U-Net 差一倍到底是 D 的锅还是学习率的锅。
"""
import os

SRC = "/public2/czz/ZhangYu/project/mffgcn/src"

p = os.path.join(SRC, "baselines.py")
s = open(p, encoding="utf-8").read()
old = '    if k == "pix2pix":\n        return Pix2pixGen()'
new = ('    if k in ("pix2pix", "unet"):\n'
       '        # unet：pix2pix 的生成器单独做纯监督 L1 训练（消融用，2026-11-13）\n'
       '        return Pix2pixGen()')
if "unet" not in s:
    assert old in s, "pix2pix 分支没找到"
    s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)
    print("[ok] baselines.py 已加 unet")
else:
    print("[skip] baselines.py 已有 unet")

p = os.path.join(SRC, "baseline_train.py")
s = open(p, encoding="utf-8").read()
old = 'choices=["ae", "deepae", "radiounet", "pix2pix", "rme_gan"]'
new = ('choices=["ae", "deepae", "radiounet", "pix2pix", "unet",\n'
       '                             "rme_gan"]')
if old in s:
    s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)
    print("[ok] baseline_train.py 已加 unet")
else:
    print("[skip] baseline_train.py choices 已是新版")

# 注意：unet 走 else 分支（build_baseline + train_stage 纯 L1），
# forward() 里 arch != radiounet 时返回 (out, out)，evaluate 正常。
