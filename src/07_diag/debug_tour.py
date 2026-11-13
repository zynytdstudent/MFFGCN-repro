"""debug_tour.py — 跟着一条主线把 MFFGCN 的前向走完，沿途随时可以打断点。

不需要数据集、不需要显卡：全部用随机张量模拟一个 batch。
    直接跑：  python src/07_diag/debug_tour.py 2        # 2 = batch size
    VSCode：  在本文件任意一行左侧点红点，F5 选「① 调试游览 · 本地 CPU」

六站顺序（每一站都有 `# --- 断点 N ---` 标记，就在那行点红点最直观）：
    1. 造数据      看稀疏观测长什么样
    2. 建模型      看参数量分布
    3. 前向        hook 抓 AFM 的融合强度（不改模型代码也能看中间量）
    4. 拆解输出    粗估计 / 残差 / 最终结果 三者的量级关系
    5. 算损失      三项各占多少
    6. 反向        看梯度有没有正常传回 U-Net

在「调试控制台」里可以直接敲变量名，常用几个：
    obs.shape          mask.sum()         obs.nonzero().shape
    coarse.min()       res.abs().mean()   (fine - coarse).abs().max()
    len(afm_log)       afm_log[:2]
"""

import sys
import time

import torch

from losses import MFFGCNLoss
from models import MFFGCN, count_params

BS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)                      # 固定随机种子，每次跑的数字都一样

print(f"\n{'='*66}\n设备 {DEV}    batch={BS}    图像 256x256\n{'='*66}\n")

# ==========================================================================
# 断点 1 · 造数据
# 真实数据里 obs 有 99% 的像素是 0（没有传感器），这里按同样稀疏度模拟
# ==========================================================================
obs = torch.rand(BS, 1, 256, 256, device=DEV)
keep = torch.rand(BS, 1, 256, 256, device=DEV) > 0.99      # 只有 1% 的位置保留
obs = obs * keep
env = torch.rand(BS, 2, 256, 256, device=DEV)              # 通道0=建筑，通道1=发射机
mask = (obs != 0).float()                                   # 哪些位置有观测
target = torch.rand(BS, 1, 256, 256, device=DEV)

sparsity = 1 - float(mask.mean())
print(f"[1/6] 数据")
print(f"      obs{tuple(obs.shape)}  env{tuple(env.shape)}  mask{tuple(mask.shape)}")
print(f"      观测覆盖率 {1-sparsity:.1%}，空白 {sparsity:.1%}"
      f"（{int(mask.sum())} 个像素有值 / {mask.numel()}）")
print(f"      target 范围 [{target.min():.3f}, {target.max():.3f}]")

# ==========================================================================
# 断点 2 · 建模型
# ==========================================================================
model = MFFGCN().to(DEV)
total = count_params(model)
p_net = count_params(model.fusion)
p_gcn = count_params(model.refine)
print(f"\n[2/6] 模型")
print(f"      总参数 {total/1e6:.2f} M")
print(f"      ├─ 双编码器 U-Net {p_net/1e6:.2f} M ({p_net/total:.1%})")
print(f"      └─ GCN 精修      {p_gcn/1e6:.2f} M ({p_gcn/total:.1%})")

# ==========================================================================
# 断点 3 · 前向
# hook 不改模型代码，但能拿到每个 AFM 的输出 —— 看"融合有多强"
# 输出均值偏离 0 越多、标准差越大，说明这一层融合得越用力
# ==========================================================================
afm_log = []


def _grab(module, inputs, output):
    t = output[0] if isinstance(output, (tuple, list)) else output
    afm_log.append((tuple(t.shape), float(t.detach().mean()), float(t.detach().std())))


handles = [m.register_forward_hook(_grab) for m in model.fusion.afm]
# 瓶颈层是单独的 afm_b（不在 afm 这个 ModuleList 里），要单独挂
if hasattr(model.fusion, "afm_b"):
    handles.append(model.fusion.afm_b.register_forward_hook(_grab))
t0 = time.time()
# 这里故意不加 torch.no_grad()：加了之后 fine 就没有 grad_fn，
# 第 6 站 backward 会报 "does not require grad"
coarse, fine, res = model(obs, env, mask)
for h in handles:
    h.remove()
dt = time.time() - t0

print(f"\n[3/6] 前向耗时 {dt*1000:.0f} ms（{DEV.type.upper()}，未开 AMP）")
print(f"      AFM 触发了 {len(afm_log)} 次（4 个编码层 + 1 个瓶颈层）：")
for i, (shape, mu, sd) in enumerate(afm_log):
    tag = "瓶颈" if i == len(afm_log) - 1 else f"L{i+1}"
    print(f"        {tag:<4} 输出{shape}  mean={mu:+.4f}  std={sd:.4f}")

# ==========================================================================
# 断点 4 · 拆解输出
# 这是全篇最该看的一站：fine = coarse + res
#   coarse 由 U-Net 出，res 由 GCN 出，两者相加才是最终结果
# ==========================================================================
print(f"\n[4/6] 三段输出")
print(f"      coarse {tuple(coarse.shape)}  范围[{coarse.min():.3f}, {coarse.max():.3f}]")
print(f"      res    {tuple(res.shape)}  范围[{res.min():+.3f}, {res.max():+.3f}]")
print(f"      fine   {tuple(fine.shape)}  范围[{fine.min():.3f}, {fine.max():.3f}]")
print(f"      残差平均改动量 |res| = {res.abs().mean():.4f}")
print(f"      恒等式检查 max|fine-(coarse+res)| = {(fine-(coarse+res)).abs().max():.2e}"
      f"   ← 应为 0")

if DEV.type == "cuda":
    print(f"      显存峰值 {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")

# ==========================================================================
# 断点 5 · 损失
# λ1=0.1 管粗估计，λ2=1.0 管最终结果，λ3=1e-4 管 MS-SSIM
# 看哪一项真正在主导梯度
# ==========================================================================
crit = MFFGCNLoss(l1=0.1, l2=1.0, l3=1e-4)
loss, parts = crit(coarse, fine, target)
print(f"\n[5/6] 损失")
for k, v in parts.items():
    print(f"      {k:<14} {float(v):.6f}")
print(f"      加权后：0.1*mse_coarse={0.1*float(parts['mse_coarse']):.6f}  "
      f"1.0*mse_fine={float(parts['mse_fine']):.6f}  "
      f"1e-4*ms_ssim={1e-4*float(parts['ms_ssim_loss']):.6f}")

# ==========================================================================
# 断点 6 · 反向
# 梯度范数能看出 GCN 会不会"抢走" U-Net 的训练信号
# ==========================================================================
loss.backward()


def grad_norm(module):
    s = 0.0
    for p in module.parameters():
        if p.grad is not None:
            s += float(p.grad.norm()) ** 2
    return s ** 0.5


print(f"\n[6/6] 梯度范数")
print(f"      U-Net (fusion) {grad_norm(model.fusion):.4f}")
print(f"      GCN   (refine) {grad_norm(model.refine):.4f}")
print(f"\n六站走完。想改哪一站，直接在那行的断点上按 F5 重跑即可。")
