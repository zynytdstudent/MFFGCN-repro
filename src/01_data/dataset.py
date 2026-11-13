"""
RadioMapSeer 数据集：5 种不均匀采样设定 + 在线观测点采样
================================================================
这个文件决定"模型看到什么"，是复现里最容易出错、也最影响结论的一环。
论文的 4 个 Benchmark Setup 全部在这里实现。

--------------------------------------------------------------------------------
一、一条样本长什么样
--------------------------------------------------------------------------------
原始数据是三联图（同一城市、同一发射机）：
    gain     灰度的电磁增益图（DPM 模型仿真），值域 uint8 0~255
    buildings 建筑物二值图，255=建筑，0=空地
    antennas  发射机位置图（本实现不读它，见 prepare_data.py）

经过 __getitem__ 之后，一条样本变成 6 个张量：

    obs    (1,256,256)  float32  稀疏观测值，未采样处为 0
    env    (2,256,256)  float32  通道0=建筑物, 通道1=发射机指示图
    mask   (1,256,256)  float32  1=该像素被采样
    target (1,256,256)  float32  完整的地图真值，值域 [0,1]
    allowed(1,256,256) float32  1=允许布传感器的区域（Setup 2/3 会挖掉一块）
    index  ()           int64    该样本在全局 index.csv 里的行号

关系恒等式：  obs == target * mask
    （观测值就是"在采样点上取真值"，其余位置清零）

--------------------------------------------------------------------------------
二、五个 Setup 到底差在哪
--------------------------------------------------------------------------------
Setup 1  全图均匀随机布点，总数 n ∈ [10, 300]，盲区 = 无
Setup 2  先挖掉一块占 25% 面积的矩形区域（贴某一条边），只在剩下区域随机布点
Setup 3  挖掉的矩形面积占比在 [10%, 50%] 内随机 —— 比 Setup 2 更严苛、更多样
Setup 4  不挖区域，而是控制"采样密度"：像素按 1%~10% 的比例随机抽取
         （即 655 ~ 6554 个观测点，比 Setup 1 多得多）
Setup 5  把地图切成两半，一半按 1% 密度采、另一半按 10% 密度采（论文 Setup 5）

⚠ Setup 2/3 和 Setup 4/5 的语义完全不同，实现上要分开处理：
   - 2/3 控制的是"哪些区域不能采"（面积概念）→ 用 allowed 掩码
   - 4/5 控制的是"采多少个点"（密度概念）→ 用 regions + ratio
  代码里用 `regions is None` 来区分这两条路径。

--------------------------------------------------------------------------------
三、按城市划分训练/验证/测试（关键，别改）
--------------------------------------------------------------------------------
RadioMapSeer 共 701 个城市（city id 0..700），每城市 80 个发射机位置。
    训练：city[0:500]      → 40000 样本
    验证：city[500:600]    →  8000 样本
    测试：city[600:701]    →  8080 样本
划分单位是"城市"而不是"样本"：同一个城市的 80 个发射机位置会被一起分到同一侧。
这样模型在测试时必须面对完全没见过的城市布局，才是真实的泛化能力。

如果按样本随机划分（--split-mode pair），同一个城市的其他发射机位置可能已经
在训练集里，模型相当于"背过这张地图"，NMSE 会虚高一大截 —— 这是诊断模式，
不能用于正式结论。
"""

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

IMG = 256


# ==========================================================================
# 预处理
# ==========================================================================
def preprocess_gain(raw_uint8):
    """把 uint8 的原始增益图转成网络要的 [0,1] 浮点图。

    RadioMapSeer 的 PNG 是把 [-127, -47] dB 线性映射到 [0, 255] 得到的，所以
        第一步：x = raw / 255                    → [0, 1]，仍是"线性灰度"
        第二步：x[x < 0.2] = 0                   → 低于阈值的一律当"无信号"
        第三步：x = (x - 0.2) / 0.8              → 把 [0.2, 1] 拉伸回 [0, 1]
    0.2 对应的物理量是 -127 + 80×0.2 = -111 dB，即把极弱信号（噪声底）截断成 0。
    这一步很关键：如果保留 −127 dB 的区域，模型会被大片"几乎无信号"的像素主导，
    NMSE 反而好看但地图是糊的。

    用 clip(out=x) 原地写回，避免额外分配 40 万张 256x256 的内存。
    """
    x = raw_uint8.astype(np.float32) / 255.0
    x[x < 0.2] = 0.0
    x = (x - 0.2) / 0.8
    return np.clip(x, 0.0, 1.0, out=x)


def restricted_area_map(frac, rng):
    """生成 Setup 2/3 的"观测受限区域"掩码。

    做法：从地图的四条边里随机挑一条，从那条边开始切掉一个面积为 frac 的
    矩形条带，被切掉的部分不允许布传感器。

    为什么必须"贴着边"？
      如果盲区是地图正中间一个矩形，模型可以靠四周的观测从八个方向"包围"它，
      插值就能做得很好，这个 Setup 就失去区分度了。贴边则意味着盲区只有一侧
      有信息，必须靠建筑物几何去外推 —— 这才是论文想考察的能力。

    frac  : 盲区占总面积的比例，0 表示不挖
    rng   : numpy Generator，保证可复现
    返回  : (256,256) 的 bool 数组，True = 可观测
    """
    allowed = np.ones((IMG, IMG), dtype=bool)
    side = int(rng.integers(4))          # 0=上 1=下 2=左 3=右
    if frac <= 0:
        return allowed
    if side in (0, 1):                   # 上下两条边 → 切出水平条带
        h = int(round(IMG * frac))       # 高度 = 边长 × 面积占比
        h = max(1, min(IMG - 1, h))      # 至少 1 行，最多留 1 行可观测
        if side == 0:
            allowed[:h, :] = False       # 顶部盲区
        else:
            allowed[-h:, :] = False      # 底部盲区
    else:                                # 左右两条边 → 切出竖直条带
        w = int(round(IMG * frac))
        w = max(1, min(IMG - 1, w))
        if side == 2:
            allowed[:, :w] = False       # 左侧盲区
        else:
            allowed[:, -w:] = False      # 右侧盲区
    return allowed


def two_side_densities(ratio_a, ratio_b, rng):
    """Setup 5：把地图切成两半，两半用不同的采样密度（1% vs 10%）。

    论文原文是"在一个半区按 1% 比例采样，另一个半区按 10% 比例采样"。
    注意这里是"比例的百分比"，不是"面积占比"—— 两个半区面积都是 50%，
    但观测点数量相差 10 倍。

    切分方向（上下切 or 左右切）和"哪一半拿哪个密度"都随机，
    所以模型不能靠"记住了左下角总是稀疏的"来作弊。

    返回：[(半区1掩码, 该区的采样比例), (半区2掩码, 该区的采样比例)]
    """
    half = IMG // 2
    vertical = bool(rng.integers(2))     # True → 切上下两半；False → 切左右两半
    flip = bool(rng.integers(2))         # 是否交换两个半区的密度
    ra, rb = (ratio_b, ratio_a) if flip else (ratio_a, ratio_b)

    m1 = np.zeros((IMG, IMG), dtype=bool)
    m2 = np.zeros((IMG, IMG), dtype=bool)
    if vertical:
        m1[:half, :] = True              # 上半区
        m2[half:, :] = True              # 下半区
    else:
        m1[:, :half] = True              # 左半区
        m2[:, half:] = True              # 右半区
    return [(m1, ra), (m2, rb)]


# ==========================================================================
# 数据集主体
# ==========================================================================
class RadioMapSeer(Dataset):
    """一条样本 = (城市布局, 发射机位置, 一组随机抽出的观测点)。

    注意"随机抽观测点"发生在 __getitem__ 里，所以同一个 index 在不同 epoch
    取到的观测点是不一样的 —— 这本身是一种数据增强。随机数种子由
    (seed, 样本序号 i, epoch, tx_id) 共同决定，既可复现又能每个 epoch 变化。

    setup 取值见模块文档；split 取 train / val / test。
    """

    def __init__(self, cache_dir, split="train", setup="s1",
                 n_min=10, n_max=300, seed=0, img=IMG,
                 cache_gain=True, augment=False, obs_on_signal=False,
                 split_mode="city", obs_snr=None, obs_scale=1.0):
        self.cache_dir = cache_dir
        self.setup = setup.lower()
        # Setup 1/2/3 的观测预算范围：每个样本在 [n_min, n_max] 里随机取一个数
        self.n_min, self.n_max = n_min, n_max
        self.img = img
        self.seed = seed
        self.augment = augment
        self.obs_on_signal = obs_on_signal

        # --- 下面两个只给鲁棒性扫描用，正式训练保持默认 ---
        # obs_snr   : 观测噪声，单位 dB（相对观测值 RMS）。None 表示无噪声
        # obs_scale : 观测预算倍率，2.0 = 观测点数量翻倍
        self.obs_snr = obs_snr
        self.obs_scale = float(obs_scale)

        # ---- 读 index.csv：(全局行号, 城市id, 发射机id) ----
        idx_path = os.path.join(cache_dir, "index.csv")
        rows = []
        with open(idx_path) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r["idx"]), int(r["city_id"]), int(r["tx_id"])))

        if split_mode == "pair":
            # 诊断模式：按"样本"随机划分，同一城市的其他发射机会跨到训练集，
            # 造成信息泄漏。只用来验证"如果数据泄漏了指标能到多低"。
            perm = np.random.default_rng(12345).permutation(len(rows))
            n = len(perm)
            a, b = int(0.8 * n), int(0.9 * n)
            part = {"train": perm[:a], "val": perm[a:b],
                    "test": perm[b:]}.get(split, perm)
            self.rows = [rows[i] for i in part]
        else:
            # 正确做法：按城市划分，保证测试城市完全没在训练时出现过
            cities = sorted({r[1] for r in rows})
            if split == "train":
                keep = set(cities[:500])     # 40000 样本
            elif split == "val":
                keep = set(cities[500:600])  # 8000 样本
            elif split == "test":
                keep = set(cities[600:])     # 8080 样本
            else:
                keep = set(cities)           # 全量，诊断用
            self.rows = [r for r in rows if r[1] in keep]

        # ---- 内存映射加载，避免把 3.5 GB 全读进 RAM ----
        # mmap_mode="r"：操作系统按需分页，多个 worker 进程共享同一份物理内存
        self.gain = np.load(os.path.join(cache_dir, "gain_dpm.npy"), mmap_mode="r")
        self.buildings = np.load(os.path.join(cache_dir, "buildings.npy"), mmap_mode="r")
        # tx 坐标不是 mmap：只有 (N,2) 的 int16，很小，直接读更快
        self.tx = np.load(os.path.join(cache_dir, "tx_coords.npy"))
        # buildings.npy 的行按 city id 排序，需要建"城市id → 行号"的映射
        bld_city = np.load(os.path.join(cache_dir, "buildings_city.npy"))
        self.city_to_bidx = {int(c): i for i, c in enumerate(bld_city)}

        # cache_gain=True 时缓存已解码的增益图（训练时关掉，验证/评估时开启）
        self._cache = {} if cache_gain else None
        # 用于控制"每个 epoch 重新抽观测点"
        self.epoch = 0

    def set_epoch(self, ep):
        """由训练循环每个 epoch 调用，改变观测点采样的随机种子。"""
        self.epoch = int(ep)

    def __len__(self):
        return len(self.rows)

    def _load_gain(self, i):
        """按 split 内的局部下标 i 取增益图（内部会换算成全局行号）。"""
        if self._cache is None:
            return self.gain[self.rows[i][0]]
        if i not in self._cache:
            # 缓存上限 4000 条，超了就整个清空 —— 简单但够用，
            # 比 LRU 实现省事，也不会无限吃内存
            if len(self._cache) > 4000:
                self._cache.clear()
            self._cache[i] = np.array(self.gain[self.rows[i][0]])
        return self._cache[i]

    def __getitem__(self, i):
        # ==================================================================
        # 1) 定位样本 + 建立随机数发生器
        # ==================================================================
        idx, city, txid = self.rows[i]          # idx 是全局行号（0..56079）
        # 种子三元组：样本自身 + epoch + 发射机 → 每条样本、每一轮都不一样，
        # 但给定 (seed, i, epoch, txid) 又完全可复现
        rng = np.random.default_rng([self.seed + i * 7919, self.epoch * 104729 + txid])

        # ==================================================================
        # 2) 读真值
        # ==================================================================
        raw = self._load_gain(i)
        target = preprocess_gain(raw)           # (256,256) float32 ∈ [0,1]

        # ==================================================================
        # 3) 决定"哪里可以布传感器"
        # ==================================================================
        allowed = np.ones((IMG, IMG), dtype=bool)   # 默认全图可观测
        regions = None                              # None = 走"总量预算"路径
        if self.setup == "s1":
            pass                                    # 全图均匀，不做限制
        elif self.setup == "s2":
            allowed = restricted_area_map(0.25, rng)        # 固定 25% 盲区
        elif self.setup == "s3":
            frac = float(rng.uniform(0.10, 0.50))           # 盲区面积随机 10~50%
            allowed = restricted_area_map(frac, rng)
        elif self.setup == "s4":
            # 不挖区域，改为按 1%~10% 的"密度"采样
            regions = [(allowed, float(rng.uniform(0.01, 0.10)))]
        elif self.setup == "s5":
            # 两个半区分别 1% 和 10% 密度
            regions = two_side_densities(0.01, 0.10, rng)
        else:
            raise ValueError(f"unknown setup {self.setup}")

        # ==================================================================
        # 4) 抽观测点
        # ==================================================================
        obs_flat = np.zeros(IMG * IMG, dtype=np.float32)   # 观测值（先占位）
        tgt_flat = target.ravel()                          # 真值展平，便于索引

        if regions is None:
            # ---------- Setup 1/2/3：控制"总点数" ----------
            if self.obs_on_signal:
                # 只在有信号的像素里挑，避免把传感器浪费在 0 值区域
                cand = np.flatnonzero((tgt_flat > 0) & allowed.ravel())
                if cand.size < self.n_min:          # 可选点太少就退回全区域
                    cand = np.flatnonzero(allowed.ravel())
            else:
                cand = np.flatnonzero(allowed.ravel())   # 全区域都能选

            n_obs = int(rng.integers(self.n_min, self.n_max + 1))   # 10~300 随机
            if self.obs_scale != 1.0:               # 鲁棒性扫描：观测点加倍/减半
                n_obs = max(1, int(round(n_obs * self.obs_scale)))
            n_obs = min(n_obs, cand.size)           # 不能超过候选点数量
            # replace=False：同一个像素不能采两次
            obs_flat[rng.choice(cand, size=n_obs, replace=False)] = 1.0
        else:
            # ---------- Setup 4/5：控制在"每个区域"里的密度 ----------
            for area, ratio in regions:
                cand = np.flatnonzero(area.ravel())
                if self.obs_on_signal:
                    sig = np.flatnonzero(area.ravel() & (tgt_flat > 0))
                    if sig.size:
                        cand = sig
                # 点数 = 比例 × 该区域像素数
                n = int(round(ratio * int(area.sum())))
                if self.obs_scale != 1.0:
                    n = max(1, int(round(n * self.obs_scale)))
                n = max(1, min(n, cand.size))       # 至少 1 个点，最多全采
                obs_flat[rng.choice(cand, size=n, replace=False)] = 1.0

        # 这个 1/0 图就是"哪些位置被采样了"，后面会同时当掩码用
        obs_mask = obs_flat.reshape(IMG, IMG)

        # ==================================================================
        # 5) 环境图：建筑物 + 发射机位置
        # ==================================================================
        bld = self.buildings[self.city_to_bidx[city]].astype(np.float32) / 255.0
        tx_map = np.zeros((IMG, IMG), dtype=np.float32)

        # ⚠⚠ 这里曾经是一个致命的 bug，务必理解 ⚠⚠
        # tx_coords.npy 是按"全局行号 idx"索引的，不是按 split 内的局部下标。
        # 早期版本写成了 self.tx[i]，导致 val/test 的发射机位置完全错乱，
        # 而 train 因为全局行号恰好从 0 开始、与局部下标重合，看起来是"对的"。
        # 症状：训练集 loss 正常下降，验证/测试 NMSE 卡在 0.09 左右下不去。
        # 修正后 NMSE 从 0.09 直接降到 0.02 量级。
        r, c = int(self.tx[idx][0]), int(self.tx[idx][1])
        # 画一个 3x3 的方块而不是单像素：单点太稀疏，第一层 3x3 卷积很容易把它
        # 和周围全零混在一起"抹掉"；3x3 有足够响应
        tx_map[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = 1.0

        # ==================================================================
        # 6) 几何增强（翻转 / 转置）
        # ==================================================================
        # 城市布局没有"正方向"概念，所以上下翻转、左右翻转、沿对角线转置都不改变
        # 物理意义。注意：所有几何相关的张量必须用同一个变换，否则 obs 和 target
        # 就对不上了（这是实现这类增强最容易犯的错）。
        if self.augment:
            k = int(rng.integers(0, 4))          # 0=不变 1=左右翻 2=上下翻 3=转置
            if k:
                def tf(x):
                    if k == 1:
                        return np.ascontiguousarray(x[:, ::-1])
                    if k == 2:
                        return np.ascontiguousarray(x[::-1, :])
                    return np.ascontiguousarray(x.T)
                target = tf(target)
                obs_mask = tf(obs_mask)
                allowed = tf(allowed)
                bld = tf(bld)
                tx_map = tf(tx_map)

        # ==================================================================
        # 7) 生成观测值：观测点取真值，其余清零
        # ==================================================================
        obs = target * obs_mask

        # ---- 观测噪声（只在鲁棒性扫描时开启）----
        if self.obs_snr:
            vals = obs[obs_mask > 0]
            if vals.size:
                # 以观测值自身的 RMS 为基准换算噪声标准差：
                # SNR = 20·log10(rms/σ)  ⇒  σ = rms / 10^(SNR/20)
                rms = float(np.sqrt(np.mean(vals.astype(np.float64) ** 2)))
                sigma = max(rms, 1e-6) / (10.0 ** (float(self.obs_snr) / 20.0))
                noise = rng.normal(0.0, sigma, obs.shape).astype(np.float32)
                # 只加到观测点上（乘 obs_mask），并且夹回 [0,1] 的合法物理范围
                obs = np.clip(obs + noise * obs_mask, 0.0, 1.0)

        # ==================================================================
        # 8) 打包
        # ==================================================================
        # env 的两个通道：[建筑物, 发射机]
        env = np.stack([bld, tx_map], axis=0)

        return {
            # [None] 把 (256,256) 变成 (1,256,256)，补上通道维
            "obs": torch.from_numpy(obs[None]),
            "env": torch.from_numpy(env),
            "mask": torch.from_numpy(obs_mask[None]),
            "target": torch.from_numpy(target[None]),
            "allowed": torch.from_numpy(allowed.astype(np.float32)[None]),
            # 全局行号，评估时用来回溯是哪条样本
            "index": torch.tensor(idx, dtype=torch.long),
        }


def build_loaders(cache_dir, setup, batch_size=32, workers=8, seed=0,
                  train_subset=None, augment=True, obs_on_signal=False,
                  split_mode="city"):
    """一次性构建 train / val / test 三个 DataLoader。

    三个 split 的 seed 故意错开（+1000 / +2000），这样它们的观测点采样序列
    互不重叠 —— 否则验证集可能恰好和训练集用了一模一样的观测点组合。
    """
    from torch.utils.data import DataLoader

    tr = RadioMapSeer(cache_dir, "train", setup, seed=seed, cache_gain=False,
                      augment=augment, obs_on_signal=obs_on_signal,
                      split_mode=split_mode)
    # ⚠ val / test 必须和训练用完全相同的"传感器布置策略"（obs_on_signal）。
    # 如果训练时只在有信号处布点、验证时却在全图布点，那验证指标衡量的就不是
    # 同一个任务，早停选出来的 checkpoint 也不对。
    va = RadioMapSeer(cache_dir, "val", setup, seed=seed + 1000,
                      cache_gain=False, obs_on_signal=obs_on_signal,
                      split_mode=split_mode)
    te = RadioMapSeer(cache_dir, "test", setup, seed=seed + 2000,
                      cache_gain=False, obs_on_signal=obs_on_signal,
                      split_mode=split_mode)
    if train_subset:
        # 冒烟测试用：只取前 N 条
        tr.rows = tr.rows[:train_subset]

    mk = lambda ds, sh: DataLoader(
        ds, batch_size=batch_size, shuffle=sh, num_workers=workers,
        pin_memory=True,              # 锁页内存，配合 .to(device, non_blocking=True)
        drop_last=sh,                 # 训练集丢掉最后一个不满的 batch（BN 更稳）
        persistent_workers=False,     # 每 epoch 重建 worker，避免 epoch 号不同步
        # 每个 worker 预取 4 个 batch，把数据准备和 GPU 计算重叠起来
        prefetch_factor=4 if workers > 0 else None)
    return mk(tr, True), mk(va, False), mk(te, False)
