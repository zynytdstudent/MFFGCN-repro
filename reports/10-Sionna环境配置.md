# 10 - Sionna 环境配置（服务器 + 集群）

> 执行时间：2026-11-07 17:00 ~ 18:20
> 目标：在 `zjut148` 服务器与 `ivsn` 集群备好 NVIDIA Sionna 的光线追踪模块（Sionna RT），
> 相互独立、且不影响正在跑训练的两个 conda 环境。

## 一、两套环境并存（互不冲突）

| | 旧环境（RT-only，py3.10） | 新环境（py3.11，17:50 新增） |
|---|---|---|
| 服务器 zjut148 | `/public2/czz/ZhangYu/venvs/sionna` | `/public2/czz/ZhangYu/envs/sionna311` |
| 集群 ivsn | `~/venvs/sionna` | `~/envs/sionna311` |
| Python | 3.10.18（借用 `zhangyu` / `mffgcn` 解释器建的 venv） | 3.11.18（uv 装的独立 CPython，不复用 conda） |
| 体积 | 662M / 657M | 598M / 578M |

新环境两端版本完全一致（已对齐）：

```
sionna-rt 2.0.0 | mitsuba 3.7.1 | drjit 1.2.0 | numpy 2.2.6 | scipy 1.16.3 | matplotlib 3.10.9
```

训练环境 `zhangyu` / `mffgcn` 全程未改动（复核：torch 2.6.0+cu118、numpy 原样）。

## 二、为什么是 py3.11，又为什么只能装到 RT

这两台机器的硬约束是 **glibc**：服务器 Ubuntu 18.04（2.27）、集群 CentOS 7（2.17）。

| 想装的东西 | 卡在哪 |
|---|---|
| `sionna` 2.x（含 PHY/SYS） | 硬性要求 `torch>=2.9.1`，而 torch 2.9.1 只有 `manylinux_2_28` 轮子（需 glibc≥2.28） |
| `sionna` 1.2.2（TF 后端） | 硬性要求 `sionna-rt==1.2.2` → `mitsuba==3.8.0`，同样是 glibc≥2.28 的轮子 |
| `sionna-rt ≥ 2.1` | 硬性要求 `mitsuba==3.9.1`，同上 |
| **`sionna-rt 2.0.0`** | ✅ 依赖 `mitsuba==3.7.1`，是最后一版提供 manylinux2014(glibc 2.17) 轮子的组合 |

所以：**这两台机器上能装的最新 Sionna 就是 sionna-rt 2.0.0**；PHY/SYS 除非换系统（或等 torch 出
glibc 2.17 轮子）否则装不了。新环境用 py3.11 是为了和上游要求对齐，将来换机器可原地升级。

## 三、仍然存在的运行阻塞（与上次一致，未变）

Sionna RT 的光追后端只有两条路，都还不通：

1. **GPU（推荐）**：mitsuba 3.7.1 携带 OptiX 8，要求 NVIDIA 驱动 **≥ R535**
   - zjut148：530.30.02 ｜ ivsn 计算节点：515.43.04（node1/node6 实测）
2. **CPU 兜底**：Dr.Jit 要求 `libLLVM.so` **≥ 11**
   - zjut148 系统只有 LLVM 6.0/10.0，ivsn 只有 7；无 sudo（`sudo` 需密码）
   - GitHub 预编译 LLVM 包在这条链路上 700MB 要 20~30 分钟且易断，暂未落地

**最小代价方案仍是把 zjut148 驱动升到 R535+**（A100 支持，向下兼容 CUDA 11.8，不影响现有训练），
升完执行 `sionna_smoke_test.py` 即可确认。

## 四、日常用法

```bash
# 服务器 zjut148（新环境）
/public2/czz/ZhangYu/envs/sionna311/bin/python your_rt_script.py

# 集群 ivsn（务必走 Slurm，登录节点无 GPU/LLVM）
sbatch -p NVIDIAA100-PCIE-40GB ~/work/mffgcn/scripts/sionna_job.sbatch your_rt_script.py
```

自检脚本两端各放一份（项目内 `src/09_sionna/sionna_smoke_test.py`）：

```bash
python sionna_smoke_test.py          # 优先 GPU
python sionna_smoke_test.py --cpu    # 强制 CPU
# 通过打印 SIONNA_RT_SELFTEST_PASS；否则同时列出两个后端各自的失败原因
```

集群模板 `sionna_job.sbatch` 里的 `VENV` 默认指向 `~/venvs/sionna`，
改用新环境时把它改成 `$HOME/envs/sionna311` 即可。

## 五、踩过的坑（下次少走弯路）

- conda 源在这两台机器上不可达（DNS 只回 IPv6，`repo.anaconda.com`、清华与 bfsu 的 conda 镜像全超时），
  因此 py3.11 是用 `uv` 装的 CPython：`UV_PYTHON_INSTALL_MIRROR` 需指向 `gh-proxy.com`
  （`ghfast.top` 对该路径返回 403，但下载 LLVM 时反而可用）。
- 集群 glibc 2.17 会让 pip/uv 去**源码编译** contourpy，必失败；装包要加 `--only-binary :all:` 并锁版本。
- `pip index versions` 会按平台过滤，看起来"最新只到 3.7.1"其实是 glibc 挡住了更高版本，别误判成镜像没同步。
- torch 的 cu118 轮子连同 CUDA 依赖约 3.5 GB，这条链路 30 分钟都下不完，故新环境暂不含 torch
  （RT 本身不需要 torch；将来需要时补 `--index-url https://download.pytorch.org/whl/cu118`）。
