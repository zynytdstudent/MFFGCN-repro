# 08_scripts — 运维层

Shell / Slurm 脚本，管理服务器和集群上的实验。**这些不参与 Python 逻辑**。

## 01_调度 — 自动派卡

| 文件 | 作用 |
|---|---|
| `autopilot.sh` | 自动找空闲 GPU、按队列拉下一个实验、跑完自动换下一个 |
| `gpu0_chain.sh` | 串行链式跑（单卡排队用） |
| `supervisor.sh` | 守护进程，进程挂了自动拉起 |
| `queue.sh` | 查看待跑队列 |

`autopilot.sh` 的两个历史坑（已修）：

- 曾有 `MAX_UTIL=25` 的利用率判定，但卡被别人压到 99% 时会误判"不可用"，
  导致空转 14 小时。后来删掉了利用率判定，只看空闲显存（`MIN_FREE=18000`）。
- `say` 函数原本往 stdout 打日志，被 `$(pick_gpu)` 捕获，导致 `--gpu` 变成中文句子。
  改成输出到 stderr 才对。

## 02_批量实验

| 文件 | 作用 |
|---|---|
| `run_all.sh` / `train_all.sh` | 跑全部 Setup |
| `run_experiments.sh` | 跑主实验矩阵 |
| `depth_sweep.sh` | 观测点数扫描 |
| `sensitivity.sh` | 超参敏感性 |
| `robustness.sh` | 噪声 / 观测预算鲁棒性 |
| `paper_bs15.sh` | 复现论文 batch size 设定的对照 |
| `full_run.sh` / `final_run.sh` / `post_final.sh` | 阶段性完整跑 |
| `smoke_test.sh` | 冒烟测试 |
| `baseq.sh` / `followup_afm.sh` | 基线队列 / AFM 后续实验 |

## 03_监控 — 看进度

| 文件 | 作用 |
|---|---|
| `status_all.py` | **双端状态**：服务器 + 集群一起看。参数 `server` / `cluster` / `all` |
| `status.sh` | 服务器端详细进度（读 train_log_stage*.csv） |
| `看训练进度.cmd` | Windows 双击入口（纯 ASCII + CRLF，否则 cmd 会乱码） |
| `scripts_cstatus.sh` | 集群版，上传到 `ivsn:~/work/mffgcn/scripts/cstatus.sh` |

```bash
python 08_scripts/03_监控/status_all.py all
```

## 04_修复补丁

一次性修复脚本，**历史留档，不要重跑**。包括 `fix_say.py`、`fix_wargs.py`、
`fix_cluster_jobs.py`、`fix_status_sh.py`、`patch_autopilot.py`、`rerun_*.sh`。

## 05_同步

`sync-from-server.cmd` —— 从服务器拉代码到本地。

## 06_报告

`make_trend_report.py`（纯 SVG 生成趋势对比 HTML）、`update_trend.py`
（SSH 拉快照 → 更新 `trend_data.json` → 有变化才重渲染）。

输出：`reports/07-与论文趋势一致性对比.html` + `figs/fig_trend_*.png`

## 环境提醒

服务器上 tmux 里的默认 `python` 是 Python 2.7，必须用全路径：

```bash
/public2/czz/ZhangYu/miniconda3/envs/zhangyu/bin/python3.10
```
