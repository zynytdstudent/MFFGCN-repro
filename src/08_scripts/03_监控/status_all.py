#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
status_all.py —— 在 Windows 本机一条命令看两边的训练进度

用法：
    python status_all.py            两边都看
    python status_all.py server     只看服务器 zjut148
    python status_all.py cluster    只看集群 ivsn

原理：两边并不互联，是这台 Windows 分别 SSH 过去各跑一个只读脚本
（服务器 ./status.sh，集群 scripts/cstatus.sh）。
"""
import subprocess
import sys
import os

# Windows 控制台默认 GBK，中文和框线会乱码
if os.name == "nt":
    try:
        subprocess.run("chcp 65001", shell=True, capture_output=True)
    except Exception:
        pass

TARGETS = {
    "server": (
        "zjut148",
        "服务器 zjut148（192.168.66.148:30905，4×A100 共享）",
        "cd /public2/czz/ZhangYu/project/mffgcn && ./status.sh",
    ),
    "cluster": (
        "ivsn",
        "集群 ivsn（192.168.66.121:30907，Slurm）",
        "cd /public/home/zy_221126030197/work/mffgcn && bash scripts/cstatus.sh",
    ),
}


def run(host, remote_cmd, timeout=90):
    """SSH 到远端执行只读脚本，返回 (成功?, 输出)"""
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        host,
        remote_cmd,
    ]
    try:
        p = subprocess.run(
            cmd, capture_output=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return False, "连接超时（%d 秒）。检查网络或 VPN。" % timeout
    except FileNotFoundError:
        return False, "找不到 ssh 命令。Windows 需启用 OpenSSH 客户端。"

    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    # OpenSSH 升级提示、已知主机警告等噪音过滤掉
    noise = (
        "openssh.com/pq.html",
        "post-quantum",
        "store now, decrypt later",
        "Warning: Permanently added",
        "Pseudo-terminal",
    )
    lines = [ln for ln in (out.splitlines() + err.splitlines())
             if ln.strip() and not any(n in ln for n in noise)]
    ok = p.returncode == 0 and any("时间:" in ln for ln in lines)
    return ok, "\n".join(lines)


def main():
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if which not in ("all", "server", "cluster"):
        print("用法： python status_all.py [all|server|cluster]")
        return 2

    keys = ["server", "cluster"] if which == "all" else [which]
    failed = []
    for k in keys:
        host, title, rcmd = TARGETS[k]
        print("=" * 66)
        print(title)
        print("=" * 66)
        ok, text = run(host, rcmd)
        print(text if text else "（无输出）")
        if not ok:
            failed.append(k)
        print()

    if failed:
        print("!! 以下目标没拿到状态：" + "、".join(failed))
        print("   先单独试： ssh %s echo ok" % failed[0])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
