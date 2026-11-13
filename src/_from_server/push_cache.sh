#!/bin/bash
# 在 zjut148 上执行：把 data/cache 与源码直推到 IVSN 集群
set -u

SRC=/public2/czz/ZhangYu/project/mffgcn
DST_USER=zy_221126030197
DST_HOST=192.168.66.121
DST_PORT=30907
DST_HOME=/public/home/zy_221126030197
SSH_OPTS="-p $DST_PORT -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

LOG=~/logs/rsync_cache.log
mkdir -p ~/logs
: > "$LOG"

echo "[$(date '+%F %T')] START push cache" >> "$LOG"

rsync -a --info=progress2 -e "ssh $SSH_OPTS" \
  "$SRC/data/cache/" \
  "$DST_USER@$DST_HOST:$DST_HOME/work/mffgcn/data/cache/" >> "$LOG" 2>&1
echo "RSYNC_CACHE_EXIT=$?" >> "$LOG"

echo "[$(date '+%F %T')] START push src" >> "$LOG"
rsync -a -e "ssh $SSH_OPTS" \
  "$SRC/src/" \
  "$DST_USER@$DST_HOST:$DST_HOME/work/mffgcn/src/" >> "$LOG" 2>&1
echo "RSYNC_SRC_EXIT=$?" >> "$LOG"

echo "[$(date '+%F %T')] DONE" >> "$LOG"
