#!/bin/bash
# Download RadioMapSeer.zip from ModelScope mirror with resume support
set -u
DATA_DIR=/public2/czz/ZhangYu/project/mffgcn/data
mkdir -p "$DATA_DIR"
cd "$DATA_DIR" || exit 1

URL="https://www.modelscope.cn/api/v1/datasets/Hzzzz001/radiomapseer/repo?Revision=master&FilePath=RadioMapSeer.zip"
TARGET=RadioMapSeer.zip
EXPECT=3139949016

echo "[$(date '+%F %T')] start download"
for attempt in 1 2 3 4 5 6 7 8; do
    echo "[$(date '+%F %T')] attempt $attempt"
    curl -L -C - --retry 5 --retry-delay 5 --connect-timeout 20 \
         --speed-time 120 --speed-limit 1024 \
         -o "$TARGET" "$URL"
    rc=$?
    sz=$(stat -c %s "$TARGET" 2>/dev/null || echo 0)
    echo "[$(date '+%F %T')] rc=$rc size=$sz"
    if [ "$sz" -ge "$EXPECT" ]; then
        break
    fi
    sleep 5
done

sz=$(stat -c %s "$TARGET" 2>/dev/null || echo 0)
echo "[$(date '+%F %T')] final size=$sz expected=$EXPECT"
if [ "$sz" -ge "$EXPECT" ]; then
    echo "[$(date '+%F %T')] unzipping"
    unzip -o -q "$TARGET" -d "$DATA_DIR/raw" && echo "[$(date '+%F %T')] unzip OK"
    du -sh "$DATA_DIR/raw" 2>/dev/null
    find "$DATA_DIR/raw" -maxdepth 3 -type d | head -40
else
    echo "[$(date '+%F %T')] DOWNLOAD INCOMPLETE"
fi
echo "[$(date '+%F %T')] ALL DONE"
