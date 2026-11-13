import io, sys
for p in ["scripts/train_job2.sbatch"]:
    s = io.open(p, encoding="utf-8").read()
    old = '''# 不传 --workers 就用 16（A100 节点有 40 核，本地盘读得动）
HAS_WORKERS=0
for a in "$@"; do [ "$a" = "--workers" ] && HAS_WORKERS=1; done
WARGS=()
[ "$HAS_WORKERS" = "0" ] && WARGS=(--workers 16)

echo "=== TRAIN tag=$TAG ==="
echo "=== args: ${ARGS[*]:-（无）} ${WARGS[*]:-} ==="

python train.py --cache "$STAGE" --gpu 0 "${WARGS[@]}" "${ARGS[@]}"'''
    new = '''# 不传 --workers 就用 16（A100 节点有 40 核，本地盘读得动）
# 注意：节点 bash 是 4.2，set -u 下空数组 "${WARGS[@]}" 会报 unbound variable
# （bash 4.4 才修）。所以不再用独立数组，直接追加进 ARGS。
HAS_WORKERS=0
for a in "$@"; do [ "$a" = "--workers" ] && HAS_WORKERS=1; done
if [ "$HAS_WORKERS" = "0" ]; then ARGS+=(--workers 16); fi

echo "=== TRAIN tag=$TAG ==="
echo "=== args: ${ARGS[*]:-（无）} ==="

python train.py --cache "$STAGE" --gpu 0 "${ARGS[@]}"'''
    assert old in s, "pattern not found in " + p
    io.open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new, 1))
    print("patched", p)
