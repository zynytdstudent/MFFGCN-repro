"""修复集群作业脚本：补内存申请、修 rme_gan 参数冲突。2026-11-05"""
import io
import os
import shutil

ROOT = "/public/home/zy_221126030197/work/mffgcn"
SCRIPTS = os.path.join(ROOT, "scripts")

MEM_LINE = "#SBATCH -t 24:00:00"

ADD = "#SBATCH -c 16\n#SBATCH --mem=32G"


def patch_header(path):
    s = io.open(path, encoding="utf-8").read()
    if "--mem=" in s:
        print("  [skip] 已有内存申请", os.path.basename(path))
        return
    if MEM_LINE not in s:
        print("  [WARN] 没找到时间行", os.path.basename(path))
        return
    s = s.replace(MEM_LINE, MEM_LINE + "\n" + ADD, 1)
    io.open(path, "w", encoding="utf-8").write(s)
    print("  [ok] 补内存/CPU", os.path.basename(path))


OLD_RME = '''if [ "$ARCH" = "rme_gan" ]; then
    python rme_gan_train.py --cache "$STAGE" --gpu 0 --workers 16 \\
        --phase both --phase1-epochs 20 --val-batches 60 "${ARGS[@]}"'''

NEW_RME = '''if [ "$ARCH" = "rme_gan" ]; then
    # rme_gan_train.py 不认 --arch，必须把它和它的值一起摘掉，否则 argparse 直接报错
    CARGS=()
    skip=0
    for ((i=0; i<${#ARGS[@]}; i++)); do
      if [ "$skip" = "1" ]; then skip=0; continue; fi
      if [ "${ARGS[$i]}" = "--arch" ]; then skip=1; continue; fi
      CARGS+=("${ARGS[$i]}")
    done
    python rme_gan_train.py --cache "$STAGE" --gpu 0 --workers 16 \\
        --phase both --phase1-epochs 20 --val-batches 60 "${CARGS[@]}"'''


def patch_base(path):
    s = io.open(path, encoding="utf-8").read()
    if "CARGS" in s:
        print("  [skip] rme_gan 已修", os.path.basename(path))
        return
    if OLD_RME not in s:
        print("  [WARN] 找不到 rme_gan 段", os.path.basename(path))
        return
    s = s.replace(OLD_RME, NEW_RME, 1)
    io.open(path, "w", encoding="utf-8").write(s)
    print("  [ok] 修 rme_gan 参数", os.path.basename(path))


for name in ("train_job2.sbatch", "base_job2.sbatch"):
    p = os.path.join(SCRIPTS, name)
    if not os.path.exists(p):
        print("  [WARN] 不存在", p)
        continue
    shutil.copyfile(p, p + ".bak.1105")
    patch_header(p)
    if "base" in name:
        patch_base(p)

print("done")
