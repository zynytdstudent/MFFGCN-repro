import subprocess, json, sys, pathlib, os
from pptx import Presentation

ROOT = pathlib.Path(r'D:\Projects\MFFGCN\ppt_work\mffgcn_report')
PPTX = ROOT / 'build35.pptx'
SLIDES = ROOT / 'slides3'
SLIDEP = 'slidep.cmd'


def run(args, check=True):
    r = subprocess.run([SLIDEP] + args, cwd=str(ROOT), capture_output=True, text=True,
                       encoding='utf-8', errors='replace')
    return r.returncode, (r.stdout or '') + (r.stderr or '')


def count():
    return len(Presentation(str(PPTX)).slides)


if PPTX.exists():
    PPTX.unlink()

code, out = run(['create', PPTX.name])
print('create rc=', code, out.strip()[:200])
n0 = count()
print('初始页数', n0)

files = sorted(SLIDES.glob('*.slide'))
assert len(files) == 35, len(files)

for i, f in enumerate(files):
    if i < n0:
        args = ['upsert-dsl', PPTX.name, '--dsl-file', str(f), '--page-index', str(i)]
    else:
        args = ['upsert-dsl', PPTX.name, '--dsl-file', str(f)]
    code, out = run(args)
    if code != 0:
        print(f'!! upsert 失败 {f.name}: {out[-400:]}')
        sys.exit(1)

print('写入后页数', count())

# 逐页 lint
bad = []
for i, f in enumerate(files):
    code, out = run(['lint', PPTX.name, '--dsl-file', str(f), '--page-index', str(i)], check=False)
    ok = '"ok": true' in out or '"ok":true' in out
    if not ok:
        bad.append((f.name, i, out[-600:]))
    print(f'{f.name} {"OK" if ok else "FAIL"}')

if bad:
    print('\n==== 失败明细 ====')
    for name, idx, msg in bad:
        print(name, idx, msg)
else:
    print('\n全部通过')
