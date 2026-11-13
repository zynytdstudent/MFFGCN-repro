# -*- coding: utf-8 -*-
import json, subprocess, pathlib, re

PPTX = 'MFFGCN组会汇报.pptx'
SLIDES = pathlib.Path('slides')
bad = []

for n in range(1, 29):
    f = SLIDES / f'{n:02d}.slide'
    idx = n - 1
    r = subprocess.run(['slidep.cmd', 'upsert-dsl', PPTX, '--dsl-file', str(f), '--page-index', str(idx)],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    ok = '"ok":true' in r.stdout
    if not ok:
        print(f'[{n:02d}] UPSERT FAIL')
        print((r.stdout + r.stderr)[-800:])
        bad.append((n, 'upsert'))
        continue

    r2 = subprocess.run(['slidep.cmd', 'lint', PPTX, '--dsl-file', str(f), '--page-index', str(idx)],
                        capture_output=True, text=True, encoding='utf-8', errors='replace')
    try:
        line = [l for l in r2.stdout.splitlines() if l.strip().startswith('{')][-1]
        d = json.loads(line)
    except Exception:
        print(f'[{n:02d}] LINT PARSE FAIL'); print(r2.stdout[-500:]); bad.append((n, 'parse')); continue

    if not d.get('ok'):
        ds = d.get('diagnostics', [])
        print(f'[{n:02d}] LINT {len(ds)} 条')
        for x in ds[:4]:
            print('     ', x.get('severity'), x.get('code'), str(x.get('message'))[:140])
        bad.append((n, len(ds)))
    else:
        print(f'[{n:02d}] ok')

print('\n==== 失败清单 ====')
for b in bad:
    print(b)
