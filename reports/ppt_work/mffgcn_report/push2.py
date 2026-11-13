# -*- coding: utf-8 -*-
import json, subprocess, pathlib

PPTX = 'MFFGCN组会汇报.pptx'
D = pathlib.Path('slides2')
bad = []

for n in range(1, 35):
    f = D / f'{n:02d}.slide'
    cmd = ['slidep.cmd', 'upsert-dsl', PPTX, '--dsl-file', str(f)]
    if n <= 28:
        cmd += ['--page-index', str(n - 1)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if '"ok":true' not in r.stdout:
        print(f'[{n:02d}] UPSERT FAIL'); print((r.stdout + r.stderr)[-600:]); bad.append(n); continue

    idx = n - 1
    r2 = subprocess.run(['slidep.cmd', 'lint', PPTX, '--dsl-file', str(f), '--page-index', str(idx)],
                        capture_output=True, text=True, encoding='utf-8', errors='replace')
    try:
        d = json.loads([l for l in r2.stdout.splitlines() if l.strip().startswith('{')][-1])
    except Exception:
        print(f'[{n:02d}] PARSE FAIL'); print(r2.stdout[-400:]); bad.append(n); continue
    if not d.get('ok'):
        ds = d.get('diagnostics', [])
        print(f'[{n:02d}] {len(ds)} 条:')
        for x in ds[:5]:
            print('      ', x.get('severity'), str(x.get('message'))[:130])
        bad.append(n)
    else:
        print(f'[{n:02d}] ok')

print('\n失败:', bad)
