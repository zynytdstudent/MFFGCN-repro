import pathlib, shutil

ROOT = pathlib.Path(r'D:\Projects\MFFGCN\ppt_work\mffgcn_report')
SRC = ROOT / 'slides3'
DST = ROOT / 'slides4'

MAP = {
    1: 1, 3: 3, 5: 7, 6: 10, 8: 12, 9: 13, 10: 14, 13: 16, 15: 17,
    17: 22, 19: 24, 20: 25, 21: 26, 22: 27, 25: 30, 26: 31, 27: 32,
    28: 33, 29: 34, 30: 35,
}

GLOBAL = [
    ('04 · 方法', '02 · 所提方法'),
    ('02 · 相关工作', '01 · 绪论'),
    ('05 · 实验总览', '03 · 实验总览'),
    ('05 · 实验设置', '03 · 实验设置'),
    ('05 · 实验结果', '03 · 实验结果'),
    ('06 · 结论', '04 · 结论'),
]

# 章节页：改编号与标题
SECTION = {
    3:  [("opacity: 0.42 }}>01", "opacity: 0.42 }}>01"),
         ("color: '#FFFFFF' }}>绪论</Text>", "color: '#FFFFFF' }}>绪论与相关工作</Text>"),
         ("letterSpacing: 2 }}>INTRODUCTION</Text>", "letterSpacing: 2 }}>INTRODUCTION &amp; RELATED WORK</Text>")],
    6:  [("opacity: 0.42 }}>04", "opacity: 0.42 }}>02")],
    13: [("opacity: 0.42 }}>05", "opacity: 0.42 }}>03")],
    28: [("opacity: 0.42 }}>06", "opacity: 0.42 }}>04")],
}

EXTRA = {
    15: [('11.8%–14.8%', '11.8%–16.4%')],
}

for new, old in MAP.items():
    s = (SRC / f'{old:02d}.slide').read_text(encoding='utf-8')
    for a, b in GLOBAL:
        s = s.replace(a, b)
    for a, b in SECTION.get(new, []):
        s = s.replace(a, b)
    for a, b in EXTRA.get(new, []):
        s = s.replace(a, b)
    (DST / f'{new:02d}.slide').write_text(s, encoding='utf-8')

have = sorted(p.name for p in DST.glob('*.slide'))
print('共', len(have), '页')
missing = [f'{i:02d}.slide' for i in range(1, 31) if f'{i:02d}.slide' not in have]
print('缺:', missing if missing else '无')
