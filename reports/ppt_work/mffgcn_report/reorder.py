# -*- coding: utf-8 -*-
"""按新大纲（六章）重排页面。
章节页统一从 03.slide 派生（该页已按模板亮蓝风格改好），只换编号/中文/英文。
"""
import pathlib, re

SRC = pathlib.Path('slides')
DST = pathlib.Path('slides2')
DST.mkdir(exist_ok=True)

CHAP_TPL = (SRC / '03.slide').read_text(encoding='utf-8')


def chapter(num: str, cn: str, en: str) -> str:
    s = CHAP_TPL.replace('>01</Text>', f'>{num}</Text>', 1)
    s = s.replace('研究背景与问题', cn, 1)          # 标题
    s = s.replace('BACKGROUND &amp; PROBLEM', en, 1)  # 英文副标
    return s


# ('C', 编号, 中文, 英文) = 章节页；('F', 源页码) = 复用；None = 待定新页
PLAN = [
    ('01', ('F', '01'), None, None),
    ('02', ('F', '02'), 'CONTENTS', None),
    ('03', ('C', '01', '绪论', 'INTRODUCTION'), None, None),
    ('04', ('F', '04'), '01 · 绪论', '研究背景：无线电地图与它的观测难题'),
    ('05', None, '01 · 绪论', None),
    ('06', ('C', '02', '相关工作', 'RELATED WORK'), None, None),
    ('07', None, '02 · 相关工作', None),
    ('08', ('C', '03', '问题定义', 'PROBLEM FORMULATION'), None, None),
    ('09', ('F', '05'), '03 · 问题定义', '问题定义：三路输入，一张功率图'),
    ('10', ('C', '04', '所提方法', 'METHODOLOGY'), None, None),
    ('11', ('F', '07'), '04 · 方法', None),
    ('12', ('F', '08'), '04 · 方法', None),
    ('13', ('F', '09'), '04 · 方法', None),
    ('14', ('F', '10'), '04 · 方法', None),
    ('15', ('F', '11'), '04 · 方法', None),
    ('16', ('C', '05', '实验', 'EXPERIMENTS'), None, None),
    ('17', ('F', '13'), '05 · 实验设置', None),
    ('18', ('F', '14'), '05 · 实验设置', None),
    ('19', ('F', '15'), '05 · 实验设置', None),
    ('20', ('F', '17'), '05 · 实验结果', None),
    ('21', ('F', '18'), '05 · 实验结果', None),
    ('22', ('F', '19'), '05 · 实验结果', None),
    ('23', ('F', '20'), '05 · 实验结果', None),
    ('24', ('F', '21'), '05 · 实验结果', None),
    ('25', ('F', '22'), '05 · 实验结果', None),
    ('26', ('F', '23'), '05 · 实验结果', None),
    ('27', None, '05 · 实验结果', None),
    ('28', None, '05 · 实验结果', None),
    ('29', None, '05 · 实验结果', None),
    ('30', ('F', '26'), '05 · 实验结果', None),
    ('31', None, '05 · 实验结果', None),
    ('32', ('C', '06', '结论与展望', 'CONCLUSION'), None, None),
    ('33', ('F', '27'), '06 · 结论', '结论、局限与未来工作'),
    ('34', ('F', '28'), None, None),
]

TAG_RE = re.compile(r"(color: '#BBD9F2', textAlign: 'right' \}\}>)([^<]*)(</Text>)")
TITLE_RE = re.compile(r"(fontSize: 28, fontWeight: 'bold', color: '#FFFFFF', letterSpacing: 1 \}\}>)([^<]*)(</Text>)")

done, todo = [], []
for new, src, tag, title in PLAN:
    if src is None:
        todo.append(new); continue
    if src[0] == 'C':
        s = chapter(*src[1:])
    else:
        s = (SRC / f'{src[1]}.slide').read_text(encoding='utf-8')
    if tag:
        s, n1 = TAG_RE.subn(lambda m: m.group(1) + tag + m.group(3), s, count=1)
        if n1 != 1: print(f'  !! {new} tag')
    if title:
        s, n2 = TITLE_RE.subn(lambda m: m.group(1) + title + m.group(3), s, count=1)
        if n2 != 1: print(f'  !! {new} 标题')
    (DST / f'{new}.slide').write_text(s, encoding='utf-8')
    done.append(new)

print('已生成:', ' '.join(done))
print('待新写:', ' '.join(todo))
