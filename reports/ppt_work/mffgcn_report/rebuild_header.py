# -*- coding: utf-8 -*-
"""把内容页的页头从「白底 + 左上深蓝标题 + 细线」改回模板版式：
顶部 image1 深蓝横幅（左白色页标题 + 右院徽 logo），内容区下移。
"""
import re, pathlib

SLIDES = pathlib.Path('slides')
SKIP = {'01', '03', '06', '12', '16', '28'}   # 封面 / 4 个章节页 / 结尾页

HDR_OPEN = "<Box style={{ flexDirection: 'row', alignItems: 'flex-end', justifyContent: 'space-between' }}>"
HDR_END = "background: '#123C6E', marginTop: 18 }} />"

BANNER = """    <Image src="assets/banner.jpg" style={{ position: 'absolute', top: 0, left: 0, width: 1280, height: 60, objectFit: 'cover' }} />
    <Text style={{ position: 'absolute', top: 15, left: 60, width: 820, fontSize: 28, fontWeight: 'bold', color: '#FFFFFF', letterSpacing: 1 }}>__TITLE__</Text>
    <Text style={{ position: 'absolute', top: 23, left: 900, width: 100, fontSize: 15, color: '#BBD9F2', textAlign: 'right' }}>__TAG__</Text>
    <Image src="assets/logo_bar.png" style={{ position: 'absolute', top: 8, left: 1010, width: 210, height: 44, objectFit: 'contain' }} />

    <Box style={{ position: 'absolute', top: 72, left: 60, width: 1160, height: 616, flexDirection: 'column' }}>
"""


def convert(num: str) -> bool:
    p = SLIDES / f'{num}.slide'
    s = p.read_text(encoding='utf-8')
    if 'assets/banner.jpg' in s:
        return False

    s = re.sub(r'<Slide style=\{\{[^}]*\}\}>',
               "<Slide style={{ width: 1280, height: 720, padding: 0, background: '#FFFFFF', position: 'relative' }}>",
               s, count=1)

    i = s.index(HDR_OPEN)
    j = s.index(HDR_END, i)
    j = s.index('\n', j) + 1
    head = s[i:j]

    texts = re.findall(r'<Text[^>]*>([^<]*)</Text>', head)
    if len(texts) < 2:
        print(f'  !! {num} 页头解析失败: {texts}')
        return False
    title, tag = texts[0].strip(), texts[1].strip()

    body = s[j:]
    body = body[:body.rindex('</Slide>')].strip('\n')
    body = re.sub(r'marginTop: (?:4[0-9]|3[0-9]|2[0-9])\b', 'marginTop: 22', body, count=1)
    body = '\n'.join(('    ' + ln) if ln.strip() else ln for ln in body.split('\n'))

    banner = BANNER.replace('__TITLE__', title).replace('__TAG__', tag)
    out = s[:i] + banner + '\n' + body + '\n    </Box>\n</Slide>\n'
    p.write_text(out, encoding='utf-8')
    print(f'  ok {num}  {title}  |  {tag}')
    return True


if __name__ == '__main__':
    nums = [f'{k:02d}' for k in range(1, 29) if f'{k:02d}' not in SKIP]
    n = sum(convert(x) for x in nums)
    print(f'共改造 {n} 页')
