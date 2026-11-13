# -*- coding: utf-8 -*-
"""在线对照页的定点修补：按图号替换某张卡的图片链接与说明文字。

为什么需要它
------------
「15-论文图逐张对照」是单文件页面，13 张图全部 base64 内嵌，整页约 4.7 MB。
改一张图时不可能靠手写正则，也不该重写整页——这里按卡片 img 的
data-page-node-id（pnid）精确定位，只替换该 img 的 src 和该卡 .cap 里的
说明文字，其余字节原样保留，便于事后 diff 自检。

图号 → pnid 的映射与 .workbuddy/memory/MEMORY.md 里的表一致。

用法
----
    # 只换图（新图须已由 manage/upload_image.py 转成 https 直链）
    python patch_compare_page.py --html in.html --fig 7 \
        --img-url "https://.../fig7_ours.png" --out out.html

    # 换图 + 换说明文字（说明文字支持简单 HTML，如 <b>…</b>）
    python patch_compare_page.py --html in.html --fig 7 \
        --img-url "https://.../fig7_ours.png" \
        --cap-file new_cap.html --out out.html

    # 只换说明文字
    python patch_compare_page.py --html in.html --fig 7 \
        --cap-file new_cap.html --out out.html

状态药丸（"对齐" / "形状对齐·基线待修" 等）不受影响，除非再传
--cap-tag 与 --cap-tag-color 一并对它做改动。
"""
import argparse
import re
import sys

# 卡片序号(1起) → (图号, img 的 data-page-node-id)
FIG2PNID = {
    6.1: "OzcJM0n7FxIGzZX86mlci3",   # Fig.6(a) loss s1
    6.2: "fM6kNXwmFdLLp1FxSuXAK1",   # Fig.6(b) loss s2
    6.3: "O7DMUBnOMEVFLwlYBFTS3b",   # Fig.6(c) loss s3
    7: "4FXJcJoIeJGI2vfzyMIYIp",     # Fig.7 观测点数
    8: "bQrmVNItfSGVqFC4FG5MG7",     # Fig.8 距离档位
    9: "MVaBrxpx7T0mUS6z9AjDGD",     # Fig.9 距离
    10: "oramfblvVMEYtsRv92TNET",    # Fig.10 边长
    11: "dMTDCPvodq2ZWVVJkc0YOV",    # Fig.11 定性
    12: "XPgAXFv37Z5rQpgN0Ojka1",    # Fig.12 kNN 图
    13: "4LodpRvofTmblA2BsC242k",    # Fig.13 深度剖面
    14: "NPaf2MbGu6IJeylMXnXLQT",    # Fig.14 建筑翻转
    15: "rcI8MDjA7bQMJJG2J2qITo",    # Fig.15 分辨率
    16: "ZPBITOwD92VbsnXKP7Sr3F",    # Fig.16 建筑数量
}


def patch_img_src(html, pnid, url):
    """只改该 pnid 对应 img 标签内的 src，返回 (新 html, 旧 src 前缀)。"""
    pat = re.compile(r'<img[^>]*data-page-node-id="%s"[^>]*>' % re.escape(pnid))
    m = pat.search(html)
    if not m:
        raise SystemExit("[fail] 找不到 img pnid=%s" % pnid)
    tag = m.group(0)
    sm = re.search(r'src="([^"]*)"', tag)
    if not sm:
        raise SystemExit("[fail] 该 img 没有 src 属性")
    old = sm.group(1)
    old_head = old[:60] + ("…" if len(old) > 60 else "")
    new_tag = tag[:sm.start(1)] + url + tag[sm.end(1):]
    return html[:m.start()] + new_tag + html[m.end():], old_head


def patch_cap(html, pnid, new_text, tag_text=None, tag_color=None):
    """替换该卡 .cap 里状态药丸之后的说明文字。

    cap 结构固定为：
        <div ... class="cap"><span ... class="tag" ...>状态</span><!--pnid:..-->说明</div>
    只动说明部分；给了 tag_text 时才改药丸文案/底色。
    """
    m = re.search(r'<img[^>]*data-page-node-id="%s"[^>]*>' % re.escape(pnid), html)
    if not m:
        raise SystemExit("[fail] 找不到 img pnid=%s" % pnid)
    cm = re.compile(r'<div[^>]*class="cap">').search(html, m.end())
    if not cm:
        raise SystemExit("[fail] 该 img 之后没有 .cap 卡片说明")
    tail = html[cm.end():]
    close = tail.find("</div>")
    if close < 0:
        raise SystemExit("[fail] .cap 没有闭合")
    inner = tail[:close]

    sm = re.search(r'<span[^>]*class="tag"[^>]*>.*?</span>', inner, re.S)
    if sm:
        span = sm.group(0)
        # 药丸内部的 pnid 注释是编辑器的锚点，换文案时要留住
        inner_pnids = re.findall(r'<!--pnid:[^>]*-->', span)
        if tag_text:
            span = re.sub(
                r'(<span[^>]*class="tag"[^>]*>).*?(</span>)',
                lambda m: m.group(1) + "".join(inner_pnids) + tag_text + m.group(2),
                span, flags=re.S)
        if tag_color:
            if 'style="' in span:
                span = re.sub(r'style="[^"]*"',
                              'style="background:%s"' % tag_color, span, count=1)
            else:
                span = span.replace('<span', '<span style="background:%s"' % tag_color, 1)
        head = inner[:sm.start()] + span
        # 药丸与说明之间通常还夹一个 pnid 注释，同样保留
        between = inner[sm.end():]
    else:
        head, between = inner, ""

    keep = "".join(re.findall(r'<!--pnid:[^>]*-->', between))
    new_inner = head + keep + new_text
    return html[:cm.end()] + new_inner + tail[close:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", required=True, help="下载下来的页面 html")
    ap.add_argument("--fig", required=True,
                    help="图号：7 / 8 / … / 16，Fig.6 用 6.1、6.2、6.3")
    ap.add_argument("--img-url", default="", help="新的图片 https 直链")
    ap.add_argument("--cap-file", default="", help="新说明文字的 html 片段文件")
    ap.add_argument("--cap-text", default="", help="新说明文字（短句可直接传）")
    ap.add_argument("--cap-tag", default="", help="新状态药丸文案")
    ap.add_argument("--cap-tag-color", default="", help="新药丸底色，如 #1a7f4b")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    try:
        key = float(a.fig)
    except ValueError:
        raise SystemExit("[fail] --fig 给数字，Fig.6 用 6.1/6.2/6.3")
    pnid = FIG2PNID.get(key)
    if not pnid:
        raise SystemExit("[fail] 未知图号 %s，可选：%s" % (a.fig, sorted(FIG2PNID)))

    # newline="" 必须给：云端页面是 CRLF，默认通用换行会把 \r 吃掉，
    # 输出又用 newline="" 写回，结果是整页 55 行全部变成 LF，diff 一片红。
    with open(a.html, encoding="utf-8", newline="") as fh:
        html = fh.read()
    n0 = len(html)

    if a.img_url:
        if not a.img_url.startswith("https://"):
            raise SystemExit("[fail] 图片必须是 https 直链（先走 upload_image 上传）")
        html, old = patch_img_src(html, pnid, a.img_url)
        print("[img] Fig.%s  src %s -> %s" % (a.fig, old, a.img_url[:70]))

    text = a.cap_text
    if a.cap_file:
        text = open(a.cap_file, encoding="utf-8").read().strip()
    if text or a.cap_tag or a.cap_tag_color:
        if not text:
            # 只改药丸时，沿用原说明文字
            text = None
        if text is None:
            raise SystemExit("[fail] 改药丸时必须同时给说明文字，避免误删")
        html = patch_cap(html, pnid, text, a.cap_tag or None,
                         a.cap_tag_color or None)
        print("[cap] Fig.%s  说明已替换（%d 字）" % (a.fig, len(text)))

    open(a.out, "w", encoding="utf-8", newline="").write(html)
    print("[ok] %s  %d -> %d 字节" % (a.out, n0, len(html)))


if __name__ == "__main__":
    sys.exit(main())
