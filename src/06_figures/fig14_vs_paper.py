# -*- coding: utf-8 -*-
"""论文 Fig.14（环境图脉冲噪声）与我们复现的逐点对照。

数据来源
--------
- `results/fig14_ours_flip.json`  旧口径：选中 p 比例像素逐个翻转（实际改 p）
- `results/fig14_ours_sp.json`    标准椒盐：选中 p 比例，一半置 1 一半置 0（实际改 p/2）
  两者都由 `supp_eval.py --mode noise --n 200 --paper-grid --span` 产出，
  区别只在 `--noise-defn`。
- 论文侧：`results/paper_digitized.json` 的 "14" 节点（Fig.14(a) NMSE 数字化），
  分母用 Table III（Setup 3）的干净 NMSE：MFFGCN 0.0046 / RME-GAN 0.0054 /
  RadioUNet 0.0063。Fig.14 本身没画 0 档，所以只能借 Table III 当分母。

为什么要区分两种口径
------------------
论文正文只写 "we use impulse noise"；图像处理里该词的默认定义是椒盐，
而椒盐在强度 p 下真正改变的像素只有 p/2。所以旧口径的 1.0% 相当于把
扰动放大了一倍。用 sp 重跑后幅度整体下降，**但三模型的抗噪排序一点没动**，
这才是结论所在。

用法： python fig14_vs_paper.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
for _rel in ("..", os.path.join("..", "..")):
    _c = os.path.abspath(os.path.join(HERE, _rel))
    if os.path.isdir(os.path.join(_c, "results")):
        ROOT = _c
        break
RES = os.path.join(ROOT, "results")

PAPER_CLEAN = {"MFFGCN": 0.0046, "RME-GAN": 0.0054, "RadioUNet": 0.0063}
PAPER_FIG14 = {
    "MFFGCN":    [None, 0.00669, 0.00907, 0.01229],
    "RME-GAN":   [0.00762, 0.01816, 0.02589, 0.03941],
    "RadioUNet": [0.00864, 0.02564, 0.04629, 0.06592],
}
LV = ["0.0", "0.001", "0.004", "0.007", "0.01"]
HDR = ["clean", "0.1%", "0.4%", "0.7%", "1.0%"]
ORDER = ["MFFGCN", "RadioUNet", "RME-GAN"]


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as fh:
        return json.load(fh)


def main():
    flip = load("fig14_ours_flip.json")
    sp = load("fig14_ours_sp.json")

    print("=" * 76)
    print("劣化倍数（该档 NMSE / 干净 NMSE）")
    print("=" * 76)
    print("%-16s %7s %7s %7s %7s %7s" % ("", *HDR))
    for m in ORDER:
        pr = ["" if v is None else "%6.2fx" % (v / PAPER_CLEAN[m])
              for v in PAPER_FIG14[m]]
        print("%-16s %7s %7s %7s %7s %7s" % ("论文 " + m, "1.00x", *pr))
        for tag, d in (("我们 flip", flip), ("我们 sp  ", sp)):
            v = [d["models"][m][l]["nmse"] for l in LV]
            print("%-16s %7s %7s %7s %7s %7s"
                  % (tag, *["%6.2fx" % (x / v[0]) for x in v]))
        print()

    print("=" * 76)
    print("1.0% 档抗噪排序")
    print("=" * 76)
    pr = sorted((PAPER_FIG14[m][3] / PAPER_CLEAN[m], m) for m in ORDER)
    print("论文     :", " < ".join("%s %.2fx" % (m, v) for v, m in pr))
    for tag, d in (("我们 flip:", flip), ("我们 sp  :", sp)):
        rr = sorted((d["models"][m]["0.01"]["nmse"]
                     / d["models"][m]["0.0"]["nmse"], m) for m in ORDER)
        print(tag, " < ".join("%s %.2fx" % (m, v) for v, m in rr))
    print()
    print("论文是 MFFGCN 最稳、RadioUNet 最差；我们两套口径下都是")
    print("RME-GAN 最稳、MFFGCN 最差 —— 排序完全反转，与噪声定义无关。")


if __name__ == "__main__":
    main()
