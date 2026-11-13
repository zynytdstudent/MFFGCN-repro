# -*- coding: utf-8 -*-
"""给 supp_eval.py 增加标准椒盐（salt & pepper）噪声口径。

论文正文只说 "we use impulse noise"，图像处理里 impulse noise 的默认定义是
椒盐：选中 p 比例的像素，一半置 1、一半置 0。置成与原来相同的值是空操作，
所以真正改变的像素只有 p/2。我们原来的实现是"选中 p 比例并逐个翻转"，
等于把扰动放大了整整一倍。

本补丁：
  1. impulse_noise 增加 defn 参数（flip=旧行为，sp=椒盐）；
  2. run_noise 把 args.noise_defn 传下去，并写进输出 json；
  3. 新增 --noise-defn 参数，默认 flip（保证旧结果可复现）；
  4. 输出文件名在 sp 口径下追加 _sp 后缀，不覆盖旧结果。

用法（在 src/ 下）：
    python supp_eval.py --mode noise --n 200 --paper-grid --span \
        --noise-defn sp --gpu 1
"""
import io

P = "supp_eval.py"

OLD = '''def impulse_noise(bld, p, rng):
    """环境图脉冲噪声：随机挑 p 比例的像素翻转（建筑↔空地）。"""
    if p <= 0:
        return bld
    x = bld.clone()
    n_pix = int(round(p * x.numel()))
    idx = torch.from_numpy(rng.choice(x.numel(), size=n_pix, replace=False))
    flat = x.view(-1)
    flat[idx] = 1.0 - flat[idx]
    return x'''

NEW = '''def impulse_noise(bld, p, rng, defn="flip"):
    """环境图脉冲噪声，两种口径。

    flip : 随机挑 p 比例的像素**翻转**（0<->1）。被选中的像素一定改变，
           所以 p 就是"实际被改动的像素比例"。这是 Fig.14 一直用的旧口径。
    sp   : 标准椒盐噪声（salt & pepper）。随机挑 p 比例的像素，其中一半置 1
           (salt)、一半置 0 (pepper)。**置成与原来相同的值是空操作**，
           因此真正改变的像素只有 p/2。

    为什么补 sp：论文正文只写 "we use impulse noise"，而图像处理里 impulse
    noise 的默认定义就是椒盐。同一张图，论文的 1.0% 等效于 flip 的 0.5%；
    之前直接拿 flip-1.0% 去对论文 1.0%，扰动被放大了一倍。
    """
    if p <= 0:
        return bld
    x = bld.clone()
    n = x.numel()
    flat = x.view(-1)
    k = int(round(p * n))
    idx = torch.from_numpy(rng.choice(n, size=k, replace=False))
    if defn == "sp":
        salt = torch.from_numpy(rng.random(k) < 0.5)
        flat[idx] = salt.float()
        return x
    flat[idx] = 1.0 - flat[idx]
    return x'''


def main():
    s = io.open(P, encoding="utf-8").read()
    n_ok = 0

    if OLD in s:
        s = s.replace(OLD, NEW)
        n_ok += 1
    elif "defn=\"sp\"" in s or 'defn="sp"' in s:
        print("[skip] impulse_noise 已是新版")
    else:
        raise SystemExit("找不到 impulse_noise 原文，先人工看一眼")

    old_call = "env[:, 0:1] = impulse_noise(env0[:, 0:1], l, rng)"
    new_call = ('env[:, 0:1] = impulse_noise(\n'
                '                    env0[:, 0:1], l, rng,\n'
                '                    getattr(args, "noise_defn", "flip"))')
    if old_call in s:
        s = s.replace(old_call, new_call)
        n_ok += 1
    else:
        print("[skip] 调用点已改过")

    old_out = '"models": {name: {str(l): acc[(name, l)].out() for l in levels}'
    if old_out in s and "noise_defn" not in s:
        s = s.replace(
            'out = {"mode": "noise", "n": args.n, "setup": args.setup, "span": args.span,',
            'out = {"mode": "noise", "n": args.n, "setup": args.setup,\n'
            '           "span": args.span,\n'
            '           "noise_defn": getattr(args, "noise_defn", "flip"),')
        n_ok += 1

    anchor = '    ap.add_argument("--span", action="store_true",'
    if "--noise-defn" not in s:
        s = s.replace(
            anchor,
            '    ap.add_argument("--noise-defn", default="flip",\n'
            '                    choices=["flip", "sp"],\n'
            '                    help="脉冲噪声口径：flip=翻转 p 比例像素（默认，"\n'
            '                         "旧行为）；sp=标准椒盐，实际只改动 p/2")\n'
            + anchor)
        n_ok += 1

    old_sfx = 'suffix = "_paper" if a.paper_grid else ""'
    if old_sfx in s and "_sp" not in s:
        s = s.replace(
            old_sfx,
            old_sfx + '\n    if getattr(a, "noise_defn", "flip") == "sp":\n'
                      '        suffix += "_sp"')
        n_ok += 1

    io.open(P, "w", encoding="utf-8").write(s)
    print("[patched] %d 处" % n_ok)


if __name__ == "__main__":
    main()
