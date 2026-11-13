"""把论文 Fig.14/15/16 的曲线从位图里反解成数值。

为什么要这么做：论文只给了图片，没给数据表。要判断"复现图和论文图差多少"，
必须先把论文曲线变成数，才能逐点算相对偏差，而不是靠肉眼说"看起来不一样"。

做法：
  1. 用坐标轴框线定位绘图区（左/右/上/下 spine）
  2. 在左轴外侧找刻度短线的行位置，用已知刻度值线性映射 像素 -> 数值
  3. 按 matplotlib 默认三色（tab:blue/orange/green）分离三条曲线，
     用连通域取"横跨最宽"那块 = 曲线本体，图例色块自动被丢掉
  4. 三条曲线共用同一组横坐标：先用标记最清楚的那条定 x，再统一采样
"""
import json
import numpy as np
from PIL import Image

BLUE = (31, 119, 179)
ORANGE = (255, 127, 14)
GREEN = (44, 159, 44)
COLORS = [("RadioUNet", BLUE), ("RME-GAN", ORANGE), ("MFFGCN", GREEN)]


def load(n):
    p = "reports/fig_compare/paper/paper_fig%s.png" % n
    return np.asarray(Image.open(p).convert("RGB")).astype(int)


# ----------------------------------------------------------------------
# 坐标轴定位
# ----------------------------------------------------------------------
def tick_rows(a, x0, y0, y1):
    """左轴外侧的刻度短线所在行（像素）。

    不能直接在 x0-9..x0-2 这一整条带上找暗像素：那里会混进纵轴标题的
    竖排文字。改成从轴框往左走"连续暗像素"，只有刻度线才是紧贴轴框、
    长度 2~8 px 的一段；数字标签和标题都不与轴框相连。
    """
    dark = a.sum(2) < 420
    rows, i = [], y0
    while i < y1:
        run = 0
        x = x0 - 1
        while x > x0 - 14 and dark[i, x]:
            run += 1
            x -= 1
        if 2 <= run <= 8:
            rows.append(float(i))
            i += 2
        else:
            i += 1
    out = []
    for r in rows:
        if out and r - out[-1] <= 2.5:
            out[-1] = (out[-1] + r) / 2.0
        else:
            out.append(r)
    return out


def axis_map(rows, v_lo, v_hi):
    """刻度行 -> 数值。rows 从上到下，v_hi 是最上面那条刻度的值。"""
    rows = sorted(rows)
    if len(rows) < 2:
        return None

    def px2v(y):
        return v_hi - (y - rows[0]) / (rows[-1] - rows[0]) * (v_hi - v_lo)

    return px2v


# ----------------------------------------------------------------------
# 颜色分离
# ----------------------------------------------------------------------
def color_mask(a, c, tol=45):
    return ((np.abs(a[:, :, 0] - c[0]) < tol) &
            (np.abs(a[:, :, 1] - c[1]) < tol) &
            (np.abs(a[:, :, 2] - c[2]) < tol))


def widest_component(mask):
    """取"横跨最宽"的连通域 = 曲线本身；图例色块是独立小块，自动被丢掉。

    本地没装 scipy，这里用纯 numpy 的 8 邻域泛洪，图像只有几百像素宽，够用。
    """
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best, bestw = None, -1
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < h and 0 <= nx < w
                                and mask[ny, nx] and not seen[ny, nx]):
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            xs = [p[1] for p in comp]
            span = max(xs) - min(xs)
            if span > bestw:
                bestw, best = span, comp
    if best is None:
        return None
    out = np.zeros_like(mask, dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out


def curve_mask(a, c, box, tol=45):
    x0, x1, y0, y1 = box
    m = color_mask(a, c, tol)
    m[:y0, :] = False
    m[y1:, :] = False
    m[:, :x0] = False
    m[:, x1:] = False
    return widest_component(m)


def marker_xs(mask, x0, x1):
    """标记符号处曲线竖跨明显变大，把这些列聚成数据点。"""
    if mask is None:
        return []
    cols = np.arange(x0, x1)
    span = np.zeros(len(cols))
    for k, x in enumerate(cols):
        ys = np.where(mask[:, x])[0]
        span[k] = (ys.max() - ys.min()) if len(ys) else 0
    base = np.median(span[span > 0]) if (span > 0).any() else 2.0
    thr = max(base + 3.0, base * 1.8)
    idx = np.where(span >= thr)[0]
    if len(idx) == 0:
        return []
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= 8:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return [float(cols[g].mean()) for g in groups if len(g) >= 2]


def sample_at(mask, x, half=3):
    """在 x 附近 half 列内取曲线中心行。"""
    if mask is None:
        return None
    sub = mask[:, int(x - half):int(x + half) + 1]
    ys = np.where(sub.any(1))[0]
    if len(ys) == 0:
        return None
    return float((ys.min() + ys.max()) / 2.0)


def bar_tops(a, c, box, baseline, tol=45):
    """柱状图：返回每个柱子的 (x中心, 柱顶y)。

    注意：图例色块就压在第一根柱子上方，直接用"该颜色最靠上的像素"会把
    图例也算进去。改成从基线往上找**连续**的一段，那才是柱子本体。
    """
    x0, x1, y0, y1 = box
    m = color_mask(a, c, tol)
    m[:y0, :] = False
    m[y1:, :] = False
    m[:, :x0] = False
    m[:, x1:] = False
    cols = np.where(m.any(0))[0]
    if len(cols) == 0:
        return []
    groups, cur = [], [cols[0]]
    for x in cols[1:]:
        if x - cur[-1] <= 2:
            cur.append(x)
        else:
            groups.append(cur)
            cur = [x]
    groups.append(cur)

    out = []
    for g in groups:
        if len(g) < 12:
            continue
        # 图例色块可能和第一根柱子粘成一列组，用"贴近基线的列"把柱体挑出来
        body = [x for x in g if m[max(0, baseline - 6), x]]
        if len(body) < 10:
            continue
        xc = int(np.mean(body))
        rows = np.where(m[:, xc])[0]
        if len(rows) == 0:
            continue
        top = rows[-1]
        k = len(rows) - 1
        while k > 0 and rows[k] - rows[k - 1] <= 2:
            top = rows[k - 1]
            k -= 1
        if rows[-1] < baseline - 8:   # 不贴基线 → 图例色块
            continue
        out.append((float(xc), float(top)))
    return out


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def run(n, panels, kind="line"):
    a = load(n)
    out = {"fig": n, "panels": {}}
    for key, (box, v_lo, v_hi) in panels.items():
        x0, x1, y0, y1 = box
        rows = tick_rows(a, x0, y0 - 4, y1 + 4)
        f = axis_map(rows, v_lo, v_hi)
        if f is None:
            print("fig%s%s 刻度识别失败" % (n, key))
            continue

        if kind == "bar":
            baseline = int(round(rows[-1]))   # 最下面那条刻度 = 柱子基线
            ser = {}
            for nm, c in COLORS:
                pts = bar_tops(a, c, box, baseline)
                ser[nm] = [round(f(y), 5) for _, y in pts]
                print("fig%s%s %-10s 柱数=%d y=%s"
                      % (n, key, nm, len(pts), ser[nm]))
            out["panels"][key] = {"series": ser}
            continue

        masks = {nm: curve_mask(a, c, box) for nm, c in COLORS}
        # 三条曲线的标记位置会互相补救：单条可能只认出 2~3 个点，
        # 把三条的标记位置并起来再聚类，就能稳定拿到 4 个横坐标。
        cand = []
        for nm, _ in COLORS:
            cand += marker_xs(masks[nm], x0, x1)
        cand.sort()
        clus = []
        for v in cand:
            if clus and v - clus[-1][-1] <= 15:
                clus[-1].append(v)
            else:
                clus.append([v])
        clus = sorted(clus, key=len, reverse=True)[:4]
        xs = sorted(float(np.mean(g)) for g in clus)
        # 尾部那个点在三条曲线汇合处常被压掉，用等间距外推补回来
        if len(xs) == 3:
            xs = xs + [xs[-1] + (xs[-1] - xs[-2])]
        if not xs:
            print("fig%s%s 取点失败" % (n, key))
            continue
        ser = {}
        for nm, _ in COLORS:
            ys = [sample_at(masks[nm], x) for x in xs]
            ser[nm] = [None if y is None else round(f(y), 5) for y in ys]
        print("fig%s%s %d 点  x=%s" % (n, key, len(xs), [int(v) for v in xs]))
        for nm, _ in COLORS:
            print("        %-10s %s" % (nm, ser[nm]))
        out["panels"][key] = {"x_pix": [round(v, 1) for v in xs], "series": ser}
    return out


# 各图的刻度值（从图上读出的最小/最大刻度）
FIGS = {
    "14": {"a": ((112, 449, 29, 412), 0.01, 0.07),
           "b": ((639, 977, 29, 412), 0.02, 0.07)},
    "15": {"a": ((112, 469, 148, 526), 0.02, 0.07),
           "b": ((653, 997, 148, 526), 0.035, 0.065)},
    "16": {"a": ((85, 470, 27, 466), 0.02, 0.09),
           "b": ((612, 1005, 27, 466), 0.03, 0.08)},
}

if __name__ == "__main__":
    allout = {}
    for n, panels in FIGS.items():
        allout[n] = run(n, panels, "bar" if n == "16" else "line")
    with open("results/paper_digitized.json", "w", encoding="utf-8") as fh:
        json.dump(allout, fh, ensure_ascii=False, indent=1)
    print("saved results/paper_digitized.json")
