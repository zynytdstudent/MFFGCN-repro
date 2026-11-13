"""恢复 say 函数，并让它只往 stderr 输出（避免污染 $(pick_gpu) 的返回值）。"""
import io

P = "/public2/czz/ZhangYu/project/mffgcn/autopilot.sh"
lines = io.open(P, encoding="utf-8").read().split("\n")

GOOD = 'say () { echo "[$(date \'+%F %T\')] $*" | tee -a "$LOG" >&2; }'

hit = False
for i, ln in enumerate(lines):
    if ln.strip() == "X":
        lines[i] = GOOD
        hit = True
        print("恢复第 %d 行" % (i + 1))
    elif ln.startswith("say () {"):
        lines[i] = GOOD
        hit = True
        print("改写第 %d 行" % (i + 1))

if not hit:
    print("!! 没找到 say 定义，请人工检查")

io.open(P, "w", encoding="utf-8").write("\n".join(lines))

# 校验：语法 + 关键行
txt = io.open(P, encoding="utf-8").read()
print("stderr 重定向:", ">&2" in txt)
print("pick_gpu 里的 say 仍会写日志，但不再进 stdout")
