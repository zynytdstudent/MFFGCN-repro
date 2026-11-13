# -*- coding: utf-8 -*-
"""把 Fig.9 / Fig.10 的"我们 vs 论文"差距拆成可解释的两个因子。

恒等式（NMSE 的定义决定的，不是拟合）：
    NMSE = RMSE² / E[t²]
所以同一档位下
    NMSE_ours / NMSE_paper = (RMSE_ours / RMSE_paper)² × (E_paper / E_ours)
                              └──── 真实模型差距 ────┘   └── 分母口径差 ──┘

E[t²] 是测试集目标图的平均能量。它的实测值：
    前 200 张（= supp_eval 的 n=200，约 2.5 个城市）  0.04478
    全 8080 张                                        0.06879
    paper（由 Fig.9/10 的 NMSE 与 RMSE 互推，18 组一致）0.0532
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..", "results")

E_OURS_200 = 0.04478      # supp_eval 实际用到的前 200 张
E_FULL = 0.06879          # 全 test 集
MODELS = ["RadioUNet", "RME-GAN", "MFFGCN"]
LEV = {"dist": ["50", "70", "90", "110"], "side": ["90", "110", "130", "150"]}


def main():
    pap = json.load(open(os.path.join(R, "paper_fig9_10_digitized.json")))
    rows = []
    for mode, fkey in (("dist", "9"), ("side", "10")):
        ours = json.load(open(os.path.join(R, "supp_%s_paper.json" % mode)))
        for m in MODELS:
            pn = [v for _, v in pap[fkey]["a"][m]]
            pr = [v for _, v in pap[fkey]["b"][m]]
            on = [ours["models"][m][l]["nmse"] for l in LEV[mode]]
            orr = [ours["models"][m][l]["rmse"] for l in LEV[mode]]
            e_pap = float(np.mean([r * r / n for n, r in zip(pn, pr)]))
            rn = float(np.mean(on)) / float(np.mean(pn))
            rr = float(np.mean(orr)) / float(np.mean(pr))
            fac_num = float(np.mean(pn)) / float(np.mean(on))     # 1/rn
            q = rr ** 2 * e_pap / E_OURS_200
            rows.append(dict(mode=mode, model=m, e_pap=round(e_pap, 5),
                             nmse_pap=round(float(np.mean(pn)), 5),
                             nmse_our=round(float(np.mean(on)), 5),
                             rmse_pap=round(float(np.mean(pr)), 5),
                             rmse_our=round(float(np.mean(orr)), 5),
                             r_nmse=round(rn, 3), r_rmse=round(rr, 3),
                             f_model=round(rr ** 2, 3),
                             f_denom=round(e_pap / E_OURS_200, 3),
                             check=round(q / rn, 3)))

    print("%-6s %-10s %8s %8s %8s %7s %7s %7s %7s %6s"
          % ("图", "模型", "NMSE论", "NMSE我", "RMSE我/论", "比值", "模型因子",
             "口径因子", "预测", "校验"))
    for r in rows:
        print("%-6s %-10s %8.5f %8.5f %8.3f %7.3f %7.3f %7.3f %7.3f %6.3f"
              % (r["mode"], r["model"], r["nmse_pap"], r["nmse_our"], r["r_rmse"],
                 r["r_nmse"], r["f_model"], r["f_denom"],
                 r["f_model"] * r["f_denom"], r["check"]))

    print("\n口径因子 = E_paper/E_ours200 = %.3f" % (0.05323 / E_OURS_200))
    print("若换成全集采样 (E=%.5f)，NMSE 会再乘 %.3f"
          % (E_FULL, E_OURS_200 / E_FULL))
    print("  即 Fig.9/10 的 NMSE 会整体降到现在的 %.0f%%"
          % (100 * E_OURS_200 / E_FULL))

    json.dump(rows, open(os.path.join(R, "fig9_10_decomp.json"), "w"),
              indent=1, ensure_ascii=False)
    print("saved results/fig9_10_decomp.json")


if __name__ == "__main__":
    main()
