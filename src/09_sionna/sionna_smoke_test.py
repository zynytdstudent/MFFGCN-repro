#!/usr/bin/env python
"""Sionna RT 环境自检 —— 确认光追后端（CUDA/OptiX 或 LLVM）可用。

用法（两端相同，激活各自 venv 后）：
    ~/venvs/sionna/bin/python sionna_smoke_test.py            # zjut148 是 /public2/czz/ZhangYu/venvs
    ~/venvs/sionna/bin/python sionna_smoke_test.py --cpu      # 强制走 CPU(LLVM) 后端

退出码 0 = 通过；1 = 后端不可用（会打印原因与所需驱动/库版本）。
"""
import argparse
import sys


def try_variant(prefer_cpu: bool) -> str:
    import mitsuba as mi

    order = ["llvm_ad_mono_polarized", "cuda_ad_mono_polarized"]
    if not prefer_cpu:
        order.reverse()
    errors = []
    for v in order:
        try:
            mi.set_variant(v)
            # set_variant 只是选后端，真正的初始化发生在第一次建场景时：
            # GPU 后端在这里才会去加载 OptiX，CPU 后端才会去加载 libLLVM。
            mi.load_dict({"type": "scene"})
            return v
        except Exception as e:  # noqa: BLE001
            errors.append(f"{v}: {type(e).__name__}: {e}")
    print("[fail] 没有可用的 mitsuba 后端：")
    for e in errors:
        print("   -", e)
    print("\n提示：")
    print("  GPU 后端需要 OptiX 8，即 NVIDIA 驱动 >= R535（当前机器低于此版本会被拒）。")
    print("  CPU 后端需要 libLLVM.so >= 11，并设置 DRJIT_LIBLLVM_PATH=<含 libLLVM 的目录>。")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="优先使用 LLVM(CPU) 后端")
    args = ap.parse_args()

    import numpy as np
    import mitsuba as mi
    import sionna.rt as rt

    print("python  :", sys.version.split()[0])
    print("mitsuba :", mi.__version__)
    print("sionna  :", rt.__version__ if hasattr(rt, "__version__") else "2.0.0")

    variant = try_variant(args.cpu)
    print("backend :", variant)

    scene = rt.load_scene(rt.scene.simple_street_canyon)
    scene.frequency = 2.14e9
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    vertical_spacing=0.5, horizontal_spacing=0.5,
                                    pattern="tr38901", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    vertical_spacing=0.5, horizontal_spacing=0.5,
                                    pattern="tr38901", polarization="V")
    scene.add(rt.Transmitter(name="tx", position=[-33, 1, 27]))
    scene.add(rt.Receiver(name="rx", position=[-10, 1, 1.5]))

    solver = rt.PathSolver(max_depth=3, samples_per_src=10 ** 5, seed=42)
    paths = solver(scene)

    a = paths.a[0].numpy()
    tau = paths.tau[0].numpy()
    keep = np.isfinite(a[:, 0, 0, 0].real) & (tau[:, 0, 0] > 0)
    print("paths   :", a.shape, "valid:", int(keep.sum()))
    print("gain[0] :", np.round(np.abs(a[keep, 0, 0, 0])[:3], 8))
    print("SIONNA_RT_SELFTEST_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
