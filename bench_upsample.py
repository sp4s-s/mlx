import time
import sys

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.nn.layers.upsample import (
        _interpolate,
        _interpolate_separable,
        _linear_indices,
    )
except ImportError as e:
    print(f"Error: {e}")
    print("mlx is not installed in this Python environment.")
    print("Run: pip install -e .  or  pip install mlx")
    sys.exit(1)

try:
    import torch
    import torch.nn.functional as F
    has_torch = True
except ImportError:
    has_torch = False

try:
    import numpy as np
    has_numpy = True
except ImportError:
    has_numpy = False


def make_linear_weight(in_size, out_size, align_corners=False, dtype=mx.float32):
    if in_size == out_size:
        return mx.eye(in_size, dtype=dtype)
    if align_corners:
        scale = (in_size - 1) / max(out_size - 1, 1)
        src = mx.arange(out_size, dtype=mx.float32) * scale
    else:
        scale = in_size / out_size
        src = (mx.arange(out_size, dtype=mx.float32) + 0.5) * scale - 0.5
    src = mx.clip(src, 0, in_size - 1)
    x0 = mx.floor(src).astype(mx.int32)
    x1 = mx.minimum(x0 + 1, in_size - 1)
    w1 = (src - x0.astype(mx.float32)).astype(dtype)
    w0 = (1.0 - w1).astype(dtype)

    w = mx.zeros((out_size, in_size), dtype=dtype)
    rows = mx.arange(out_size)
    w[rows, x0] = w[rows, x0] + w0
    w[rows, x1] = w[rows, x1] + w1
    return w


def matmul_resize_2d(x, scale_factor, align_corners=False):
    n, h, w, c = x.shape
    if isinstance(scale_factor, (list, tuple)):
        sh, sw = scale_factor
    else:
        sh = sw = scale_factor
    out_h = int(h * sh)
    out_w = int(w * sw)
    wh = make_linear_weight(h, out_h, align_corners, x.dtype)
    ww = make_linear_weight(w, out_w, align_corners, x.dtype)
    x = (wh @ x.transpose(1, 0, 2, 3).reshape(h, n * w * c)).reshape(out_h, n, w, c).transpose(1, 0, 2, 3)
    x = (ww @ x.transpose(2, 0, 1, 3).reshape(w, n * out_h * c)).reshape(out_w, n, out_h, c).transpose(1, 2, 0, 3)
    return x


def separable_take_resize(x, scale_factor, align_corners=False):
    dims = x.ndim - 2
    if isinstance(scale_factor, (list, tuple)):
        sf = tuple(scale_factor)
    else:
        sf = (scale_factor,) * dims
    return _interpolate_separable(x, sf, _linear_indices, align_corners=align_corners)


def baseline_upsample(x, scale_factor, align_corners=False):
    dims = x.ndim - 2
    if isinstance(scale_factor, (list, tuple)):
        sf = tuple(scale_factor)
    else:
        sf = (scale_factor,) * dims
    return _interpolate(x, sf, _linear_indices, align_corners=align_corners)


def timed(fn, n=30, warmup=8):
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
        mx.eval(out)
    return (time.perf_counter() - t0) / n * 1000.0


def run_benchmark():
    print("=" * 90)
    print("MLX Upsample Performance & Accuracy Analysis")
    print(f"Device: {mx.default_device()}")
    print("=" * 90)

    test_configs = [
        # (N, H, W, C, scale, align_corners, label)
        (1, 192, 320, 4, 4.0, False, "192x320 -> 768x1280 (4x, RIFE shape)"),
        (1, 384, 640, 4, 2.0, False, "384x640 -> 768x1280 (2x)"),
        (1, 270, 480, 3, 2.0, False, "270x480 -> 540x960 (2x, 3ch)"),
        (1, 540, 960, 3, 2.0, False, "540x960 -> 1080x1920 (2x, 1080p, 3ch)"),
        (2, 64, 64, 32, 2.0, False, "64x64 -> 128x128 (2x, batch=2, C=32)"),
        (1, 768, 1280, 4, 0.5, False, "768x1280 -> 384x640 (0.5x downsample)"),
        (1, 768, 1280, 4, 0.25, False, "768x1280 -> 192x320 (0.25x downsample)"),
        (1, 192, 320, 4, 4.0, True, "192x320 -> 768x1280 (4x, align_corners=True)"),
        (1, 384, 640, 4, 2.0, True, "384x640 -> 768x1280 (2x, align_corners=True)"),
    ]

    print("\n--- COMPILED MODE (mx.compile) ---")
    header = (
        f"{'Configuration':<42} | {'Baseline':>9} | {'Separable':>10} | "
        f"{'Matmul':>9} | {'Sep vs Base':>11} | {'MM vs Base':>10} | {'Max Diff':>9}"
    )
    print(header)
    print("-" * len(header))

    for N, H, W, C, scale, ac, label in test_configs:
        mx.random.seed(42)
        x = mx.random.normal((N, H, W, C)).astype(mx.float32)

        fn_base = mx.compile(lambda a: baseline_upsample(a, scale, align_corners=ac))
        fn_sep = mx.compile(lambda a: separable_take_resize(a, scale, align_corners=ac))
        fn_mm = mx.compile(lambda a: matmul_resize_2d(a, scale, align_corners=ac))

        out_base = fn_base(x)
        out_sep = fn_sep(x)
        out_mm = fn_mm(x)
        mx.eval(out_base, out_sep, out_mm)

        diff_sep = float(mx.abs(out_base - out_sep).max())
        diff_mm = float(mx.abs(out_base - out_mm).max())

        t_base = timed(lambda: fn_base(x))
        t_sep = timed(lambda: fn_sep(x))
        t_mm = timed(lambda: fn_mm(x))

        sep_speedup = f"{t_base / t_sep:5.2f}x"
        mm_speedup = f"{t_base / t_mm:5.2f}x"

        print(
            f"{label:<42} | {t_base:7.3f}ms | {t_sep:8.3f}ms | {t_mm:7.3f}ms | "
            f"{sep_speedup:>11} | {mm_speedup:>10} | {max(diff_sep, diff_mm):9.2e}"
        )

    print("\n--- EAGER MODE ---")
    print(header)
    print("-" * len(header))

    for N, H, W, C, scale, ac, label in test_configs:
        mx.random.seed(42)
        x = mx.random.normal((N, H, W, C)).astype(mx.float32)

        t_base = timed(lambda: baseline_upsample(x, scale, align_corners=ac))
        t_sep = timed(lambda: separable_take_resize(x, scale, align_corners=ac))
        t_mm = timed(lambda: matmul_resize_2d(x, scale, align_corners=ac))

        sep_speedup = f"{t_base / t_sep:5.2f}x"
        mm_speedup = f"{t_base / t_mm:5.2f}x"

        out_base = baseline_upsample(x, scale, align_corners=ac)
        out_mm = matmul_resize_2d(x, scale, align_corners=ac)
        diff_mm = float(mx.abs(out_base - out_mm).max())

        print(
            f"{label:<42} | {t_base:7.3f}ms | {t_sep:8.3f}ms | {t_mm:7.3f}ms | "
            f"{sep_speedup:>11} | {mm_speedup:>10} | {diff_mm:9.2e}"
        )

    print("\n--- 1D and 3D DIMENSIONAL CHECKS (Separable) ---")
    x1 = mx.random.normal((1, 512, 4))
    out1_base = baseline_upsample(x1, 2.0)
    out1_sep = separable_take_resize(x1, 2.0)
    mx.eval(out1_base, out1_sep)
    diff1 = float(mx.abs(out1_base - out1_sep).max())
    print(f"1D (512 -> 1024): separable vs baseline diff = {diff1:.2e}")

    x3 = mx.random.normal((1, 16, 32, 32, 4))
    out3_base = baseline_upsample(x3, 2.0)
    out3_sep = separable_take_resize(x3, 2.0)
    mx.eval(out3_base, out3_sep)
    diff3 = float(mx.abs(out3_base - out3_sep).max())
    print(f"3D (16x32x32 -> 32x64x64): separable vs baseline diff = {diff3:.2e}")

    if has_torch and has_numpy:
        print("\n--- PYTORCH BIT-ACCURACY CHECK ---")
        for N, H, W, C, scale, ac, label in test_configs[:4]:
            x_mx = mx.random.normal((N, H, W, C)).astype(mx.float32)
            x_np = np.array(x_mx)
            x_pt = torch.from_numpy(x_np.transpose(0, 3, 1, 2))
            out_pt = F.interpolate(
                x_pt,
                scale_factor=scale,
                mode="bilinear",
                align_corners=ac,
            ).permute(0, 2, 3, 1).numpy()
            out_sep = np.array(separable_take_resize(x_mx, scale, align_corners=ac))
            out_mm = np.array(matmul_resize_2d(x_mx, scale, align_corners=ac))
            diff_pt_sep = float(np.abs(out_pt - out_sep).max())
            diff_pt_mm = float(np.abs(out_pt - out_mm).max())
            print(f"{label}: diff vs PyTorch -> sep={diff_pt_sep:.2e}, matmul={diff_pt_mm:.2e}")


if __name__ == "__main__":
    run_benchmark()
