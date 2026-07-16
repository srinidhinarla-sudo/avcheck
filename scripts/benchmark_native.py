#!/usr/bin/env python3
"""Benchmark: naive Python loop vs NumPy-vectorized vs C++ (pybind11) tone-curve
application, on a realistic HD frame. Answers the question a C++/Python
boundary is supposed to answer: how much does moving the hot loop into C++
actually buy you, versus just using NumPy?
"""

import time

import numpy as np

import avcheck_native


def build_lut(brightness: float, gamma: float) -> np.ndarray:
    i = np.arange(256, dtype=np.float64)
    normalized = i / 255.0
    gamma_corrected = np.power(normalized, 1.0 / gamma)
    brightened = gamma_corrected * 255.0 + brightness * 255.0
    return np.clip(brightened, 0, 255).astype(np.uint8)


def apply_naive_python(frame: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """The realistic 'no C++, no NumPy tricks' baseline: a per-pixel Python loop."""
    height, width, channels = frame.shape
    out = np.empty_like(frame)
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                out[y, x, c] = lut[frame[y, x, c]]
    return out


def apply_numpy(frame: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return lut[frame]


def apply_cpp(frame: np.ndarray, filt: avcheck_native.ToneCurveFilter) -> np.ndarray:
    return filt.apply_frame(frame)


def timeit(fn, *args, repeats: int) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    return (time.perf_counter() - start) / repeats


def main() -> None:
    rng = np.random.default_rng(0)
    brightness, gamma = 0.1, 1.2
    lut = build_lut(brightness, gamma)
    filt = avcheck_native.ToneCurveFilter(brightness, gamma)

    print(f"{'resolution':<14}{'naive python':>16}{'numpy':>14}{'C++ (pybind11)':>18}{'C++ vs naive':>16}{'C++ vs numpy':>16}")
    for label, (h, w) in [("320x180", (180, 320)), ("640x360", (360, 640)), ("1280x720", (720, 1280))]:
        frame = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)

        naive_repeats = 3 if (h * w) < 100_000 else 1
        naive_s = timeit(apply_naive_python, frame, lut, repeats=naive_repeats)
        numpy_s = timeit(apply_numpy, frame, lut, repeats=20)
        cpp_s = timeit(apply_cpp, frame, filt, repeats=20)

        print(
            f"{label:<14}{naive_s * 1000:>13.2f}ms{numpy_s * 1000:>11.3f}ms"
            f"{cpp_s * 1000:>15.3f}ms{naive_s / cpp_s:>13.0f}x{numpy_s / cpp_s:>13.1f}x"
        )


if __name__ == "__main__":
    main()
