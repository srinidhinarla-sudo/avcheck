"""Python wrapper around the C++ tone-curve filter (avcheck_native, pybind11)."""

import avcheck_native

from avcheck.inject.media_io import read_video_frames, write_video_frames


class ToneCurveFilter:
    """Brightness/gamma frame filter backed by the C++ ToneCurveFilter class."""

    def __init__(self, brightness: float = 0.0, gamma: float = 1.0, skip_clamp: bool = False):
        self._impl = avcheck_native.ToneCurveFilter(brightness, gamma, skip_clamp)

    @property
    def brightness(self) -> float:
        return self._impl.brightness

    @property
    def gamma(self) -> float:
        return self._impl.gamma

    @property
    def skip_clamp(self) -> bool:
        return self._impl.skip_clamp

    def apply(self, frame):
        return self._impl.apply_frame(frame)


def process_video(input_path: str, output_path: str, brightness: float = 0.0, gamma: float = 1.0, skip_clamp: bool = False) -> None:
    """Run every frame of input_path through the C++ tone-curve filter and write output_path."""
    frames, fps = read_video_frames(input_path)
    filt = ToneCurveFilter(brightness, gamma, skip_clamp)
    processed = [filt.apply(f) for f in frames]
    write_video_frames(processed, fps, output_path)
