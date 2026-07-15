import cv2
import numpy as np
import pytest

SR = 22050
FRAME_SIZE = (64, 64)  # (width, height)


def sine(freq: float, duration_sec: float, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(duration_sec * sr)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def sr() -> int:
    return SR


@pytest.fixture
def ref_tone() -> np.ndarray:
    """3 seconds of a 440Hz tone (A4) at moderate amplitude — the 'reference' signal."""
    return sine(440.0, 3.0, amplitude=0.5)


def make_checkerboard_frames(num_frames: int, size=FRAME_SIZE, seed: int = 0) -> list:
    """Synthetic frames with per-pixel texture (not flat color) so PSNR/SSIM are meaningful."""
    rng = np.random.default_rng(seed)
    width, height = size
    frames = []
    for i in range(num_frames):
        base = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        shift = np.roll(base, shift=i * 2, axis=1)
        frames.append(shift)
    return frames


def write_video(path: str, frames: list, fps: int = 10) -> None:
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
