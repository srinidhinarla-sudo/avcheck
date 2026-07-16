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


def make_gradient_frames(num_frames: int, size=FRAME_SIZE) -> list:
    """Smooth horizontal gradient frames — realistic content for banding, since
    banding is specifically about loss of smooth tonal gradation."""
    width, height = size
    x = np.linspace(0, 255, width)
    return [np.stack([np.tile(x, (height, 1)).astype(np.uint8)] * 3, axis=-1) for _ in range(num_frames)]


def make_demo_frames(num_frames: int, fps: float, size=FRAME_SIZE, flash_interval_sec: float = 1.0) -> list:
    """A moving smooth gradient with periodic brightness 'flash' frames.

    Smoothness makes banding/color-shift detectable (unlike pure noise); frame
    motion (the shift) makes frame-diff-based detectors meaningful; the flashes
    give A/V-desync detection a real visual event to correlate against audio clicks.
    """
    width, height = size
    x = np.linspace(0, 255, width)
    frames = []
    for i in range(num_frames):
        shift = (i * 3) % width
        grad_row = np.roll(x, shift).astype(np.uint8)
        frame = np.stack([np.tile(grad_row, (height, 1))] * 3, axis=-1)
        t = i / fps
        if (t % flash_interval_sec) < (1.0 / fps):
            frame = np.clip(frame.astype(int) + 150, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def make_demo_audio(duration_sec: float, sr: int = SR, flash_interval_sec: float = 1.0, tone_freq: float = 440.0) -> np.ndarray:
    """A quiet tone with periodic click bursts timed to match make_demo_frames' flashes."""
    t = np.arange(int(duration_sec * sr)) / sr
    audio = (0.2 * np.sin(2 * np.pi * tone_freq * t)).astype(np.float64)
    rng = np.random.default_rng(0)
    click_dur = 0.05
    for click_time in np.arange(0, duration_sec, flash_interval_sec):
        start = int(click_time * sr)
        end = min(int((click_time + click_dur) * sr), len(audio))
        audio[start:end] += np.clip(0.3 * rng.standard_normal(end - start), -0.5, 0.5)
    return audio.astype(np.float32)
