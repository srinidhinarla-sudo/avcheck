import numpy as np
import pytest

SR = 22050


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
