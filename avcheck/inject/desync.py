"""A/V desync injector: shifts the audio track relative to video's shared timeline."""

import numpy as np


def inject_av_desync(samples: np.ndarray, sr: int, offset_ms: float = 200.0) -> tuple:
    """Shift audio by offset_ms while preserving total duration (video is untouched).

    offset_ms > 0: audio lags video ("audio_delayed") — pad silence at the start
    and trim the same amount off the end.
    offset_ms < 0: audio leads video ("audio_advanced") — trim from the start
    and pad silence at the end.
    """
    offset_samples = min(int(round(abs(offset_ms) / 1000.0 * sr)), len(samples))

    if offset_ms >= 0:
        pad = np.zeros(offset_samples, dtype=samples.dtype)
        shifted = np.concatenate([pad, samples[: len(samples) - offset_samples]])
        direction = "audio_delayed"
    else:
        pad = np.zeros(offset_samples, dtype=samples.dtype)
        shifted = np.concatenate([samples[offset_samples:], pad])
        direction = "audio_advanced"

    ground_truth = {
        "defect_type": "av_desync",
        "media": "audio",
        "sr": sr,
        "params": {"offset_ms": offset_ms, "direction": direction},
        "events": [{"offset_ms": offset_ms, "direction": direction}],
    }
    return shifted, ground_truth
