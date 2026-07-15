"""Audio-side defect injectors. Each returns (new_samples, ground_truth_dict)."""

import numpy as np


def inject_audio_clipping(
    samples: np.ndarray, sr: int, gain: float = 3.0, num_events: int = 1, duration_sec: float = 0.3, seed: int = 0
) -> tuple:
    """Overdrive short regions by a fixed gain then hard-limit to [-1, 1]."""
    rng = np.random.default_rng(seed)
    n = len(samples)
    duration_samples = int(duration_sec * sr)
    max_start = n - duration_samples - 1
    if max_start <= 0:
        raise ValueError("Audio too short for requested clipping duration")
    starts = sorted(rng.integers(0, max_start, size=num_events).tolist())

    new_samples = samples.copy()
    events = []
    for start in starts:
        end = start + duration_samples
        new_samples[start:end] = np.clip(new_samples[start:end] * gain, -1.0, 1.0)
        events.append({"start_sec": start / sr, "end_sec": end / sr})

    return new_samples, {
        "defect_type": "audio_clipping",
        "media": "audio",
        "sr": sr,
        "params": {"gain": gain},
        "events": events,
    }


def inject_audio_dropouts(
    samples: np.ndarray, sr: int, num_events: int = 1, duration_sec: float = 0.3, seed: int = 0
) -> tuple:
    """Zero out short regions (simulates a buffer underrun / dropped packet)."""
    rng = np.random.default_rng(seed)
    n = len(samples)
    duration_samples = int(duration_sec * sr)
    max_start = n - duration_samples - 1
    if max_start <= 0:
        raise ValueError("Audio too short for requested dropout duration")
    starts = sorted(rng.integers(0, max_start, size=num_events).tolist())

    new_samples = samples.copy()
    events = []
    for start in starts:
        end = start + duration_samples
        new_samples[start:end] = 0.0
        events.append({"start_sec": start / sr, "end_sec": end / sr})

    return new_samples, {"defect_type": "audio_dropout", "media": "audio", "sr": sr, "events": events}
