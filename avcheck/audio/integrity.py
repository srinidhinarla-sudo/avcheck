"""Audio integrity checks: clipping, loudness, silence/dropouts, dynamic range, pitch content."""

import librosa
import numpy as np


def detect_clipping(samples: np.ndarray, sr: int, threshold: float = 0.99, min_run: int = 2) -> dict:
    """Detect hard-clipped regions in a mono audio signal.

    samples: float audio in [-1.0, 1.0] (e.g. from librosa.load)
    sr: sample rate in Hz
    threshold: absolute amplitude considered "at the ceiling"
    min_run: minimum consecutive clipped samples to count as a region
             (filters out single legitimate peak samples)
    """
    is_clipped = np.abs(samples) >= threshold

    # Pad both ends with False so a clipped run touching the start/end
    # of the file still produces a rising/falling edge in the diff.
    padded = np.concatenate(([False], is_clipped, [False]))
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    run_lengths = ends - starts
    keep = run_lengths >= min_run
    regions = [(float(s) / sr, float(e) / sr) for s, e in zip(starts[keep], ends[keep])]

    total = len(samples)
    num_clipped = int(np.sum(is_clipped))

    return {
        "num_clipped_samples": num_clipped,
        "clipped_ratio": num_clipped / total if total else 0.0,
        "num_clipped_regions": len(regions),
        "clipped_regions": regions,
    }


def _windowed_rms_db(x: np.ndarray, window_size: int, eps: float) -> np.ndarray:
    """RMS in dB per non-overlapping window, via reshape (vectorized, no Python loop)."""
    num_windows = len(x) // window_size
    if num_windows == 0:
        return np.array([])
    trimmed = x[: num_windows * window_size].reshape(num_windows, window_size)
    rms = np.sqrt(np.mean(trimmed**2, axis=1))
    return 20 * np.log10(rms + eps)


def compute_loudness_delta(
    ref: np.ndarray, test: np.ndarray, sr: int, window_sec: float = 0.5, eps: float = 1e-9
) -> dict:
    """Compare overall and windowed RMS loudness between reference and test signals.

    Positive delta_db means test is louder than ref; negative means quieter.
    """
    n = min(len(ref), len(test))
    ref, test = ref[:n], test[:n]

    global_rms_ref = float(np.sqrt(np.mean(ref**2))) if n else 0.0
    global_rms_test = float(np.sqrt(np.mean(test**2))) if n else 0.0
    global_delta_db = float(20 * np.log10((global_rms_test + eps) / (global_rms_ref + eps)))

    window_size = max(1, int(window_sec * sr))
    ref_db = _windowed_rms_db(ref, window_size, eps)
    test_db = _windowed_rms_db(test, window_size, eps)
    delta_db = test_db - ref_db
    times = np.arange(len(delta_db)) * window_size / sr

    windowed_deltas_db = [(float(t), float(d)) for t, d in zip(times, delta_db)]
    worst_window = None
    if len(delta_db):
        worst_idx = int(np.argmax(np.abs(delta_db)))
        worst_window = windowed_deltas_db[worst_idx]

    return {
        "global_rms_ref": global_rms_ref,
        "global_rms_test": global_rms_test,
        "global_delta_db": global_delta_db,
        "windowed_deltas_db": windowed_deltas_db,
        "worst_window": worst_window,
    }


def detect_silence_dropouts(
    ref: np.ndarray,
    test: np.ndarray,
    sr: int,
    window_sec: float = 0.05,
    silence_floor_db: float = -50.0,
    drop_db: float = -20.0,
    eps: float = 1e-9,
) -> dict:
    """Flag windows where ref has real content but test's level collapsed relative to it.

    Reusing the same rising/falling-edge run-detection trick as detect_clipping:
    a boolean "is this window a dropout" array gets padded and diffed to find
    contiguous dropout regions in one vectorized pass.
    """
    n = min(len(ref), len(test))
    ref, test = ref[:n], test[:n]
    window_size = max(1, int(window_sec * sr))

    ref_db = _windowed_rms_db(ref, window_size, eps)
    test_db = _windowed_rms_db(test, window_size, eps)
    if len(ref_db) == 0:
        return {"num_dropout_regions": 0, "dropout_regions": []}

    ref_has_content = ref_db > silence_floor_db
    level_drop = test_db - ref_db
    is_dropout = ref_has_content & (level_drop < drop_db)

    padded = np.concatenate(([False], is_dropout, [False]))
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    regions = [
        (float(s) * window_size / sr, float(e) * window_size / sr) for s, e in zip(starts, ends)
    ]

    return {
        "num_dropout_regions": len(regions),
        "dropout_regions": regions,
    }


def compute_dynamic_range_delta(
    ref: np.ndarray,
    test: np.ndarray,
    sr: int,
    window_sec: float = 0.1,
    low_pct: float = 10.0,
    high_pct: float = 95.0,
    eps: float = 1e-9,
) -> dict:
    """Compare the spread between loud and quiet passages (a proxy for compression/limiting).

    Computed as the gap between high_pct and low_pct of windowed RMS-dB values.
    A smaller gap in test than ref means dynamics were squashed (over-compression),
    even if the average loudness (see compute_loudness_delta) is unchanged.
    """
    n = min(len(ref), len(test))
    ref, test = ref[:n], test[:n]
    window_size = max(1, int(window_sec * sr))

    ref_db = _windowed_rms_db(ref, window_size, eps)
    test_db = _windowed_rms_db(test, window_size, eps)
    if len(ref_db) == 0:
        return {"ref_dynamic_range_db": 0.0, "test_dynamic_range_db": 0.0, "dynamic_range_delta_db": 0.0}

    ref_range = float(np.percentile(ref_db, high_pct) - np.percentile(ref_db, low_pct))
    test_range = float(np.percentile(test_db, high_pct) - np.percentile(test_db, low_pct))

    return {
        "ref_dynamic_range_db": ref_range,
        "test_dynamic_range_db": test_range,
        "dynamic_range_delta_db": test_range - ref_range,
    }


def compare_chromagram(ref: np.ndarray, test: np.ndarray, sr: int, hop_length: int = 512, eps: float = 1e-9) -> dict:
    """Compare pitch-class content between ref and test via frame-wise chroma cosine similarity.

    A chromagram folds pitched energy into 12 octave-independent bins (C, C#, ..., B).
    Cosine similarity per frame is 1.0 for identical pitch content and drops toward 0
    as harmonic content diverges, independent of overall loudness.
    """
    chroma_ref = librosa.feature.chroma_stft(y=ref, sr=sr, hop_length=hop_length)
    chroma_test = librosa.feature.chroma_stft(y=test, sr=sr, hop_length=hop_length)

    n_frames = min(chroma_ref.shape[1], chroma_test.shape[1])
    chroma_ref = chroma_ref[:, :n_frames]
    chroma_test = chroma_test[:, :n_frames]

    dot = np.sum(chroma_ref * chroma_test, axis=0)
    norm_ref = np.linalg.norm(chroma_ref, axis=0)
    norm_test = np.linalg.norm(chroma_test, axis=0)
    similarity = dot / (norm_ref * norm_test + eps)

    if n_frames == 0:
        return {"mean_chroma_similarity": 0.0, "worst_frame_similarity": 0.0, "worst_frame_time_sec": 0.0}

    worst_idx = int(np.argmin(similarity))

    return {
        "mean_chroma_similarity": float(np.mean(similarity)),
        "worst_frame_similarity": float(similarity[worst_idx]),
        "worst_frame_time_sec": float(worst_idx * hop_length / sr),
    }
