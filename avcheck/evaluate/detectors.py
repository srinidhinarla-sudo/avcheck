"""One detector per injected defect class. Each returns {"defect_type", "events", ...}."""

import cv2
import librosa
import numpy as np

from avcheck.audio.integrity import detect_clipping as _detect_clipping_core
from avcheck.audio.integrity import detect_silence_dropouts as _detect_dropouts_core


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def detect_frame_drops(ref_frames: list, test_frames: list, fps: float, z_thresh: float = 2.5) -> dict:
    """Two corroborating signals:

    - timestamp gap: match each test frame to its nearest reference frame (min
      MSE) by content; a jump of >1 in that matched index between consecutive
      test frames means a reference frame was skipped.
    - frame-diff spike: frame-to-frame difference within the test clip itself
      spikes at a drop point, since the two now-adjacent frames are further
      apart in "real" motion than a normal step.
    """
    n_test = len(test_frames)

    matched_ref_idx = []
    for tf in test_frames:
        errors = [np.mean((tf.astype(np.int32) - rf.astype(np.int32)) ** 2) for rf in ref_frames]
        matched_ref_idx.append(int(np.argmin(errors)))
    matched_ref_idx = np.array(matched_ref_idx)
    gap_at = set(int(i) for i in np.where(np.diff(matched_ref_idx) > 1)[0] + 1)

    diffs = np.array([_mean_abs_diff(test_frames[i - 1], test_frames[i]) for i in range(1, n_test)])
    spike_at = set()
    if len(diffs) > 1 and diffs.std() > 1e-9:
        z = (diffs - diffs.mean()) / diffs.std()
        spike_at = set(int(i) for i in np.where(z > z_thresh)[0] + 1)

    predicted_indices = sorted(gap_at | spike_at)
    events = [{"frame_index": i, "timestamp_sec": i / fps} for i in predicted_indices]
    return {"defect_type": "frame_drop", "events": events}


def detect_frame_freezes(test_frames: list, fps: float, ssim_threshold: float = 0.98, min_run: int = 1) -> dict:
    """Flag runs of consecutive test frames that are near-identical (SSIM ~1),
    via the same rising/falling-edge run-length trick as detect_clipping."""
    from skimage.metrics import structural_similarity as ssim

    n = len(test_frames)
    if n < 2:
        return {"defect_type": "frame_freeze", "events": []}

    is_frozen = np.array(
        [ssim(_gray(test_frames[i - 1]), _gray(test_frames[i]), data_range=255) >= ssim_threshold for i in range(1, n)]
    )

    padded = np.concatenate(([False], is_frozen, [False]))
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    events = []
    for s, e in zip(starts, ends):
        if e - s >= min_run:
            events.append(
                {"start_frame": int(s), "end_frame": int(e), "start_timestamp_sec": s / fps, "end_timestamp_sec": e / fps}
            )
    return {"defect_type": "frame_freeze", "events": events}


def detect_black_frames(test_frames: list, fps: float, luminance_threshold: float = 10.0, min_run: int = 1) -> dict:
    """Flag runs of frames whose mean luminance falls below threshold."""
    means = np.array([_gray(f).mean() for f in test_frames])
    is_black = means < luminance_threshold

    padded = np.concatenate(([False], is_black, [False]))
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    events = []
    for s, e in zip(starts, ends):
        if e - s >= min_run:
            events.append(
                {
                    "start_frame": int(s),
                    "end_frame": int(e - 1),
                    "start_timestamp_sec": s / fps,
                    "end_timestamp_sec": (e - 1) / fps,
                }
            )
    return {"defect_type": "black_frame", "events": events}


def detect_banding(ref_frames: list, test_frames: list, fps: float, drop_ratio_threshold: float = 0.3) -> dict:
    """Flag banding via loss of local tonal gradation: count unique luminance
    values per frame and compare test vs reference. Quantization/banding
    collapses many original tonal steps into fewer discrete levels, showing up
    as a large relative drop in unique-value count.

    This is a practical proxy for classic "block-artifact energy" — a true
    DCT-block-edge metric needs the encoder's block grid, unavailable post-decode.
    Documented as a limitation.
    """
    n = min(len(ref_frames), len(test_frames))
    if n == 0:
        return {"defect_type": "banding", "events": [], "score": 0.0}

    ref_unique = np.array([len(np.unique(_gray(ref_frames[i]))) for i in range(n)])
    test_unique = np.array([len(np.unique(_gray(test_frames[i]))) for i in range(n)])
    mean_drop = float(np.mean((ref_unique - test_unique) / np.maximum(ref_unique, 1)))

    events = []
    if mean_drop > drop_ratio_threshold:
        duration = (n - 1) / fps
        events.append({"start_frame": 0, "end_frame": n - 1, "start_timestamp_sec": 0.0, "end_timestamp_sec": duration})

    return {"defect_type": "banding", "events": events, "score": mean_drop}


def detect_color_shift(ref_frames: list, test_frames: list, fps: float, distance_threshold: float = 0.3) -> dict:
    """Flag a global color-pipeline defect via Bhattacharyya distance between
    hue histograms of reference and test frames, averaged across the clip."""
    n = min(len(ref_frames), len(test_frames))
    if n == 0:
        return {"defect_type": "color_shift", "events": [], "score": 0.0}

    distances = []
    for i in range(n):
        ref_hsv = cv2.cvtColor(ref_frames[i], cv2.COLOR_BGR2HSV)
        test_hsv = cv2.cvtColor(test_frames[i], cv2.COLOR_BGR2HSV)
        ref_hist = cv2.calcHist([ref_hsv], [0], None, [180], [0, 180])
        test_hist = cv2.calcHist([test_hsv], [0], None, [180], [0, 180])
        cv2.normalize(ref_hist, ref_hist)
        cv2.normalize(test_hist, test_hist)
        distances.append(cv2.compareHist(ref_hist, test_hist, cv2.HISTCMP_BHATTACHARYYA))

    mean_distance = float(np.mean(distances))
    events = []
    if mean_distance > distance_threshold:
        duration = (n - 1) / fps
        events.append({"start_frame": 0, "end_frame": n - 1, "start_timestamp_sec": 0.0, "end_timestamp_sec": duration})

    return {"defect_type": "color_shift", "events": events, "score": mean_distance}


def _correlate_at_lag(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """Correlation between a[i] and b[i + lag] over their valid overlap."""
    n = len(a)
    if lag >= 0:
        a_seg = a[: n - lag] if lag > 0 else a
        b_seg = b[lag:]
    else:
        a_seg = a[-lag:]
        b_seg = b[: n + lag]
    m = min(len(a_seg), len(b_seg))
    if m < 5:
        return 0.0
    a_seg, b_seg = a_seg[:m], b_seg[:m]
    denom = np.std(a_seg) * np.std(b_seg) + 1e-9
    return float(np.mean(a_seg * b_seg) / denom)


def detect_av_desync(
    ref_frames: list,
    test_audio: np.ndarray,
    sr: int,
    fps: float,
    max_lag_ms: float = 400.0,
    min_offset_ms: float = 50.0,
    confidence_threshold: float = 0.15,
) -> dict:
    """Estimate A/V offset by cross-correlating video motion energy against audio
    onset strength. Reference video is assumed untouched by desync injection, so
    its own frame-diff motion envelope stands in for "true" event timing; the
    lag that best aligns it with the test audio's onset envelope is the
    estimated desync offset. Positive offset = audio lags behind video.
    """
    motion = np.array([_mean_abs_diff(ref_frames[i - 1], ref_frames[i]) for i in range(1, len(ref_frames))])
    motion_times = np.arange(1, len(ref_frames)) / fps

    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=test_audio, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)

    if len(motion_times) < 2 or len(onset_times) < 2:
        return {"defect_type": "av_desync", "events": [], "estimated_offset_ms": 0.0, "confidence": 0.0}

    duration = min(motion_times[-1], onset_times[-1])
    step_sec = 0.01
    grid = np.arange(0, duration, step_sec)
    motion_grid = np.interp(grid, motion_times, motion)
    onset_grid = np.interp(grid, onset_times, onset_env)
    motion_grid = motion_grid - motion_grid.mean()
    onset_grid = onset_grid - onset_grid.mean()

    max_lag_steps = int(max_lag_ms / 1000 / step_sec)
    lags = list(range(-max_lag_steps, max_lag_steps + 1))
    correlations = [_correlate_at_lag(motion_grid, onset_grid, lag) for lag in lags]

    best_idx = int(np.argmax(correlations))
    best_lag = lags[best_idx]
    confidence = correlations[best_idx]
    offset_ms = best_lag * step_sec * 1000

    events = []
    if abs(offset_ms) >= min_offset_ms and confidence >= confidence_threshold:
        direction = "audio_delayed" if offset_ms > 0 else "audio_advanced"
        events.append({"offset_ms": offset_ms, "direction": direction})

    return {"defect_type": "av_desync", "events": events, "estimated_offset_ms": offset_ms, "confidence": confidence}


def _merge_close_intervals(intervals: list, gap_sec: float) -> list:
    """Coalesce nearby intervals into one. An oscillating waveform clipped at its
    peaks produces many short physically-clipped bursts (one per half-cycle)
    rather than one continuous run; merging bursts separated by less than
    gap_sec reports "one clipping incident" the way a human reviewer would,
    rather than hundreds of scalloped micro-events.
    """
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] <= gap_sec:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def detect_audio_clipping(test_samples: np.ndarray, sr: int, merge_gap_sec: float = 0.02) -> dict:
    """Reuses Phase 1's sample-ceiling run detector, then coalesces nearby bursts."""
    result = _detect_clipping_core(test_samples, sr)
    merged = _merge_close_intervals(result["clipped_regions"], merge_gap_sec)
    events = [{"start_sec": s, "end_sec": e} for s, e in merged]
    return {"defect_type": "audio_clipping", "events": events}


def detect_audio_dropouts(ref_samples: np.ndarray, test_samples: np.ndarray, sr: int) -> dict:
    """Reuses Phase 1's ref-vs-test energy-gap detector directly."""
    result = _detect_dropouts_core(ref_samples, test_samples, sr)
    events = [{"start_sec": s, "end_sec": e} for s, e in result["dropout_regions"]]
    return {"defect_type": "audio_dropout", "events": events}
