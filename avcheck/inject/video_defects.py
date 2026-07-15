"""Video-side defect injectors. Each returns (new_frames, ground_truth_dict)."""

import cv2
import numpy as np


def inject_frame_drops(frames: list, fps: float, num_drops: int = 3, seed: int = 0) -> tuple:
    """Remove whole frames at random interior points, shortening the clip."""
    rng = np.random.default_rng(seed)
    n = len(frames)
    if num_drops >= n - 2:
        raise ValueError("num_drops must leave at least 2 frames untouched")
    drop_indices = sorted(rng.choice(np.arange(1, n - 1), size=num_drops, replace=False).tolist())

    drop_set = set(drop_indices)
    new_frames = [f for i, f in enumerate(frames) if i not in drop_set]
    events = [{"frame_index": i, "timestamp_sec": i / fps} for i in drop_indices]

    return new_frames, {"defect_type": "frame_drop", "media": "video", "fps": fps, "events": events}


def inject_frame_freezes(frames: list, fps: float, num_freezes: int = 1, freeze_len: int = 5, seed: int = 0) -> tuple:
    """Repeat a single frame for freeze_len frames starting at a random point (a stuck-frame defect)."""
    rng = np.random.default_rng(seed)
    n = len(frames)
    max_start = n - freeze_len - 1
    if max_start <= 0:
        raise ValueError("Video too short for requested freeze length")
    starts = sorted(rng.choice(np.arange(1, max_start), size=num_freezes, replace=False).tolist())

    new_frames = list(frames)
    events = []
    for start in starts:
        frozen = new_frames[start]
        for offset in range(1, freeze_len):
            new_frames[start + offset] = frozen.copy()
        events.append(
            {
                "start_frame": start,
                "end_frame": start + freeze_len - 1,
                "start_timestamp_sec": start / fps,
                "end_timestamp_sec": (start + freeze_len - 1) / fps,
            }
        )

    return new_frames, {"defect_type": "frame_freeze", "media": "video", "fps": fps, "events": events}


def inject_black_frames(frames: list, fps: float, num_events: int = 1, duration_frames: int = 5, seed: int = 0) -> tuple:
    """Replace a run of frames with solid black (simulates a decoder stall / signal loss)."""
    rng = np.random.default_rng(seed)
    n = len(frames)
    max_start = n - duration_frames - 1
    if max_start <= 0:
        raise ValueError("Video too short for requested black-frame duration")
    starts = sorted(rng.choice(np.arange(1, max_start), size=num_events, replace=False).tolist())

    new_frames = list(frames)
    events = []
    for start in starts:
        for offset in range(duration_frames):
            new_frames[start + offset] = np.zeros_like(frames[0])
        events.append(
            {
                "start_frame": start,
                "end_frame": start + duration_frames - 1,
                "start_timestamp_sec": start / fps,
                "end_timestamp_sec": (start + duration_frames - 1) / fps,
            }
        )

    return new_frames, {"defect_type": "black_frame", "media": "video", "fps": fps, "events": events}


def inject_banding(frames: list, fps: float, levels: int = 8) -> tuple:
    """Quantize pixel values to `levels` steps across the whole clip, producing
    visible banding in smooth gradients (a classic low-bitrate encoding artifact)."""
    step = 256 // levels
    new_frames = [((f.astype(np.uint16) // step) * step).astype(np.uint8) for f in frames]
    duration = (len(frames) - 1) / fps if frames else 0.0

    ground_truth = {
        "defect_type": "banding",
        "media": "video",
        "fps": fps,
        "params": {"levels": levels},
        "events": [{"start_frame": 0, "end_frame": len(frames) - 1, "start_timestamp_sec": 0.0, "end_timestamp_sec": duration}],
    }
    return new_frames, ground_truth


def inject_color_shift(frames: list, fps: float, hue_shift: int = 40) -> tuple:
    """Rotate hue in HSV space across the whole clip (simulates a color-pipeline/LUT bug)."""
    new_frames = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
        new_frames.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))

    duration = (len(frames) - 1) / fps if frames else 0.0
    ground_truth = {
        "defect_type": "color_shift",
        "media": "video",
        "fps": fps,
        "params": {"hue_shift": hue_shift},
        "events": [{"start_frame": 0, "end_frame": len(frames) - 1, "start_timestamp_sec": 0.0, "end_timestamp_sec": duration}],
    }
    return new_frames, ground_truth
