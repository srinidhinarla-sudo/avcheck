"""Cross-detector evaluation: run every detector against every injected variant.

For each variant, the detector matching its defect class is scored for TP/FN
(recall) using its own ground truth; every *other* detector run on that same
variant contributes only false positives (it has no ground truth there, so any
event it reports is a false alarm for its own class). Aggregating across all
8 variants gives one precision/recall pair per defect class that reflects both
localization accuracy and cross-class false-alarm rate.
"""

import json
import os

from avcheck.evaluate.detectors import (
    detect_audio_clipping,
    detect_audio_dropouts,
    detect_av_desync,
    detect_banding,
    detect_black_frames,
    detect_color_shift,
    detect_frame_drops,
    detect_frame_freezes,
)
from avcheck.evaluate.matching import match_events
from avcheck.inject.media_io import read_audio, read_video_frames

DEFECT_CLASSES = [
    "frame_drop",
    "frame_freeze",
    "black_frames",
    "banding",
    "color_shift",
    "audio_clipping",
    "audio_dropout",
    "av_desync",
]

VIDEO_DEFECTS = {"frame_drop", "frame_freeze", "black_frames", "banding", "color_shift"}


def _run_detector(name: str, ref_frames: list, test_frames: list, fps: float, ref_audio, test_audio, sr: int) -> dict:
    if name == "frame_drop":
        return detect_frame_drops(ref_frames, test_frames, fps)
    if name == "frame_freeze":
        return detect_frame_freezes(test_frames, fps)
    if name == "black_frames":
        return detect_black_frames(test_frames, fps)
    if name == "banding":
        return detect_banding(ref_frames, test_frames, fps)
    if name == "color_shift":
        return detect_color_shift(ref_frames, test_frames, fps)
    if name == "audio_clipping":
        return detect_audio_clipping(test_audio, sr)
    if name == "audio_dropout":
        return detect_audio_dropouts(ref_audio, test_audio, sr)
    if name == "av_desync":
        return detect_av_desync(ref_frames, test_audio, sr, fps)
    raise ValueError(f"Unknown detector: {name}")


def evaluate_variants(manifest_path: str, ref_video_path: str, ref_audio_path: str) -> dict:
    """Run all 8 detectors against all 8 variants and compute precision/recall per class."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    ref_frames, fps = read_video_frames(ref_video_path)
    ref_audio, sr = read_audio(ref_audio_path)
    clip_duration = (len(ref_frames) - 1) / fps if ref_frames else 0.0

    variants_by_name = {v["name"]: v for v in manifest["variants"]}
    counts = {name: {"tp": 0, "fp": 0, "fn": 0} for name in DEFECT_CLASSES}

    for variant_name, variant in variants_by_name.items():
        is_video_variant = variant_name in VIDEO_DEFECTS
        test_frames = read_video_frames(variant["file"])[0] if is_video_variant else ref_frames

        if "audio_file" in variant:
            test_audio, test_sr = read_audio(variant["audio_file"])
        else:
            test_audio, test_sr = ref_audio, sr

        with open(variant["ground_truth"]) as f:
            ground_truth = json.load(f)

        for detector_name in DEFECT_CLASSES:
            predicted = _run_detector(detector_name, ref_frames, test_frames, fps, ref_audio, test_audio, test_sr)
            gt_events = ground_truth["events"] if detector_name == variant_name else []
            result = match_events(predicted["events"], gt_events, clip_duration)
            counts[detector_name]["tp"] += result["tp"]
            counts[detector_name]["fp"] += result["fp"]
            counts[detector_name]["fn"] += result["fn"]

    report = {}
    for name, c in counts.items():
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        report[name] = {**c, "precision": precision, "recall": recall}

    return report


def write_evaluation_csv(report: dict, output_path: str) -> None:
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["defect_class", "tp", "fp", "fn", "precision", "recall"])
        for name, r in report.items():
            writer.writerow([name, r["tp"], r["fp"], r["fn"], f"{r['precision']:.3f}", f"{r['recall']:.3f}"])


def write_evaluation_json(report: dict, output_path: str) -> None:
    import json

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
