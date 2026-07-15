"""Orchestrates all defect injectors into one labeled variant per defect class."""

import json
import os

from avcheck.inject.audio_defects import inject_audio_clipping, inject_audio_dropouts
from avcheck.inject.desync import inject_av_desync
from avcheck.inject.media_io import mux_video_audio, read_audio, read_video_frames, write_audio, write_video_frames
from avcheck.inject.video_defects import (
    inject_banding,
    inject_black_frames,
    inject_color_shift,
    inject_frame_drops,
    inject_frame_freezes,
)


def _write_json(data: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def inject_all_defects(video_path: str, audio_path: str, output_dir: str, seed: int = 0) -> dict:
    """Generate one ground-truth-labeled variant per defect class.

    Each variant is a standalone playable file (video defects: the reference
    audio remuxed back in unchanged; audio/desync defects: the reference video
    remuxed with the modified audio) plus a `<name>.ground_truth.json`. A
    manifest.json indexes every variant for `avcheck evaluate` to consume.
    """
    os.makedirs(output_dir, exist_ok=True)
    frames, fps = read_video_frames(video_path)
    samples, sr = read_audio(audio_path)

    manifest = {"reference_video": video_path, "reference_audio": audio_path, "variants": []}

    def add_video_variant(name: str, new_frames: list) -> str:
        video_only = os.path.join(output_dir, f"_{name}_video_only.mp4")
        final_path = os.path.join(output_dir, f"{name}.mp4")
        write_video_frames(new_frames, fps, video_only)
        mux_video_audio(video_only, audio_path, final_path)
        os.remove(video_only)
        return final_path

    def add_audio_variant(name: str, new_samples) -> str:
        audio_only = os.path.join(output_dir, f"{name}.wav")
        final_path = os.path.join(output_dir, f"{name}.mp4")
        write_audio(new_samples, sr, audio_only)
        mux_video_audio(video_path, audio_only, final_path)
        return final_path

    def register(name: str, media: str, file_path: str, gt: dict, audio_file: str = None) -> None:
        gt_path = os.path.join(output_dir, f"{name}.ground_truth.json")
        _write_json(gt, gt_path)
        entry = {"name": name, "media": media, "file": file_path, "ground_truth": gt_path}
        if audio_file:
            entry["audio_file"] = audio_file
        manifest["variants"].append(entry)

    fd_frames, fd_gt = inject_frame_drops(frames, fps, seed=seed)
    register("frame_drop", "video", add_video_variant("frame_drop", fd_frames), fd_gt)

    fz_frames, fz_gt = inject_frame_freezes(frames, fps, seed=seed)
    register("frame_freeze", "video", add_video_variant("frame_freeze", fz_frames), fz_gt)

    bf_frames, bf_gt = inject_black_frames(frames, fps, seed=seed)
    register("black_frames", "video", add_video_variant("black_frames", bf_frames), bf_gt)

    band_frames, band_gt = inject_banding(frames, fps)
    register("banding", "video", add_video_variant("banding", band_frames), band_gt)

    cs_frames, cs_gt = inject_color_shift(frames, fps)
    register("color_shift", "video", add_video_variant("color_shift", cs_frames), cs_gt)

    clip_samples, clip_gt = inject_audio_clipping(samples, sr, seed=seed)
    audio_only_path = os.path.join(output_dir, "audio_clipping.wav")
    register("audio_clipping", "audio", add_audio_variant("audio_clipping", clip_samples), clip_gt, audio_only_path)

    drop_samples, drop_gt = inject_audio_dropouts(samples, sr, seed=seed)
    audio_only_path = os.path.join(output_dir, "audio_dropout.wav")
    register("audio_dropout", "audio", add_audio_variant("audio_dropout", drop_samples), drop_gt, audio_only_path)

    desync_samples, desync_gt = inject_av_desync(samples, sr, offset_ms=200.0)
    audio_only_path = os.path.join(output_dir, "av_desync.wav")
    register("av_desync", "audio", add_audio_variant("av_desync", desync_samples), desync_gt, audio_only_path)

    manifest_path = os.path.join(output_dir, "manifest.json")
    _write_json(manifest, manifest_path)
    manifest["manifest_path"] = manifest_path
    return manifest
