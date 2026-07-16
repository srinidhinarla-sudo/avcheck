#!/usr/bin/env python3
"""Mini end-to-end validation: inject -> evaluate -> triage, driven through the
real `avcheck` CLI exactly as a user would run it.

Generates its own tiny (<1MB) synthetic reference clip at run time -- no
binary test assets are checked into the repo. Used both locally
(`python scripts/e2e_demo.py`) and in CI (see .github/workflows/ci.yml).
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from avcheck.inject.media_io import write_video_frames


def make_demo_frames(num_frames, fps, size=(64, 64), flash_interval_sec=1.0):
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


def make_demo_audio(duration_sec, sr=22050, flash_interval_sec=1.0, tone_freq=440.0):
    t = np.arange(int(duration_sec * sr)) / sr
    audio = (0.2 * np.sin(2 * np.pi * tone_freq * t)).astype(np.float64)
    rng = np.random.default_rng(0)
    click_dur = 0.05
    for click_time in np.arange(0, duration_sec, flash_interval_sec):
        start = int(click_time * sr)
        end = min(int((click_time + click_dur) * sr), len(audio))
        audio[start:end] += np.clip(0.3 * rng.standard_normal(end - start), -0.5, 0.5)
    return audio.astype(np.float32)


def run(cmd: list) -> None:
    print(f"      $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="avcheck_e2e_"))
    try:
        fps, sr = 10, 22050
        ref_video = workdir / "ref.mp4"
        ref_audio = workdir / "ref.wav"
        write_video_frames(make_demo_frames(40, fps=fps), fps, str(ref_video))
        sf.write(str(ref_audio), make_demo_audio(4.0, sr=sr), sr)
        clip_size_kb = (ref_video.stat().st_size + ref_audio.stat().st_size) / 1024
        print(f"[0/3] Generated {clip_size_kb:.1f} KB synthetic reference clip.")

        variants_dir = workdir / "variants"
        print("[1/3] avcheck inject")
        run(["avcheck", "inject", str(ref_video), str(ref_audio), "-o", str(variants_dir)])

        eval_json = workdir / "evaluation.json"
        eval_csv = workdir / "evaluation.csv"
        print("[2/3] avcheck evaluate")
        run(
            [
                "avcheck",
                "evaluate",
                str(variants_dir / "manifest.json"),
                str(ref_video),
                str(ref_audio),
                "-o",
                str(eval_csv),
                "--json-output",
                str(eval_json),
            ]
        )

        triage_md = workdir / "triage.md"
        print("[3/3] avcheck triage")
        run(["avcheck", "triage", str(eval_json), "-o", str(triage_md)])

        assert eval_json.stat().st_size > 0
        assert triage_md.stat().st_size > 0
        print("\nMini end-to-end validation passed: inject -> detect -> triage.")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
