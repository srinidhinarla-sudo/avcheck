import json

import cv2
import soundfile as sf

from avcheck.inject.engine import inject_all_defects
from tests.conftest import make_checkerboard_frames, sine, write_video


def _make_reference_media(tmp_path):
    frames = make_checkerboard_frames(30, size=(64, 64))
    video_path = tmp_path / "ref.mp4"
    write_video(str(video_path), frames, fps=10)

    sr = 22050
    audio = sine(440.0, 3.0, sr=sr, amplitude=0.5)
    audio_path = tmp_path / "ref.wav"
    sf.write(str(audio_path), audio, sr)

    return str(video_path), str(audio_path)


def test_inject_all_defects_creates_all_variants(tmp_path):
    video_path, audio_path = _make_reference_media(tmp_path)
    output_dir = tmp_path / "variants"

    manifest = inject_all_defects(video_path, audio_path, str(output_dir), seed=0)

    expected_names = {
        "frame_drop",
        "frame_freeze",
        "black_frames",
        "banding",
        "color_shift",
        "audio_clipping",
        "audio_dropout",
        "av_desync",
    }
    variant_names = {v["name"] for v in manifest["variants"]}
    assert variant_names == expected_names
    assert len(manifest["variants"]) == 8

    for variant in manifest["variants"]:
        assert (output_dir / f"{variant['name']}.mp4").exists()
        gt_path = output_dir / f"{variant['name']}.ground_truth.json"
        assert gt_path.exists()
        with open(gt_path) as f:
            gt = json.load(f)
        assert gt["defect_type"] == variant["name"] or gt["defect_type"] in variant["name"]
        assert "events" in gt

        cap = cv2.VideoCapture(str(output_dir / f"{variant['name']}.mp4"))
        assert cap.isOpened()
        ret, _ = cap.read()
        assert ret
        cap.release()

    assert (output_dir / "manifest.json").exists()


def test_inject_all_defects_frame_drop_variant_has_fewer_frames(tmp_path):
    video_path, audio_path = _make_reference_media(tmp_path)
    output_dir = tmp_path / "variants"
    manifest = inject_all_defects(video_path, audio_path, str(output_dir), seed=0)

    ref_cap = cv2.VideoCapture(video_path)
    ref_count = 0
    while True:
        ret, _ = ref_cap.read()
        if not ret:
            break
        ref_count += 1
    ref_cap.release()

    drop_variant = next(v for v in manifest["variants"] if v["name"] == "frame_drop")
    drop_cap = cv2.VideoCapture(drop_variant["file"])
    drop_count = 0
    while True:
        ret, _ = drop_cap.read()
        if not ret:
            break
        drop_count += 1
    drop_cap.release()

    assert drop_count < ref_count
