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
from avcheck.inject.audio_defects import inject_audio_clipping, inject_audio_dropouts
from avcheck.inject.desync import inject_av_desync
from avcheck.inject.media_io import read_video_frames, write_video_frames
from avcheck.inject.video_defects import (
    inject_banding,
    inject_black_frames,
    inject_color_shift,
    inject_frame_drops,
    inject_frame_freezes,
)
from tests.conftest import make_demo_audio, make_demo_frames, make_gradient_frames


def test_detect_frame_drops_finds_injected_drops(tmp_path):
    frames = make_demo_frames(60, fps=10)
    dropped, gt = inject_frame_drops(frames, fps=10, num_drops=3, seed=0)
    write_video_frames(dropped, 10, str(tmp_path / "test.mp4"))
    test_frames, _ = read_video_frames(str(tmp_path / "test.mp4"))

    result = detect_frame_drops(frames, test_frames, fps=10)
    detected_times = {round(e["timestamp_sec"], 1) for e in result["events"]}
    gt_times = {round(e["timestamp_sec"], 1) for e in gt["events"]}
    assert detected_times & gt_times  # at least one real drop recovered


def test_detect_frame_freezes_finds_injected_freeze(tmp_path):
    frames = make_demo_frames(60, fps=10)
    frozen, gt = inject_frame_freezes(frames, fps=10, num_freezes=1, freeze_len=5, seed=0)
    write_video_frames(frozen, 10, str(tmp_path / "test.mp4"))
    test_frames, _ = read_video_frames(str(tmp_path / "test.mp4"))

    result = detect_frame_freezes(test_frames, fps=10)
    assert len(result["events"]) >= 1
    event = result["events"][0]
    gt_event = gt["events"][0]
    assert abs(event["start_frame"] - gt_event["start_frame"]) <= 1


def test_detect_frame_freezes_no_false_positive_on_untouched_clip(tmp_path):
    frames = make_demo_frames(60, fps=10)
    write_video_frames(frames, 10, str(tmp_path / "ref.mp4"))
    test_frames, _ = read_video_frames(str(tmp_path / "ref.mp4"))
    result = detect_frame_freezes(test_frames, fps=10)
    assert result["events"] == []


def test_detect_black_frames_finds_injected_region(tmp_path):
    frames = make_demo_frames(60, fps=10)
    blacked, gt = inject_black_frames(frames, fps=10, num_events=1, duration_frames=4, seed=0)
    write_video_frames(blacked, 10, str(tmp_path / "test.mp4"))
    test_frames, _ = read_video_frames(str(tmp_path / "test.mp4"))

    result = detect_black_frames(test_frames, fps=10)
    assert len(result["events"]) == 1
    assert result["events"][0]["start_frame"] == gt["events"][0]["start_frame"]


def test_detect_banding_finds_defect_on_gradient_content(tmp_path):
    frames = make_gradient_frames(20)
    banded, gt = inject_banding(frames, fps=10, levels=8)
    write_video_frames(frames, 10, str(tmp_path / "ref.mp4"))
    write_video_frames(banded, 10, str(tmp_path / "test.mp4"))
    ref_frames, _ = read_video_frames(str(tmp_path / "ref.mp4"))
    test_frames, _ = read_video_frames(str(tmp_path / "test.mp4"))

    result = detect_banding(ref_frames, test_frames, fps=10)
    assert len(result["events"]) == 1


def test_detect_banding_no_false_positive_on_untouched_clip(tmp_path):
    frames = make_gradient_frames(20)
    write_video_frames(frames, 10, str(tmp_path / "ref.mp4"))
    ref_frames, _ = read_video_frames(str(tmp_path / "ref.mp4"))
    result = detect_banding(ref_frames, ref_frames, fps=10)
    assert result["events"] == []


def test_detect_color_shift_finds_defect(tmp_path):
    frames = make_demo_frames(30, fps=10)
    shifted, gt = inject_color_shift(frames, fps=10, channel_shift_bgr=(30, -10, -20))
    write_video_frames(frames, 10, str(tmp_path / "ref.mp4"))
    write_video_frames(shifted, 10, str(tmp_path / "test.mp4"))
    ref_frames, _ = read_video_frames(str(tmp_path / "ref.mp4"))
    test_frames, _ = read_video_frames(str(tmp_path / "test.mp4"))

    result = detect_color_shift(ref_frames, test_frames, fps=10)
    assert len(result["events"]) == 1


def test_detect_audio_clipping_finds_injected_region():
    audio = make_demo_audio(3.0)
    clipped, gt = inject_audio_clipping(audio, sr=22050, gain=5.0, num_events=1, duration_sec=0.2, seed=0)
    result = detect_audio_clipping(clipped, sr=22050)
    assert len(result["events"]) >= 1
    gt_event = gt["events"][0]
    detected = result["events"][0]
    assert abs(detected["start_sec"] - gt_event["start_sec"]) < 0.05


def test_detect_audio_dropouts_finds_injected_region():
    audio = make_demo_audio(3.0)
    dropped, gt = inject_audio_dropouts(audio, sr=22050, num_events=1, duration_sec=0.3, seed=0)
    result = detect_audio_dropouts(audio, dropped, sr=22050)
    assert len(result["events"]) == 1
    gt_event = gt["events"][0]
    detected = result["events"][0]
    assert abs(detected["start_sec"] - gt_event["start_sec"]) < 0.05


def test_detect_av_desync_recovers_offset_on_correlated_fixture(tmp_path):
    fps = 10
    frames = make_demo_frames(60, fps=fps, flash_interval_sec=1.0)
    write_video_frames(frames, fps, str(tmp_path / "ref.mp4"))
    ref_frames, _ = read_video_frames(str(tmp_path / "ref.mp4"))

    audio = make_demo_audio(6.0, flash_interval_sec=1.0)
    shifted, gt = inject_av_desync(audio, sr=22050, offset_ms=200.0)

    result = detect_av_desync(ref_frames, shifted, sr=22050, fps=fps)
    assert len(result["events"]) == 1
    assert result["events"][0]["direction"] == "audio_delayed"
    assert abs(result["estimated_offset_ms"] - 200.0) < 50.0
