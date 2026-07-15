import numpy as np

from avcheck.video.plotting import plot_quality_curve
from avcheck.video.quality import compute_psnr, compute_ssim, score_video
from tests.conftest import make_checkerboard_frames, write_video


def test_compute_psnr_identical_frames_is_inf():
    frame = np.full((32, 32), 128, dtype=np.uint8)
    assert compute_psnr(frame, frame) == float("inf")


def test_compute_psnr_decreases_with_noise():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(32, 32), dtype=np.uint8)
    light_noise = np.clip(frame.astype(int) + rng.integers(-5, 5, frame.shape), 0, 255).astype(np.uint8)
    heavy_noise = np.clip(frame.astype(int) + rng.integers(-80, 80, frame.shape), 0, 255).astype(np.uint8)

    psnr_light = compute_psnr(frame, light_noise)
    psnr_heavy = compute_psnr(frame, heavy_noise)
    assert psnr_light > psnr_heavy


def test_compute_ssim_identical_frames_is_one():
    frame = np.full((32, 32), 128, dtype=np.uint8)
    assert compute_ssim(frame, frame) == 1.0


def test_compute_ssim_decreases_with_noise():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(32, 32), dtype=np.uint8).astype(np.uint8)
    heavy_noise = np.clip(frame.astype(int) + rng.integers(-100, 100, frame.shape), 0, 255).astype(np.uint8)
    assert compute_ssim(frame, heavy_noise) < compute_ssim(frame, frame)


def test_score_video_identical_videos_have_high_quality(tmp_path):
    frames = make_checkerboard_frames(10)
    ref_path = tmp_path / "ref.mp4"
    test_path = tmp_path / "test.mp4"
    write_video(str(ref_path), frames)
    write_video(str(test_path), frames)

    result = score_video(str(ref_path), str(test_path))
    assert result["summary"]["num_frames"] == 10
    assert result["summary"]["mean_ssim"] > 0.99


def test_score_video_degraded_video_scores_lower(tmp_path):
    frames = make_checkerboard_frames(10)
    rng = np.random.default_rng(1)
    degraded = [np.clip(f.astype(int) + rng.integers(-60, 60, f.shape), 0, 255).astype(np.uint8) for f in frames]

    ref_path = tmp_path / "ref.mp4"
    test_path = tmp_path / "test.mp4"
    write_video(str(ref_path), frames)
    write_video(str(test_path), degraded)

    clean_result = score_video(str(ref_path), str(ref_path))
    degraded_result = score_video(str(ref_path), str(test_path))

    assert degraded_result["summary"]["mean_ssim"] < clean_result["summary"]["mean_ssim"]
    assert degraded_result["summary"]["mean_psnr"] < clean_result["summary"]["mean_psnr"]


def test_score_video_stops_at_shorter_video(tmp_path):
    long_frames = make_checkerboard_frames(10)
    short_frames = make_checkerboard_frames(4)
    ref_path = tmp_path / "ref.mp4"
    test_path = tmp_path / "test.mp4"
    write_video(str(ref_path), long_frames)
    write_video(str(test_path), short_frames)

    result = score_video(str(ref_path), str(test_path))
    assert result["summary"]["num_frames"] == 4


def test_plot_quality_curve_creates_file(tmp_path):
    per_frame = [{"timestamp": i * 0.1, "psnr": 30 + i, "ssim": 0.9} for i in range(5)]
    out_path = tmp_path / "curve.png"
    plot_quality_curve(per_frame, str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0
