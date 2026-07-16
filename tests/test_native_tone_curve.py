import numpy as np

from avcheck.native_filter import ToneCurveFilter, process_video
from avcheck.video.quality import score_video
from tests.conftest import make_demo_frames, write_video


def test_identity_filter_leaves_pixels_unchanged():
    filt = ToneCurveFilter(brightness=0.0, gamma=1.0)
    frame = np.array([0, 50, 128, 200, 255], dtype=np.uint8)
    assert np.array_equal(filt.apply(frame), frame)


def test_brightness_increases_pixel_values():
    filt = ToneCurveFilter(brightness=0.1, gamma=1.0)
    frame = np.array([0, 100, 200], dtype=np.uint8)
    result = filt.apply(frame)
    assert np.all(result >= frame)
    assert np.any(result > frame)


def test_gamma_correction_changes_midtones():
    filt = ToneCurveFilter(brightness=0.0, gamma=2.0)
    frame = np.array([0, 128, 255], dtype=np.uint8)
    result = filt.apply(frame)
    assert result[0] == 0
    assert result[2] == 255
    assert result[1] != 128  # midtone shifts under gamma != 1.0


def test_clamps_at_ceiling_and_floor_by_default():
    filt = ToneCurveFilter(brightness=0.5, gamma=1.0, skip_clamp=False)
    frame = np.array([250, 255], dtype=np.uint8)
    result = filt.apply(frame)
    assert np.all(result == 255)


def test_skip_clamp_wraps_around_instead_of_saturating():
    clamped = ToneCurveFilter(brightness=0.5, gamma=1.0, skip_clamp=False)
    unclamped = ToneCurveFilter(brightness=0.5, gamma=1.0, skip_clamp=True)
    frame = np.array([250, 255], dtype=np.uint8)

    clamped_result = clamped.apply(frame)
    unclamped_result = unclamped.apply(frame)

    assert np.all(clamped_result == 255)
    assert np.all(unclamped_result < 255)  # wrapped around mod 256 instead of saturating


def test_filter_properties_reflect_constructor_args():
    filt = ToneCurveFilter(brightness=0.2, gamma=1.5, skip_clamp=True)
    assert filt.brightness == 0.2
    assert filt.gamma == 1.5
    assert filt.skip_clamp is True


def test_many_filters_can_be_created_and_destroyed():
    # Smoke test for the RAII lookup-table lifecycle: no leak or double-free
    # across repeated construction/destruction.
    for i in range(200):
        ToneCurveFilter(brightness=i / 200, gamma=1.0 + i / 200)


def test_process_video_preserves_frame_count(tmp_path):
    frames = make_demo_frames(20, fps=10)
    ref_path = tmp_path / "ref.mp4"
    out_path = tmp_path / "out.mp4"
    write_video(str(ref_path), frames, fps=10)

    process_video(str(ref_path), str(out_path), brightness=0.1, gamma=1.1)
    result = score_video(str(ref_path), str(out_path))
    assert result["summary"]["num_frames"] == 20


def test_end_to_end_correct_filter_is_a_bounded_quality_change(tmp_path):
    """Validating the C++ module with the toolkit itself: a legitimate tone-curve
    adjustment should show up as a real but bounded PSNR/SSIM delta, not corruption."""
    frames = make_demo_frames(30, fps=10)
    ref_path = tmp_path / "ref.mp4"
    corrected_path = tmp_path / "corrected.mp4"
    write_video(str(ref_path), frames, fps=10)

    process_video(str(ref_path), str(corrected_path), brightness=0.1, gamma=1.2, skip_clamp=False)
    result = score_video(str(ref_path), str(corrected_path))

    assert result["summary"]["mean_ssim"] > 0.7


def test_end_to_end_buggy_filter_is_caught_by_quality_scorer(tmp_path):
    """Plant a deliberate bug (skip_clamp=True) and confirm avcheck's own
    PSNR/SSIM scorer flags it as a severe regression relative to the correctly
    clamped version — the same validation loop an SDK team would run."""
    frames = make_demo_frames(30, fps=10)
    ref_path = tmp_path / "ref.mp4"
    correct_path = tmp_path / "correct.mp4"
    buggy_path = tmp_path / "buggy.mp4"
    write_video(str(ref_path), frames, fps=10)

    process_video(str(ref_path), str(correct_path), brightness=0.2, gamma=1.2, skip_clamp=False)
    process_video(str(ref_path), str(buggy_path), brightness=0.2, gamma=1.2, skip_clamp=True)

    correct_result = score_video(str(ref_path), str(correct_path))
    buggy_result = score_video(str(ref_path), str(buggy_path))

    assert buggy_result["summary"]["mean_ssim"] < correct_result["summary"]["mean_ssim"] - 0.15
    assert buggy_result["summary"]["mean_psnr"] < correct_result["summary"]["mean_psnr"] - 4.0
