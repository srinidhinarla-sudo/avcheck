import numpy as np

from avcheck.inject.video_defects import (
    inject_banding,
    inject_black_frames,
    inject_color_shift,
    inject_frame_drops,
    inject_frame_freezes,
)
from tests.conftest import make_checkerboard_frames


def test_inject_frame_drops_removes_correct_count():
    frames = make_checkerboard_frames(20)
    new_frames, gt = inject_frame_drops(frames, fps=10, num_drops=3, seed=1)
    assert len(new_frames) == 17
    assert gt["defect_type"] == "frame_drop"
    assert len(gt["events"]) == 3
    for event in gt["events"]:
        assert event["timestamp_sec"] == event["frame_index"] / 10


def test_inject_frame_freezes_repeats_frame():
    frames = make_checkerboard_frames(20)
    new_frames, gt = inject_frame_freezes(frames, fps=10, num_freezes=1, freeze_len=5, seed=2)
    start = gt["events"][0]["start_frame"]
    end = gt["events"][0]["end_frame"]
    for i in range(start, end + 1):
        assert np.array_equal(new_frames[i], new_frames[start])
    assert end - start == 4


def test_inject_black_frames_zeros_region():
    frames = make_checkerboard_frames(20)
    new_frames, gt = inject_black_frames(frames, fps=10, num_events=1, duration_frames=4, seed=3)
    start = gt["events"][0]["start_frame"]
    end = gt["events"][0]["end_frame"]
    for i in range(start, end + 1):
        assert np.all(new_frames[i] == 0)
    # untouched frame elsewhere should not be all-black
    assert not np.all(new_frames[0] == 0)


def test_inject_banding_reduces_unique_pixel_values():
    frames = make_checkerboard_frames(5)
    new_frames, gt = inject_banding(frames, fps=10, levels=4)
    assert gt["params"]["levels"] == 4
    for orig, banded in zip(frames, new_frames):
        assert len(np.unique(banded)) <= len(np.unique(orig))
        assert banded.shape == orig.shape


def test_inject_color_shift_changes_pixels():
    frames = make_checkerboard_frames(5)
    new_frames, gt = inject_color_shift(frames, fps=10, channel_shift_bgr=(30, -10, -20))
    assert gt["params"]["channel_shift_bgr"] == [30, -10, -20]
    assert not np.array_equal(frames[0], new_frames[0])
