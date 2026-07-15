import csv

import numpy as np

from avcheck.cli import main
from tests.conftest import make_checkerboard_frames, write_video


def test_avcheck_video_writes_csv_summary_and_plot(tmp_path, capsys):
    frames = make_checkerboard_frames(8)
    rng = np.random.default_rng(2)
    degraded = [np.clip(f.astype(int) + rng.integers(-40, 40, f.shape), 0, 255).astype(np.uint8) for f in frames]

    ref_path = tmp_path / "ref.mp4"
    test_path = tmp_path / "test.mp4"
    write_video(str(ref_path), frames)
    write_video(str(test_path), degraded)

    out_csv = tmp_path / "report.csv"
    out_plot = tmp_path / "curve.png"

    exit_code = main(["video", str(ref_path), str(test_path), "-o", str(out_csv), "--plot", str(out_plot)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "AVCheck Video Report" in captured.out
    assert "Mean SSIM" in captured.out

    assert out_csv.exists()
    assert out_plot.exists()
    with open(out_csv) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["frame", "timestamp_sec", "psnr_db", "ssim"]
    assert len(rows) - 1 == 8
