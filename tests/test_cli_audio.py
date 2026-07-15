import csv

import soundfile as sf

from avcheck.cli import main
from tests.conftest import sine


def test_avcheck_audio_writes_csv_and_summary(tmp_path, capsys):
    sr = 22050
    ref = sine(440.0, 2.0, sr=sr)
    test = ref * 0.5  # quieter copy

    ref_path = tmp_path / "ref.wav"
    test_path = tmp_path / "test.wav"
    out_path = tmp_path / "report.csv"
    sf.write(ref_path, ref, sr)
    sf.write(test_path, test, sr)

    exit_code = main(["audio", str(ref_path), str(test_path), "-o", str(out_path)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "AVCheck Audio Report" in captured.out
    assert "Loudness delta" in captured.out

    assert out_path.exists()
    with open(out_path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["check", "metric", "value"]
    metrics = {(r[0], r[1]) for r in rows[1:]}
    assert ("loudness", "global_delta_db") in metrics
    assert ("chromagram", "mean_chroma_similarity") in metrics
