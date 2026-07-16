import soundfile as sf

from avcheck.cli import main
from tests.conftest import make_demo_audio, make_demo_frames, write_video


def test_avcheck_evaluate_writes_csv_and_summary(tmp_path, capsys):
    frames = make_demo_frames(60, fps=10)
    video_path = tmp_path / "ref.mp4"
    write_video(str(video_path), frames, fps=10)

    audio = make_demo_audio(6.0)
    audio_path = tmp_path / "ref.wav"
    sf.write(str(audio_path), audio, 22050)

    variants_dir = tmp_path / "variants"
    exit_code = main(["inject", str(video_path), str(audio_path), "-o", str(variants_dir)])
    assert exit_code == 0

    out_csv = tmp_path / "eval.csv"
    manifest_path = variants_dir / "manifest.json"
    exit_code = main(["evaluate", str(manifest_path), str(video_path), str(audio_path), "-o", str(out_csv)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "AVCheck Evaluation Report" in captured.out
    assert "frame_drop" in captured.out
    assert out_csv.exists()
