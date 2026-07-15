import soundfile as sf

from avcheck.cli import main
from tests.conftest import make_checkerboard_frames, sine, write_video


def test_avcheck_inject_writes_variants_and_manifest(tmp_path, capsys):
    frames = make_checkerboard_frames(30)
    video_path = tmp_path / "ref.mp4"
    write_video(str(video_path), frames, fps=10)

    sr = 22050
    audio = sine(440.0, 3.0, sr=sr)
    audio_path = tmp_path / "ref.wav"
    sf.write(str(audio_path), audio, sr)

    output_dir = tmp_path / "out"
    exit_code = main(["inject", str(video_path), str(audio_path), "-o", str(output_dir), "--seed", "1"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "AVCheck Injection Report" in captured.out
    assert "Generated 8 labeled variants" in captured.out

    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "frame_drop.mp4").exists()
    assert (output_dir / "av_desync.ground_truth.json").exists()
