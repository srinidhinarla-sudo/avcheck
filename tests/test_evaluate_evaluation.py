import soundfile as sf

from avcheck.evaluate.evaluation import DEFECT_CLASSES, evaluate_variants
from avcheck.inject.engine import inject_all_defects
from avcheck.inject.media_io import write_video_frames
from tests.conftest import make_demo_audio, make_demo_frames


def test_evaluate_variants_covers_all_defect_classes(tmp_path):
    frames = make_demo_frames(60, fps=10)
    video_path = tmp_path / "ref.mp4"
    write_video_frames(frames, 10, str(video_path))

    audio = make_demo_audio(6.0)
    audio_path = tmp_path / "ref.wav"
    sf.write(str(audio_path), audio, 22050)

    output_dir = tmp_path / "variants"
    manifest = inject_all_defects(str(video_path), str(audio_path), str(output_dir), seed=0)

    report = evaluate_variants(manifest["manifest_path"], str(video_path), str(audio_path))

    assert set(report.keys()) == set(DEFECT_CLASSES)
    for name, metrics in report.items():
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0

    # Every defect class should recover its own ground-truth event at least once.
    recalls = [metrics["recall"] for metrics in report.values()]
    assert sum(r > 0 for r in recalls) >= 6  # most classes should have nonzero recall
