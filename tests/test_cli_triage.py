import json

from avcheck.cli import main

SAMPLE_EVALUATION = {
    "frame_drop": {"tp": 2, "fp": 3, "fn": 1, "precision": 0.4, "recall": 0.667},
    "black_frames": {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0},
}


def test_avcheck_triage_writes_markdown(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    eval_path = tmp_path / "evaluation.json"
    eval_path.write_text(json.dumps(SAMPLE_EVALUATION))
    out_path = tmp_path / "triage.md"

    exit_code = main(["triage", str(eval_path), "-o", str(out_path)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Triage report written to" in captured.out
    assert out_path.exists()
    assert "frame_drop" in out_path.read_text()
