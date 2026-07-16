import json
from types import SimpleNamespace

from avcheck.triage.report import generate_triage_report, write_triage_report

SAMPLE_EVALUATION = {
    "frame_drop": {"tp": 2, "fp": 3, "fn": 1, "precision": 0.4, "recall": 0.667},
    "black_frames": {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0},
    "audio_clipping": {"tp": 0, "fp": 0, "fn": 1, "precision": 0.0, "recall": 0.0},
}


def test_fallback_used_when_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    report = generate_triage_report(SAMPLE_EVALUATION)
    assert "Rule-based fallback" in report
    assert "frame_drop" in report
    assert "audio_clipping" in report


def test_fallback_ranks_worst_defect_first(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    report = generate_triage_report(SAMPLE_EVALUATION)
    # audio_clipping (precision=0, recall=0) is strictly worse than black_frames (perfect score)
    assert report.index("## audio_clipping") < report.index("## black_frames")


def test_write_triage_report_writes_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    eval_path = tmp_path / "evaluation.json"
    eval_path.write_text(json.dumps(SAMPLE_EVALUATION))
    out_path = tmp_path / "triage.md"

    report = write_triage_report(str(eval_path), str(out_path))
    assert out_path.exists()
    assert out_path.read_text() == report


def test_generate_triage_report_calls_anthropic_when_key_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="# LLM Triage Report\n\nfake content")])

    class FakeClient:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key
            self.messages = FakeMessages()

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)

    report = generate_triage_report(SAMPLE_EVALUATION)
    assert report == "# LLM Triage Report\n\nfake content"
    assert captured["api_key"] == "fake-key-for-test"
    assert captured["model"] == "claude-opus-4-8"
    assert json.loads(captured["messages"][0]["content"]) == SAMPLE_EVALUATION
