"""LLM-powered triage: turns an `avcheck evaluate` report into a plain-English
markdown summary. A log-analysis assistant, not an agent — one API call, no
tool use, no loops. Falls back to a rule-based summary when no Anthropic API
key is configured.
"""

import json
import os

SYSTEM_PROMPT = (
    "You are a media QC triage assistant. You are given precision/recall results "
    "from an audio/video defect-detection pipeline (JSON: per defect class, "
    "true positives, false positives, false negatives, precision, recall). "
    "Produce a concise markdown triage report. For each defect class with any "
    "detections or misses (tp, fp, or fn > 0), give: a plain-English description "
    "of the failure, a probable root-cause category (encoding, sync, or source "
    "corruption), and one concrete next debugging step. Rank defects worst-first "
    "by how unreliable detection was (low recall or low precision). Do not invent "
    "defects not present in the data. Output markdown only, no preamble."
)


def _severity_score(metrics: dict) -> float:
    """Lower precision/recall make a defect class more severe."""
    return (1 - metrics["recall"]) + (1 - metrics["precision"])


def _fallback_report(evaluation: dict) -> str:
    """Rule-based triage used when no ANTHROPIC_API_KEY is configured."""
    ranked = sorted(evaluation.items(), key=lambda kv: _severity_score(kv[1]), reverse=True)

    lines = ["# AVCheck Triage Report", "", "_Rule-based fallback — set ANTHROPIC_API_KEY for LLM-generated root-cause analysis._", ""]
    for name, m in ranked:
        lines.append(f"## {name}")
        lines.append(f"- Precision: {m['precision']:.2f}, Recall: {m['recall']:.2f} (tp={m['tp']}, fp={m['fp']}, fn={m['fn']})")
        if m["fn"] > 0:
            lines.append(f"- Missed {m['fn']} true instance(s) — detector under-triggers on this defect class.")
        if m["fp"] > 0:
            lines.append(f"- Raised {m['fp']} false alarm(s) — detector over-triggers or is confused by another defect class.")
        lines.append("- Root cause: unavailable without LLM triage.")
        lines.append("")
    return "\n".join(lines)


def generate_triage_report(evaluation: dict) -> str:
    """Generate a markdown triage report from an `avcheck evaluate` result dict."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _fallback_report(evaluation)

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(evaluation, indent=2)}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return text or _fallback_report(evaluation)


def write_triage_report(evaluation_path: str, output_path: str) -> str:
    with open(evaluation_path) as f:
        evaluation = json.load(f)
    report = generate_triage_report(evaluation)
    with open(output_path, "w") as f:
        f.write(report)
    return report
