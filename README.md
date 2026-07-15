# AVCheck

An automated audio/video validation toolkit that detects quality defects in media
**and explains them** — built to demonstrate audio/video validation engineering.

## Status

🚧 Phase 1 in progress: audio integrity core (clipping, loudness delta, silence/dropout
detection, dynamic range compression delta, chromagram/pitch-content comparison).

## Architecture

| Phase | What | Status |
|---|---|---|
| 1 | Audio validation core (`avcheck audio`) | 🚧 in progress |
| 2 | Video quality scoring (`avcheck video`) | ⬜ |
| 3 | Defect injection engine (`avcheck inject`) | ⬜ |
| 4 | Detectors + evaluation (`avcheck evaluate`) | ⬜ |
| 5 | C++ SDK-style frame filter module (pybind11) | ⬜ |
| 6 | LLM-powered triage (`avcheck triage`) | ⬜ |
| 7 | CI + polish | ⬜ |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage (Phase 1)

```bash
avcheck audio ref.wav test.wav
```

## Why I built this

_(written up in Phase 7)_

## How I used AI to build this

_(written up in Phase 7 — including at least one case where I caught an incorrect AI suggestion)_
