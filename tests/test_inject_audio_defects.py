import numpy as np

from avcheck.inject.audio_defects import inject_audio_clipping, inject_audio_dropouts
from avcheck.inject.desync import inject_av_desync
from tests.conftest import sine


def test_inject_audio_clipping_hits_ceiling_in_region():
    sr = 22050
    samples = sine(440.0, 2.0, sr=sr, amplitude=0.5)
    new_samples, gt = inject_audio_clipping(samples, sr, gain=5.0, num_events=1, duration_sec=0.2, seed=0)
    start_sample = int(gt["events"][0]["start_sec"] * sr)
    end_sample = int(gt["events"][0]["end_sec"] * sr)
    assert np.max(np.abs(new_samples[start_sample:end_sample])) >= 0.99
    # untouched region should be unchanged
    assert np.array_equal(new_samples[:start_sample], samples[:start_sample])


def test_inject_audio_dropouts_zeros_region():
    sr = 22050
    samples = sine(440.0, 2.0, sr=sr, amplitude=0.5)
    new_samples, gt = inject_audio_dropouts(samples, sr, num_events=1, duration_sec=0.3, seed=0)
    start_sample = int(gt["events"][0]["start_sec"] * sr)
    end_sample = int(gt["events"][0]["end_sec"] * sr)
    assert np.all(new_samples[start_sample:end_sample] == 0.0)


def test_inject_av_desync_positive_delays_audio():
    sr = 1000
    samples = np.arange(1, 2001, dtype=np.float32)  # distinctive ramp, no zeros
    shifted, gt = inject_av_desync(samples, sr, offset_ms=200.0)
    assert gt["params"]["direction"] == "audio_delayed"
    assert len(shifted) == len(samples)
    assert np.all(shifted[:200] == 0.0)
    assert np.array_equal(shifted[200:], samples[:-200])


def test_inject_av_desync_negative_advances_audio():
    sr = 1000
    samples = np.arange(1, 2001, dtype=np.float32)
    shifted, gt = inject_av_desync(samples, sr, offset_ms=-200.0)
    assert gt["params"]["direction"] == "audio_advanced"
    assert len(shifted) == len(samples)
    assert np.all(shifted[-200:] == 0.0)
    assert np.array_equal(shifted[:-200], samples[200:])
