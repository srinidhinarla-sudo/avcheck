import numpy as np

from avcheck.audio.integrity import (
    compare_chromagram,
    compute_dynamic_range_delta,
    compute_loudness_delta,
    detect_clipping,
    detect_silence_dropouts,
)
from tests.conftest import sine


def test_detect_clipping_finds_injected_clip(sr):
    x = np.zeros(sr)  # 1 second of silence
    x[100:110] = 1.0  # 10-sample clipped run
    result = detect_clipping(x, sr)
    assert result["num_clipped_regions"] == 1
    start, end = result["clipped_regions"][0]
    assert start == 100 / sr
    assert end == 110 / sr
    assert result["num_clipped_samples"] == 10


def test_detect_clipping_ignores_isolated_peak(sr):
    x = np.zeros(sr)
    x[500] = 1.0  # single sample, below default min_run=2
    result = detect_clipping(x, sr, min_run=2)
    assert result["num_clipped_regions"] == 0
    assert result["num_clipped_samples"] == 1  # still counted, just not a "region"


def test_detect_clipping_on_clean_signal_finds_nothing(sr, ref_tone):
    result = detect_clipping(ref_tone, sr)
    assert result["num_clipped_regions"] == 0
    assert result["num_clipped_samples"] == 0


def test_compute_loudness_delta_zero_for_identical_signal(sr, ref_tone):
    result = compute_loudness_delta(ref_tone, ref_tone, sr)
    assert abs(result["global_delta_db"]) < 1e-6


def test_compute_loudness_delta_detects_quieter_test(sr, ref_tone):
    quieter = ref_tone * 0.5  # half amplitude => ~ -6.02 dB
    result = compute_loudness_delta(ref_tone, quieter, sr)
    assert result["global_delta_db"] < 0
    assert abs(result["global_delta_db"] - (-6.02)) < 0.1


def test_detect_silence_dropouts_finds_zeroed_region(sr, ref_tone):
    test = ref_tone.copy()
    test[sr : sr + sr // 2] = 0.0  # zero out 0.5s starting at t=1s
    result = detect_silence_dropouts(ref_tone, test, sr)
    assert result["num_dropout_regions"] >= 1
    start, end = result["dropout_regions"][0]
    assert 0.9 < start < 1.1
    assert 1.4 < end < 1.6


def test_detect_silence_dropouts_no_false_positive_on_identical_signal(sr, ref_tone):
    result = detect_silence_dropouts(ref_tone, ref_tone, sr)
    assert result["num_dropout_regions"] == 0


def test_compute_dynamic_range_delta_detects_compression(sr):
    # Alternate loud/quiet half-second bursts -> real dynamic range in ref.
    loud = sine(440.0, 0.5, sr=sr, amplitude=0.9)
    quiet = sine(440.0, 0.5, sr=sr, amplitude=0.05)
    ref = np.concatenate([loud, quiet] * 4)

    # Compressed test: quiet parts boosted, loud parts left alone -> smaller spread.
    compressed_quiet = sine(440.0, 0.5, sr=sr, amplitude=0.5)
    test = np.concatenate([loud, compressed_quiet] * 4)

    result = compute_dynamic_range_delta(ref, test, sr)
    assert result["test_dynamic_range_db"] < result["ref_dynamic_range_db"]
    assert result["dynamic_range_delta_db"] < 0


def test_compute_dynamic_range_delta_zero_for_identical_signal(sr, ref_tone):
    result = compute_dynamic_range_delta(ref_tone, ref_tone, sr)
    assert abs(result["dynamic_range_delta_db"]) < 1e-6


def test_compare_chromagram_high_similarity_for_identical_signal(sr, ref_tone):
    result = compare_chromagram(ref_tone, ref_tone, sr)
    assert result["mean_chroma_similarity"] > 0.99


def test_compare_chromagram_low_similarity_for_different_pitch(sr):
    # 440Hz (A) vs 466.16Hz (A#) land in different chroma bins.
    a = sine(440.0, 3.0, sr=sr)
    a_sharp = sine(466.16, 3.0, sr=sr)
    result = compare_chromagram(a, a_sharp, sr)
    assert result["mean_chroma_similarity"] < 0.9
