from avcheck.evaluate.matching import match_events


def test_match_events_perfect_match():
    predicted = [{"start_sec": 1.0, "end_sec": 1.3}]
    ground_truth = [{"start_sec": 1.0, "end_sec": 1.3}]
    result = match_events(predicted, ground_truth, clip_duration=10.0)
    assert result == {"tp": 1, "fp": 0, "fn": 0}


def test_match_events_false_positive_and_negative():
    predicted = [{"start_sec": 5.0, "end_sec": 5.1}]
    ground_truth = [{"start_sec": 1.0, "end_sec": 1.1}]
    result = match_events(predicted, ground_truth, clip_duration=10.0)
    assert result == {"tp": 0, "fp": 1, "fn": 1}


def test_match_events_point_event_matches_within_tolerance():
    predicted = [{"frame_index": 5, "timestamp_sec": 0.52}]
    ground_truth = [{"frame_index": 5, "timestamp_sec": 0.50}]
    result = match_events(predicted, ground_truth, clip_duration=10.0, tolerance_sec=0.05)
    assert result["tp"] == 1


def test_match_events_point_event_outside_tolerance_is_miss():
    predicted = [{"frame_index": 5, "timestamp_sec": 0.9}]
    ground_truth = [{"frame_index": 5, "timestamp_sec": 0.50}]
    result = match_events(predicted, ground_truth, clip_duration=10.0, tolerance_sec=0.05)
    assert result == {"tp": 0, "fp": 1, "fn": 1}


def test_match_events_global_event_matches_regardless_of_location():
    predicted = [{"start_frame": 0, "end_frame": 10, "start_timestamp_sec": 0.0, "end_timestamp_sec": 1.0}]
    ground_truth = [{"start_frame": 0, "end_frame": 59, "start_timestamp_sec": 0.0, "end_timestamp_sec": 6.0}]
    result = match_events(predicted, ground_truth, clip_duration=6.0)
    assert result["tp"] == 1


def test_match_events_no_predictions_all_false_negative():
    ground_truth = [{"start_sec": 1.0, "end_sec": 1.1}, {"start_sec": 2.0, "end_sec": 2.1}]
    result = match_events([], ground_truth, clip_duration=10.0)
    assert result == {"tp": 0, "fp": 0, "fn": 2}


def test_match_events_no_ground_truth_all_false_positive():
    predicted = [{"start_sec": 1.0, "end_sec": 1.1}]
    result = match_events(predicted, [], clip_duration=10.0)
    assert result == {"tp": 0, "fp": 1, "fn": 0}
