"""Time-tolerant matching between detector output and ground truth, for precision/recall."""


def _as_interval(event: dict, clip_duration: float) -> tuple:
    """Canonicalize an event dict into a (start_sec, end_sec) interval.

    Point events (e.g. a single dropped frame's timestamp) become zero-width
    intervals. Clip-global events (banding, color shift, desync — defects that
    describe the whole clip rather than a specific span) become the full
    [0, clip_duration] interval, since "where" isn't meaningful for them and
    matching degrades to "did the detector fire at all."
    """
    if "start_sec" in event and "end_sec" in event:
        return event["start_sec"], event["end_sec"]
    if "start_timestamp_sec" in event and "end_timestamp_sec" in event:
        return event["start_timestamp_sec"], event["end_timestamp_sec"]
    if "timestamp_sec" in event:
        return event["timestamp_sec"], event["timestamp_sec"]
    return 0.0, clip_duration


def _overlaps(a: tuple, b: tuple, tolerance_sec: float) -> bool:
    a_start, a_end = a
    b_start, b_end = b
    return (a_start - tolerance_sec) <= b_end and (b_start - tolerance_sec) <= a_end


def match_events(predicted: list, ground_truth: list, clip_duration: float, tolerance_sec: float = 0.15) -> dict:
    """Greedily match predicted events to ground-truth events within tolerance_sec.

    Returns counts: true_positives (gt events matched), false_negatives (gt events
    unmatched), false_positives (predicted events that matched no gt event).
    """
    gt_intervals = [_as_interval(e, clip_duration) for e in ground_truth]
    pred_intervals = [_as_interval(e, clip_duration) for e in predicted]

    matched_pred = set()
    true_positives = 0
    for gt in gt_intervals:
        match_idx = next(
            (i for i, p in enumerate(pred_intervals) if i not in matched_pred and _overlaps(gt, p, tolerance_sec)),
            None,
        )
        if match_idx is not None:
            matched_pred.add(match_idx)
            true_positives += 1

    false_negatives = len(gt_intervals) - true_positives
    false_positives = len(pred_intervals) - len(matched_pred)

    return {"tp": true_positives, "fp": false_positives, "fn": false_negatives}
