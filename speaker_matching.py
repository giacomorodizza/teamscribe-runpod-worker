def speaker_for_segment(
    start: float,
    end: float,
    turns: list[tuple[float, float, str]],
    max_gap: float,
) -> str:
    overlap, label = 0.0, "UNK"
    before = after = None
    for turn_start, turn_end, speaker in turns:
        value = max(0.0, min(end, turn_end) - max(start, turn_start))
        if value > overlap:
            overlap, label = value, speaker
        if turn_end <= start and (before is None or turn_end > before[1]):
            before = (turn_start, turn_end, speaker)
        if turn_start >= end and (after is None or turn_start < after[0]):
            after = (turn_start, turn_end, speaker)
    if overlap or before is None or after is None:
        return label
    if (
        before[2] == after[2]
        and start - before[1] <= max_gap
        and after[0] - end <= max_gap
    ):
        return before[2]
    return label
