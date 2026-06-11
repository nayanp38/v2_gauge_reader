from __future__ import annotations

from math import isfinite

from .geometry import normalize_direction, sweep_delta
from .models import InterpolationResult, ValueTick


def interpolate_reading(
    needle_angle: float,
    value_ticks: list[ValueTick],
    direction: str | None = None,
) -> InterpolationResult | None:
    ticks = [tick for tick in value_ticks if isfinite(tick.angle) and isfinite(tick.value)]
    if len(ticks) < 2:
        return None

    normalized_direction = normalize_direction(direction)
    directions = ["clockwise", "counterclockwise"] if normalized_direction == "auto" else [normalized_direction]
    candidates: list[tuple[float, float, InterpolationResult]] = []

    for start in ticks:
        for end in ticks:
            if start is end:
                continue
            if abs(start.value - end.value) < 1e-9:
                continue
            for candidate_direction in directions:
                span = sweep_delta(start.angle, end.angle, candidate_direction)
                if span <= 1e-6 or span >= 359.999:
                    continue
                offset = sweep_delta(start.angle, needle_angle, candidate_direction)
                if offset > span + 1e-6:
                    continue
                fraction = max(0.0, min(1.0, offset / span))
                reading = start.value + (end.value - start.value) * fraction
                lower_tick, upper_tick = (
                    (start, end) if start.value <= end.value else (end, start)
                )
                confidence = _span_confidence(span) * _source_confidence(start, end)
                result = InterpolationResult(
                    reading=reading,
                    confidence=confidence,
                    lower_tick=lower_tick,
                    upper_tick=upper_tick,
                    span_degrees=span,
                    direction=candidate_direction,
                    angular_fraction=fraction,
                )
                value_span = abs(start.value - end.value)
                candidates.append((span, value_span, result))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _span_confidence(span: float) -> float:
    if span <= 45.0:
        return 0.95
    if span <= 90.0:
        return 0.85
    if span <= 150.0:
        return 0.70
    if span <= 240.0:
        return 0.48
    return 0.34


def _source_confidence(a: ValueTick, b: ValueTick) -> float:
    return max(0.05, min(1.0, (a.confidence + b.confidence) / 2.0))
