from __future__ import annotations

import math
from collections.abc import Iterable

from .models import DialGeometry, Point, TickDetection


def normalize_angle(angle: float) -> float:
    value = float(angle) % 360.0
    if value < 0.0:
        value += 360.0
    return value


def angle_from_point(center: Point, point: Point) -> float:
    """Return image-space polar angle: 0 right, 90 up, 180 left, 270 down."""
    dx = point.x - center.x
    dy = center.y - point.y
    return normalize_angle(math.degrees(math.atan2(dy, dx)))


def angle_from_dial_point(dial: DialGeometry, point: Point) -> float:
    local_x, local_y = _ellipse_local_coordinates(dial, point)
    if dial.is_ellipse:
        local_x /= max(dial.major, 1e-6)
        local_y /= max(dial.minor, 1e-6)
    return normalize_angle(math.degrees(math.atan2(local_y, local_x)))


def point_at_angle(center: Point, angle: float, radius: float) -> Point:
    radians = math.radians(angle)
    return Point(
        center.x + math.cos(radians) * radius,
        center.y - math.sin(radians) * radius,
    )


def point_on_dial(dial: DialGeometry, angle: float, radius_fraction: float = 1.0) -> Point:
    radians = math.radians(angle)
    local_x = math.cos(radians) * dial.major * radius_fraction
    local_y = math.sin(radians) * dial.minor * radius_fraction
    rotation = math.radians(dial.rotation_deg)
    x_math = local_x * math.cos(rotation) - local_y * math.sin(rotation)
    y_math = local_x * math.sin(rotation) + local_y * math.cos(rotation)
    return Point(dial.center.x + x_math, dial.center.y - y_math)


def dial_radial_fraction(dial: DialGeometry, point: Point) -> float:
    local_x, local_y = _ellipse_local_coordinates(dial, point)
    return math.hypot(local_x / max(dial.major, 1e-6), local_y / max(dial.minor, 1e-6))


def angular_distance(a: float, b: float) -> float:
    delta = abs(normalize_angle(a) - normalize_angle(b)) % 360.0
    return min(delta, 360.0 - delta)


def sweep_delta(start: float, end: float, direction: str) -> float:
    start = normalize_angle(start)
    end = normalize_angle(end)
    direction = normalize_direction(direction)
    if direction == "counterclockwise":
        return (end - start) % 360.0
    return (start - end) % 360.0


def normalize_direction(direction: str | None) -> str:
    if direction is None or direction == "auto":
        return "auto"
    cleaned = str(direction).strip().lower()
    if cleaned in {"ccw", "counter-clockwise", "counterclockwise", "anticlockwise"}:
        return "counterclockwise"
    if cleaned in {"cw", "clock-wise", "clockwise"}:
        return "clockwise"
    raise ValueError(f"unsupported direction: {direction}")


def angle_between(start: float, target: float, end: float, direction: str, tolerance: float = 1e-6) -> bool:
    span = sweep_delta(start, end, direction)
    offset = sweep_delta(start, target, direction)
    return offset <= span + tolerance


def merge_ticks_by_angle(ticks: Iterable[TickDetection], tolerance_degrees: float = 2.0) -> list[TickDetection]:
    ordered = sorted(ticks, key=lambda tick: tick.angle)
    clusters: list[list[TickDetection]] = []
    for tick in ordered:
        if not clusters or angular_distance(clusters[-1][-1].angle, tick.angle) > tolerance_degrees:
            clusters.append([tick])
        else:
            clusters[-1].append(tick)

    if len(clusters) > 1 and angular_distance(clusters[0][0].angle, clusters[-1][-1].angle) <= tolerance_degrees:
        clusters[0] = clusters[-1] + clusters[0]
        clusters.pop()

    merged: list[TickDetection] = []
    for cluster in clusters:
        weight_sum = sum(max(0.01, tick.confidence) for tick in cluster)
        sin_sum = sum(math.sin(math.radians(tick.angle)) * max(0.01, tick.confidence) for tick in cluster)
        cos_sum = sum(math.cos(math.radians(tick.angle)) * max(0.01, tick.confidence) for tick in cluster)
        angle = normalize_angle(math.degrees(math.atan2(sin_sum / weight_sum, cos_sum / weight_sum)))
        best = max(cluster, key=lambda tick: tick.confidence)
        merged.append(
            TickDetection(
                angle=angle,
                point=best.point,
                confidence=sum(t.confidence for t in cluster) / len(cluster),
                length=best.length,
                source=best.source,
            )
        )
    return sorted(merged, key=lambda tick: tick.angle)


def _ellipse_local_coordinates(dial: DialGeometry, point: Point) -> tuple[float, float]:
    x_math = point.x - dial.center.x
    y_math = dial.center.y - point.y
    rotation = math.radians(dial.rotation_deg)
    local_x = x_math * math.cos(rotation) + y_math * math.sin(rotation)
    local_y = -x_math * math.sin(rotation) + y_math * math.cos(rotation)
    return local_x, local_y
