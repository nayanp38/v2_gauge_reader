from __future__ import annotations

import math

from .models import BBox, DialGeometry, NeedleDetection, NumericLabel, Point, TickDetection


def dial_crop_box(
    dial: DialGeometry | None,
    image_width: int,
    image_height: int,
    *,
    padding_fraction: float = 0.12,
    min_padding_px: float = 10.0,
) -> BBox | None:
    if dial is None or image_width <= 0 or image_height <= 0:
        return None

    center = dial.ellipse_center
    rotation = math.radians(dial.rotation_deg)
    half_width = abs(math.cos(rotation)) * dial.major + abs(math.sin(rotation)) * dial.minor
    half_height = abs(math.sin(rotation)) * dial.major + abs(math.cos(rotation)) * dial.minor
    padding = max(float(min_padding_px), max(dial.major, dial.minor, dial.radius) * padding_fraction)

    x0 = max(0, int(math.floor(center.x - half_width - padding)))
    y0 = max(0, int(math.floor(center.y - half_height - padding)))
    x1 = min(int(image_width), int(math.ceil(center.x + half_width + padding)))
    y1 = min(int(image_height), int(math.ceil(center.y + half_height + padding)))
    if x1 <= x0 or y1 <= y0:
        return None
    return BBox(float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def translate_point(point: Point | None, offset: Point) -> Point | None:
    if point is None:
        return None
    return Point(point.x + offset.x, point.y + offset.y)


def translate_bbox(bbox: BBox | None, offset: Point) -> BBox | None:
    if bbox is None:
        return None
    return BBox(bbox.x + offset.x, bbox.y + offset.y, bbox.width, bbox.height)


def translate_numeric_label(label: NumericLabel, offset: Point) -> NumericLabel:
    return NumericLabel(
        value=label.value,
        text=label.text,
        bbox=translate_bbox(label.bbox, offset),
        center=translate_point(label.center, offset),
        confidence=label.confidence,
        source=label.source,
    )


def translate_tick(tick: TickDetection, offset: Point) -> TickDetection:
    return TickDetection(
        angle=tick.angle,
        point=translate_point(tick.point, offset),
        confidence=tick.confidence,
        length=tick.length,
        source=tick.source,
    )


def translate_dial(dial: DialGeometry | None, offset: Point) -> DialGeometry | None:
    if dial is None:
        return None
    return DialGeometry(
        center=translate_point(dial.center, offset) or dial.center,
        radius=dial.radius,
        confidence=dial.confidence,
        major_radius=dial.major_radius,
        minor_radius=dial.minor_radius,
        rotation_deg=dial.rotation_deg,
        boundary_center=translate_point(dial.boundary_center, offset),
        source=dial.source,
    )


def translate_needle(needle: NeedleDetection | None, offset: Point) -> NeedleDetection | None:
    if needle is None:
        return None
    return NeedleDetection(
        angle=needle.angle,
        confidence=needle.confidence,
        start=translate_point(needle.start, offset),
        end=translate_point(needle.end, offset),
        source=needle.source,
    )


def translate_numeric_labels(labels: list[NumericLabel], offset: Point) -> list[NumericLabel]:
    return [translate_numeric_label(label, offset) for label in labels]
