from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .geometry import angle_from_dial_point, angle_from_point
from .models import BBox, DialGeometry, NeedleDetection, NumericLabel, Point, TickDetection
from .ocr import parse_numeric_text


@dataclass(frozen=True)
class VisionExtraction:
    labels: list[NumericLabel] = field(default_factory=list)
    ticks: list[TickDetection] = field(default_factory=list)
    dial: DialGeometry | None = None
    needle: NeedleDetection | None = None
    unit: str | None = None


class CloudVisionAdapter:
    def extract(self, image_path: str | Path, context: dict[str, Any] | None = None) -> VisionExtraction:
        raise NotImplementedError


class NullCloudVisionAdapter(CloudVisionAdapter):
    def extract(self, image_path: str | Path, context: dict[str, Any] | None = None) -> VisionExtraction:
        return VisionExtraction()


class CommandCloudVisionAdapter(CloudVisionAdapter):
    """Run a user-supplied command that emits structured OCR/VLM JSON."""

    def __init__(self, command: str):
        self.command = command

    @classmethod
    def from_env(cls) -> "CommandCloudVisionAdapter | None":
        command = os.environ.get("GAUGE_READER_CLOUD_COMMAND")
        if not command:
            return None
        return cls(command)

    def extract(self, image_path: str | Path, context: dict[str, Any] | None = None) -> VisionExtraction:
        image = str(image_path)
        if "{image}" in self.command:
            command = shlex.split(self.command.format(image=image))
        else:
            command = shlex.split(self.command) + [image]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            return VisionExtraction()
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return VisionExtraction()
        return extraction_from_mapping(payload, source="cloud_command")


def default_cloud_adapter() -> CloudVisionAdapter:
    return CommandCloudVisionAdapter.from_env() or NullCloudVisionAdapter()


def extraction_from_mapping(payload: dict[str, Any], source: str = "cloud") -> VisionExtraction:
    labels = [_label_from_mapping(item, source) for item in payload.get("labels", [])]
    ticks = [_tick_from_mapping(item, source) for item in payload.get("ticks", [])]
    dial = _dial_from_mapping(payload.get("dial"), source)
    needle = _needle_from_mapping(payload.get("needle"), dial, source)
    return VisionExtraction(
        labels=[label for label in labels if label is not None],
        ticks=[tick for tick in ticks if tick is not None],
        dial=dial,
        needle=needle,
        unit=payload.get("unit"),
    )


def _label_from_mapping(item: dict[str, Any], source: str) -> NumericLabel | None:
    text = str(item.get("text", item.get("value", "")))
    value = item.get("value")
    if value is None:
        value = parse_numeric_text(text)
    if value is None:
        return None
    bbox = None
    center = None
    if item.get("bbox") is not None:
        try:
            bbox = BBox.from_any(item["bbox"])
        except ValueError:
            bbox = None
    if item.get("center") is not None:
        try:
            center = Point.from_sequence(item["center"])
        except ValueError:
            center = None
    return NumericLabel(
        value=float(value),
        text=text,
        bbox=bbox,
        center=center,
        confidence=float(item.get("confidence", 0.75)),
        source=source,
    )


def _tick_from_mapping(item: dict[str, Any], source: str) -> TickDetection | None:
    angle = item.get("angle", item.get("angle_deg"))
    point = None
    if item.get("point") is not None:
        try:
            point = Point.from_sequence(item["point"])
        except ValueError:
            point = None
    if angle is None and point is None:
        return None
    return TickDetection(
        angle=float(angle) if angle is not None else float("nan"),
        point=point,
        confidence=float(item.get("confidence", 0.75)),
        source=source,
    )


def _dial_from_mapping(item: Any, source: str) -> DialGeometry | None:
    if not isinstance(item, dict):
        return None

    center = _point_from_any(item.get("center"))
    bbox = None
    if item.get("bbox") is not None:
        try:
            bbox = BBox.from_any(item["bbox"])
        except ValueError:
            bbox = None
    if center is None and bbox is not None:
        center = bbox.center

    radius = _float_or_none(item.get("radius"))
    major_radius = _float_or_none(item.get("major_radius", item.get("major")))
    minor_radius = _float_or_none(item.get("minor_radius", item.get("minor")))
    if bbox is not None:
        major_radius = major_radius if major_radius is not None else max(bbox.width, bbox.height) / 2.0
        minor_radius = minor_radius if minor_radius is not None else min(bbox.width, bbox.height) / 2.0
        radius = radius if radius is not None else (bbox.width + bbox.height) / 4.0

    if center is None or radius is None or radius <= 0:
        return None
    return DialGeometry(
        center=center,
        radius=radius,
        confidence=_confidence(item),
        major_radius=major_radius,
        minor_radius=minor_radius,
        rotation_deg=float(item.get("rotation_deg", item.get("rotation", 0.0))),
        source=f"{source}_dial",
    )


def _needle_from_mapping(item: Any, dial: DialGeometry | None, source: str) -> NeedleDetection | None:
    if not isinstance(item, dict):
        return None

    start = _point_from_any(item.get("base", item.get("start")))
    tip = _point_from_any(item.get("tip", item.get("end")))
    angle = _float_or_none(item.get("angle", item.get("angle_deg")))
    if angle is None and tip is not None and dial is not None:
        angle = angle_from_dial_point(dial, tip)
        if start is None:
            start = dial.center
    elif angle is None and start is not None and tip is not None:
        angle = angle_from_point(start, tip)
    if angle is None:
        return None

    return NeedleDetection(
        angle=angle,
        confidence=_confidence(item),
        start=start,
        end=tip,
        source=f"{source}_needle",
    )


def _point_from_any(value: Any) -> Point | None:
    if value is None:
        return None
    if isinstance(value, Point):
        return value
    if isinstance(value, dict):
        if value.get("x") is None or value.get("y") is None:
            return None
        return Point(float(value["x"]), float(value["y"]))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return Point.from_sequence(value)
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _confidence(item: dict[str, Any]) -> float:
    value = _float_or_none(item.get("confidence"))
    if value is None:
        return 0.75
    return max(0.0, min(1.0, value))
