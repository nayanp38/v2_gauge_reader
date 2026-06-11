from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import BBox, NumericLabel, Point, TickDetection
from .ocr import parse_numeric_text


@dataclass(frozen=True)
class VisionExtraction:
    labels: list[NumericLabel] = field(default_factory=list)
    ticks: list[TickDetection] = field(default_factory=list)
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
    return VisionExtraction(
        labels=[label for label in labels if label is not None],
        ticks=[tick for tick in ticks if tick is not None],
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
