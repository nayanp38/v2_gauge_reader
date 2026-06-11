from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not isfinite(number):
        return None
    return number


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @classmethod
    def from_sequence(cls, values: list[float] | tuple[float, float]) -> "Point":
        if len(values) != 2:
            raise ValueError("point must contain exactly two numbers")
        return cls(float(values[0]), float(values[1]))

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y}

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class BBox:
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_any(cls, value: Any) -> "BBox":
        if isinstance(value, dict):
            return cls(
                float(value["x"]),
                float(value["y"]),
                float(value.get("width", value.get("w"))),
                float(value.get("height", value.get("h"))),
            )
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return cls(float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        raise ValueError("bbox must be [x, y, width, height] or an object with x/y/width/height")

    @property
    def center(self) -> Point:
        return Point(self.x + self.width / 2.0, self.y + self.height / 2.0)

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class DialGeometry:
    center: Point
    radius: float
    confidence: float = 0.0
    major_radius: float | None = None
    minor_radius: float | None = None
    rotation_deg: float = 0.0
    boundary_center: Point | None = None
    source: str = "circle"

    @property
    def major(self) -> float:
        return self.major_radius if self.major_radius is not None else self.radius

    @property
    def minor(self) -> float:
        return self.minor_radius if self.minor_radius is not None else self.radius

    @property
    def ellipse_center(self) -> Point:
        return self.boundary_center if self.boundary_center is not None else self.center

    @property
    def is_ellipse(self) -> bool:
        return abs(self.major - self.minor) > max(2.0, self.radius * 0.03)

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": self.center.to_dict(),
            "radius": self.radius,
            "confidence": _clamp01(self.confidence),
            "source": self.source,
            "ellipse": {
                "center": self.ellipse_center.to_dict(),
                "major_radius": self.major,
                "minor_radius": self.minor,
                "rotation_deg": self.rotation_deg,
                "is_ellipse": self.is_ellipse,
            },
        }


@dataclass(frozen=True)
class NeedleDetection:
    angle: float
    confidence: float
    start: Point | None = None
    end: Point | None = None
    source: str = "local_cv"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "angle": self.angle,
            "confidence": _clamp01(self.confidence),
            "source": self.source,
        }
        if self.start is not None:
            data["start"] = self.start.to_dict()
        if self.end is not None:
            data["end"] = self.end.to_dict()
        return data


@dataclass(frozen=True)
class TickDetection:
    angle: float
    point: Point | None = None
    confidence: float = 0.0
    length: float | None = None
    source: str = "local_cv"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "angle": self.angle,
            "confidence": _clamp01(self.confidence),
            "source": self.source,
        }
        if self.point is not None:
            data["point"] = self.point.to_dict()
        if self.length is not None:
            data["length"] = self.length
        return data


@dataclass(frozen=True)
class NumericLabel:
    value: float
    text: str
    bbox: BBox | None = None
    center: Point | None = None
    confidence: float = 0.0
    source: str = "local_ocr"

    @property
    def location(self) -> Point | None:
        if self.center is not None:
            return self.center
        if self.bbox is not None:
            return self.bbox.center
        return None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "value": self.value,
            "text": self.text,
            "confidence": _clamp01(self.confidence),
            "source": self.source,
        }
        if self.bbox is not None:
            data["bbox"] = self.bbox.to_dict()
        if self.center is not None:
            data["center"] = self.center.to_dict()
        return data


@dataclass(frozen=True)
class ValueTick:
    value: float
    angle: float
    point: Point | None = None
    confidence: float = 0.0
    source: str = "ocr"
    label: NumericLabel | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "value": self.value,
            "angle": self.angle,
            "confidence": _clamp01(self.confidence),
            "source": self.source,
        }
        if self.point is not None:
            data["point"] = self.point.to_dict()
        if self.label is not None:
            data["label"] = self.label.to_dict()
        return data


@dataclass(frozen=True)
class GaugeObservation:
    dial: DialGeometry | None
    needle: NeedleDetection | None
    ticks: list[TickDetection] = field(default_factory=list)
    labels: list[NumericLabel] = field(default_factory=list)
    unit: str | None = None
    debug: dict[str, str] = field(default_factory=dict)
    reason: str | None = None


@dataclass(frozen=True)
class InterpolationResult:
    reading: float
    confidence: float
    lower_tick: ValueTick
    upper_tick: ValueTick
    span_degrees: float
    direction: str
    angular_fraction: float


@dataclass(frozen=True)
class GaugeReading:
    reading: float | None
    confidence: float
    status: str
    reason: str
    needle_angle: float | None = None
    unit: str | None = None
    lower_tick: ValueTick | None = None
    upper_tick: ValueTick | None = None
    debug: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "reading": self.reading,
            "unit": self.unit,
            "confidence": _clamp01(self.confidence),
            "status": self.status,
            "reason": self.reason,
            "needle_angle": self.needle_angle,
            "lower_tick": self.lower_tick.to_dict() if self.lower_tick is not None else None,
            "upper_tick": self.upper_tick.to_dict() if self.upper_tick is not None else None,
            "debug": dict(self.debug),
        }
        if self.debug:
            data["debug_artifacts"] = dict(self.debug)
        return data


@dataclass(frozen=True)
class CalibrationTick:
    value: float
    angle: float | None = None
    point: Point | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class GaugeCalibration:
    ticks: list[CalibrationTick]
    unit: str | None = None
    direction: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | "GaugeCalibration" | None) -> "GaugeCalibration | None":
        if data is None:
            return None
        if isinstance(data, GaugeCalibration):
            return data
        raw_ticks = data.get("ticks", [])
        ticks: list[CalibrationTick] = []
        for raw in raw_ticks:
            value = _float_or_none(raw.get("value"))
            if value is None:
                continue
            angle = _float_or_none(raw.get("angle", raw.get("angle_deg")))
            point = None
            if raw.get("point") is not None:
                point = Point.from_sequence(raw["point"])
            elif raw.get("x") is not None and raw.get("y") is not None:
                point = Point(float(raw["x"]), float(raw["y"]))
            ticks.append(
                CalibrationTick(
                    value=value,
                    angle=angle,
                    point=point,
                    confidence=_clamp01(float(raw.get("confidence", 1.0))),
                )
            )
        direction = data.get("direction")
        if direction is not None:
            direction = str(direction).lower()
        return cls(ticks=ticks, unit=data.get("unit"), direction=direction)
