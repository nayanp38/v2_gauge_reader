from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .cloud import CloudVisionAdapter, default_cloud_adapter
from .cv_pipeline import ImageReadError, LocalCVPipeline, MissingDependencyError
from .geometry import angle_from_dial_point, angular_distance
from .interpolation import interpolate_reading
from .models import (
    DialGeometry,
    GaugeCalibration,
    GaugeObservation,
    GaugeReading,
    NumericLabel,
    Point,
    TickDetection,
    ValueTick,
)


class GaugeReader:
    def __init__(
        self,
        *,
        pipeline: Any | None = None,
        cloud_adapter: CloudVisionAdapter | None = None,
        use_cloud: bool = True,
        confidence_threshold: float = 0.65,
    ):
        self.pipeline = pipeline or LocalCVPipeline()
        self.cloud_adapter = cloud_adapter if cloud_adapter is not None else default_cloud_adapter()
        self.use_cloud = use_cloud
        self.confidence_threshold = confidence_threshold

    def read(
        self,
        image_path: str | Path,
        calibration: GaugeCalibration | dict[str, Any] | str | Path | None = None,
        debug_dir: str | Path | None = None,
    ) -> GaugeReading:
        calibration_obj = self._load_calibration(calibration)
        try:
            observation = self.pipeline.process(image_path)
        except MissingDependencyError as exc:
            return GaugeReading(
                reading=None,
                confidence=0.0,
                status="failed",
                reason=f"missing_image_dependencies: {exc}",
            )
        except ImageReadError as exc:
            return GaugeReading(
                reading=None,
                confidence=0.0,
                status="failed",
                reason=str(exc),
            )

        result = self._read_from_observation(observation, calibration_obj)
        used_cloud = False
        if (
            self.use_cloud
            and calibration_obj is None
            and (result.reading is None or result.confidence < self.confidence_threshold)
        ):
            cloud_observation = self._augment_with_cloud(image_path, observation)
            if cloud_observation is not observation:
                cloud_result = self._read_from_observation(cloud_observation, calibration_obj)
                if cloud_result.confidence >= result.confidence:
                    observation = cloud_observation
                    result = cloud_result
                    used_cloud = True

        debug = dict(result.debug)
        if debug_dir is not None:
            debug.update(self._write_debug(image_path, observation, result, debug_dir))

        reason = result.reason
        if used_cloud and reason == "ok":
            reason = "ok_cloud_fallback"
        return GaugeReading(
            reading=result.reading,
            confidence=result.confidence,
            status=result.status,
            reason=reason,
            needle_angle=result.needle_angle,
            unit=result.unit,
            lower_tick=result.lower_tick,
            upper_tick=result.upper_tick,
            debug=debug,
        )

    def _read_from_observation(
        self,
        observation: GaugeObservation,
        calibration: GaugeCalibration | None,
    ) -> GaugeReading:
        if observation.needle is None:
            return GaugeReading(
                reading=None,
                confidence=0.0,
                status="failed",
                reason="needle_not_found",
                unit=calibration.unit if calibration else observation.unit,
            )

        dial_confidence = observation.dial.confidence if observation.dial is not None else 0.0
        value_ticks = self._value_ticks(observation, calibration)
        if len(value_ticks) < 2:
            confidence = 0.35 * observation.needle.confidence + 0.15 * dial_confidence
            return GaugeReading(
                reading=None,
                confidence=confidence,
                status="failed",
                reason="not_enough_known_ticks",
                needle_angle=observation.needle.angle,
                unit=calibration.unit if calibration else observation.unit,
            )

        interpolation = interpolate_reading(
            observation.needle.angle,
            value_ticks,
            direction=calibration.direction if calibration else None,
        )
        if interpolation is None:
            confidence = 0.35 * observation.needle.confidence + 0.15 * dial_confidence
            return GaugeReading(
                reading=None,
                confidence=confidence,
                status="failed",
                reason="no_bracketing_ticks",
                needle_angle=observation.needle.angle,
                unit=calibration.unit if calibration else observation.unit,
            )

        confidence = _combined_confidence(
            needle=observation.needle.confidence,
            dial=dial_confidence,
            interpolation=interpolation.confidence,
            value_ticks=value_ticks,
        )
        status = "ok" if confidence >= self.confidence_threshold else "low_confidence"
        return GaugeReading(
            reading=interpolation.reading,
            confidence=confidence,
            status=status,
            reason="ok" if status == "ok" else "low_confidence",
            needle_angle=observation.needle.angle,
            unit=calibration.unit if calibration and calibration.unit else observation.unit,
            lower_tick=interpolation.lower_tick,
            upper_tick=interpolation.upper_tick,
        )

    def _value_ticks(
        self,
        observation: GaugeObservation,
        calibration: GaugeCalibration | None,
    ) -> list[ValueTick]:
        if calibration is not None and calibration.ticks:
            return self._value_ticks_from_calibration(observation.dial, calibration)
        return associate_labels_to_ticks(observation.labels, observation.ticks, observation.dial)

    def _value_ticks_from_calibration(
        self,
        dial: DialGeometry | None,
        calibration: GaugeCalibration,
    ) -> list[ValueTick]:
        value_ticks: list[ValueTick] = []
        for tick in calibration.ticks:
            angle = tick.angle
            if angle is None and tick.point is not None and dial is not None:
                angle = angle_from_dial_point(dial, tick.point)
            if angle is None:
                continue
            value_ticks.append(
                ValueTick(
                    value=tick.value,
                    angle=angle,
                    point=tick.point,
                    confidence=tick.confidence,
                    source="calibration",
                )
            )
        return value_ticks

    def _augment_with_cloud(self, image_path: str | Path, observation: GaugeObservation) -> GaugeObservation:
        extraction = self.cloud_adapter.extract(
            image_path,
            context={
                "needle_angle": observation.needle.angle if observation.needle else None,
                "dial": observation.dial.to_dict() if observation.dial else None,
            },
        )
        if not extraction.labels and not extraction.ticks and extraction.unit is None:
            return observation
        return GaugeObservation(
            dial=observation.dial,
            needle=observation.needle,
            ticks=[*observation.ticks, *extraction.ticks],
            labels=[*observation.labels, *extraction.labels],
            unit=extraction.unit or observation.unit,
            debug=observation.debug,
            reason=observation.reason,
        )

    def _write_debug(
        self,
        image_path: str | Path,
        observation: GaugeObservation,
        result: GaugeReading,
        debug_dir: str | Path,
    ) -> dict[str, str]:
        writer = getattr(self.pipeline, "write_debug_overlay", None)
        if writer is None:
            return {}
        try:
            return writer(
                image_path,
                observation,
                debug_dir,
                lower_angle=result.lower_tick.angle if result.lower_tick else None,
                upper_angle=result.upper_tick.angle if result.upper_tick else None,
            )
        except MissingDependencyError:
            return {}

    def _load_calibration(
        self,
        calibration: GaugeCalibration | dict[str, Any] | str | Path | None,
    ) -> GaugeCalibration | None:
        if calibration is None:
            return None
        if isinstance(calibration, GaugeCalibration):
            return calibration
        if isinstance(calibration, (str, Path)):
            with Path(calibration).open("r", encoding="utf-8") as handle:
                return GaugeCalibration.from_mapping(json.load(handle))
        return GaugeCalibration.from_mapping(calibration)


def associate_labels_to_ticks(
    labels: list[NumericLabel],
    ticks: list[TickDetection],
    dial: DialGeometry | None,
) -> list[ValueTick]:
    value_ticks: list[ValueTick] = []
    normalized_ticks = _ticks_with_angles(ticks, dial)
    seen: set[tuple[float, int]] = set()
    for label in labels:
        label_point = label.location
        label_angle = None
        if label_point is not None and dial is not None:
            label_angle = angle_from_dial_point(dial, label_point)
        nearest_tick = _nearest_tick(label_angle, normalized_ticks)
        if nearest_tick is not None and (label_angle is None or angular_distance(label_angle, nearest_tick.angle) <= 18.0):
            angle = nearest_tick.angle
            point = nearest_tick.point
            confidence = (label.confidence * 0.7) + (nearest_tick.confidence * 0.3)
        elif label_angle is not None:
            angle = label_angle
            point = label_point
            confidence = label.confidence * 0.72
        else:
            continue
        key = (label.value, round(angle))
        if key in seen:
            continue
        seen.add(key)
        value_ticks.append(
            ValueTick(
                value=label.value,
                angle=angle,
                point=point,
                confidence=confidence,
                source=label.source,
                label=label,
            )
        )
    return value_ticks


def _nearest_tick(label_angle: float | None, ticks: list[TickDetection]) -> TickDetection | None:
    if label_angle is None or not ticks:
        return None
    return min(ticks, key=lambda tick: angular_distance(label_angle, tick.angle))


def _ticks_with_angles(ticks: list[TickDetection], dial: DialGeometry | None) -> list[TickDetection]:
    normalized: list[TickDetection] = []
    for tick in ticks:
        if math.isfinite(tick.angle):
            normalized.append(tick)
            continue
        if tick.point is not None and dial is not None:
            normalized.append(
                TickDetection(
                    angle=angle_from_dial_point(dial, tick.point),
                    point=tick.point,
                    confidence=tick.confidence,
                    length=tick.length,
                    source=tick.source,
                )
            )
    return normalized


def _combined_confidence(
    *,
    needle: float,
    dial: float,
    interpolation: float,
    value_ticks: list[ValueTick],
) -> float:
    tick_confidence = sum(tick.confidence for tick in value_ticks) / max(1, len(value_ticks))
    confidence = (
        0.35 * max(0.0, min(1.0, needle))
        + 0.15 * max(0.0, min(1.0, dial))
        + 0.35 * max(0.0, min(1.0, interpolation))
        + 0.15 * max(0.0, min(1.0, tick_confidence))
    )
    return max(0.0, min(1.0, confidence))
