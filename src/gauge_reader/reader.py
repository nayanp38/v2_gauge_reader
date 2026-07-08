from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

from .cloud import CloudVisionAdapter, VisionExtraction, default_cloud_adapter, forced_cloud_ocr_adapter
from .cropping import (
    dial_crop_box,
    translate_dial,
    translate_needle,
    translate_numeric_labels,
    translate_tick,
)
from .cv_pipeline import ImageReadError, LocalCVPipeline, MissingDependencyError
from .geometry import (
    angle_from_dial_point,
    angular_distance,
    label_in_dial_ring,
    minor_tick_spacing_degrees,
    needle_angle_at_tick_ring,
    point_on_dial,
)
from .interpolation import interpolate_reading
from .models import (
    DialGeometry,
    GaugeCalibration,
    GaugeObservation,
    GaugeReading,
    NeedleDetection,
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
        force_cloud_ocr: bool = False,
        crop_cloud_to_dial: bool = True,
        confidence_threshold: float = 0.65,
    ):
        self.pipeline = pipeline or LocalCVPipeline()
        self.cloud_adapter = (
            cloud_adapter
            if cloud_adapter is not None
            else forced_cloud_ocr_adapter()
            if force_cloud_ocr
            else default_cloud_adapter()
        )
        self.use_cloud = use_cloud or force_cloud_ocr
        self.force_cloud_ocr = force_cloud_ocr
        self.crop_cloud_to_dial = crop_cloud_to_dial
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
            local_failure = GaugeReading(
                reading=None,
                confidence=0.0,
                status="failed",
                reason=f"missing_image_dependencies: {exc}",
            )
            cloud_attempt = self._read_with_cloud_only(image_path, calibration_obj)
            if cloud_attempt is not None:
                cloud_observation, cloud_result = cloud_attempt
                if _prefer_cloud_result(local_failure, cloud_result):
                    return self._finalize_reading(
                        image_path,
                        cloud_observation,
                        cloud_result,
                        debug_dir,
                        used_cloud=True,
                    )
            return local_failure
        except ImageReadError as exc:
            local_failure = GaugeReading(
                reading=None,
                confidence=0.0,
                status="failed",
                reason=str(exc),
            )
            cloud_attempt = self._read_with_cloud_only(image_path, calibration_obj)
            if cloud_attempt is not None:
                cloud_observation, cloud_result = cloud_attempt
                if _prefer_cloud_result(local_failure, cloud_result):
                    return self._finalize_reading(
                        image_path,
                        cloud_observation,
                        cloud_result,
                        debug_dir,
                        used_cloud=True,
                    )
            return local_failure

        used_cloud = False
        cloud_mode = "fallback"
        if self.force_cloud_ocr:
            cloud_observation = self._augment_with_cloud(
                image_path,
                observation,
                force_ocr=True,
            )
            if cloud_observation is not observation:
                observation = cloud_observation
                used_cloud = True
                cloud_mode = "ocr"

        result = self._read_from_observation(observation, calibration_obj)
        if (
            self.use_cloud
            and not self.force_cloud_ocr
            and (result.reading is None or result.confidence < self.confidence_threshold)
        ):
            cloud_observation = self._augment_with_cloud(image_path, observation)
            if cloud_observation is not observation:
                cloud_result = self._read_from_observation(cloud_observation, calibration_obj)
                if _prefer_cloud_result(result, cloud_result):
                    observation = cloud_observation
                    result = cloud_result
                    used_cloud = True

        return self._finalize_reading(
            image_path,
            observation,
            result,
            debug_dir,
            used_cloud=used_cloud,
            cloud_mode=cloud_mode,
        )

    def _finalize_reading(
        self,
        image_path: str | Path,
        observation: GaugeObservation,
        result: GaugeReading,
        debug_dir: str | Path | None,
        *,
        used_cloud: bool,
        cloud_mode: str = "fallback",
    ) -> GaugeReading:
        debug = dict(result.debug)
        if debug_dir is not None:
            debug.update(self._write_debug(image_path, observation, result, debug_dir))

        reason = result.reason
        if used_cloud and reason == "ok":
            reason = "ok_cloud_ocr" if cloud_mode == "ocr" else "ok_cloud_fallback"
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

    def _read_with_cloud_only(
        self,
        image_path: str | Path,
        calibration: GaugeCalibration | None,
    ) -> tuple[GaugeObservation, GaugeReading] | None:
        if not self.use_cloud:
            return None
        empty_observation = GaugeObservation(dial=None, needle=None)
        cloud_observation = self._augment_with_cloud(image_path, empty_observation)
        if cloud_observation is empty_observation:
            return None
        return cloud_observation, self._read_from_observation(cloud_observation, calibration)

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
                needle_angle=_effective_needle_angle(observation),
                unit=calibration.unit if calibration else observation.unit,
            )

        needle_angle = _effective_needle_angle(observation)
        minor_spacing = minor_tick_spacing_degrees(observation.ticks)
        interpolation = interpolate_reading(
            needle_angle,
            value_ticks,
            direction=calibration.direction if calibration else None,
            minor_tick_spacing=minor_spacing,
        )
        if interpolation is None:
            confidence = 0.35 * observation.needle.confidence + 0.15 * dial_confidence
            return GaugeReading(
                reading=None,
                confidence=confidence,
                status="failed",
                reason="no_bracketing_ticks",
                needle_angle=needle_angle,
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
            needle_angle=needle_angle,
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
        filtered_labels = filter_labels_in_dial_ring(observation.labels, observation.dial)
        return associate_labels_to_ticks(filtered_labels, observation.ticks, observation.dial)

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

    def _augment_with_cloud(
        self,
        image_path: str | Path,
        observation: GaugeObservation,
        *,
        force_ocr: bool = False,
    ) -> GaugeObservation:
        extraction = self._extract_cloud(image_path, observation)
        if (
            not extraction.labels
            and not extraction.ticks
            and extraction.unit is None
            and extraction.dial is None
            and extraction.needle is None
        ):
            if force_ocr and observation.labels:
                return GaugeObservation(
                    dial=observation.dial,
                    needle=observation.needle,
                    ticks=observation.ticks,
                    labels=[],
                    unit=observation.unit,
                    debug=observation.debug,
                    reason=observation.reason,
                )
            return observation
        dial = _choose_detection(observation.dial, extraction.dial, weak_threshold=0.58)
        needle = _choose_detection(observation.needle, extraction.needle, weak_threshold=0.62)
        needle = _needle_reprojected_to_dial(needle, dial)
        labels = extraction.labels if force_ocr else [*observation.labels, *extraction.labels]
        merged_labels = filter_labels_in_dial_ring(
            labels,
            dial,
        )
        return GaugeObservation(
            dial=dial,
            needle=needle,
            ticks=[*observation.ticks, *extraction.ticks],
            labels=merged_labels,
            unit=extraction.unit or observation.unit,
            debug=observation.debug,
            reason=observation.reason,
        )

    def _extract_cloud(self, image_path: str | Path, observation: GaugeObservation) -> VisionExtraction:
        crop = self._write_cloud_crop(image_path, observation.dial)
        if crop is None:
            return self.cloud_adapter.extract(image_path, context=_cloud_context(observation))

        crop_path, crop_offset = crop
        crop_observation = GaugeObservation(
            dial=translate_dial(observation.dial, Point(-crop_offset.x, -crop_offset.y)),
            needle=translate_needle(observation.needle, Point(-crop_offset.x, -crop_offset.y)),
            ticks=[translate_tick(tick, Point(-crop_offset.x, -crop_offset.y)) for tick in observation.ticks],
            labels=translate_numeric_labels(observation.labels, Point(-crop_offset.x, -crop_offset.y)),
            unit=observation.unit,
            debug=observation.debug,
            reason=observation.reason,
        )
        try:
            extraction = self.cloud_adapter.extract(crop_path, context=_cloud_context(crop_observation))
            return _translate_extraction(extraction, crop_offset)
        finally:
            try:
                crop_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _write_cloud_crop(self, image_path: str | Path, dial: DialGeometry | None) -> tuple[Path, Point] | None:
        if not self.crop_cloud_to_dial or dial is None:
            return None
        image_file = Path(image_path)
        if not image_file.is_file():
            return None
        try:
            import cv2
        except Exception:
            return None

        image = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        height, width = image.shape[:2]
        crop_box = dial_crop_box(dial, width, height)
        if crop_box is None:
            return None

        x0 = int(crop_box.x)
        y0 = int(crop_box.y)
        x1 = int(crop_box.x + crop_box.width)
        y1 = int(crop_box.y + crop_box.height)
        if x1 <= x0 or y1 <= y0:
            return None

        suffix = image_file.suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
            suffix = ".png"
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = Path(temp.name)
        temp.close()
        if not cv2.imwrite(str(temp_path), image[y0:y1, x0:x1]):
            temp_path.unlink(missing_ok=True)
            return None
        return temp_path, Point(crop_box.x, crop_box.y)

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


def filter_labels_in_dial_ring(
    labels: list[NumericLabel],
    dial: DialGeometry | None,
    *,
    inner_fraction: float = 0.25,
    outer_fraction: float = 0.85,
) -> list[NumericLabel]:
    if dial is None:
        return labels
    filtered: list[NumericLabel] = []
    for label in labels:
        location = label.location
        if location is None:
            continue
        if label_in_dial_ring(
            dial,
            location,
            inner_fraction=inner_fraction,
            outer_fraction=outer_fraction,
        ):
            filtered.append(label)
    return filtered


def associate_labels_to_ticks(
    labels: list[NumericLabel],
    ticks: list[TickDetection],
    dial: DialGeometry | None,
) -> list[ValueTick]:
    """Map OCR/VLM labels to nearby tick marks; value angles come from ticks only."""
    value_ticks: list[ValueTick] = []
    normalized_ticks = _ticks_with_angles(ticks, dial)
    seen: set[tuple[float, int]] = set()
    for label in labels:
        label_point = label.location
        if label_point is None or dial is None:
            continue
        label_angle = angle_from_dial_point(dial, label_point)
        if normalized_ticks:
            nearest_tick = _nearest_tick(label_angle, normalized_ticks)
            if nearest_tick is not None and angular_distance(label_angle, nearest_tick.angle) <= 5.0:
                angle = nearest_tick.angle
                point = nearest_tick.point
                confidence = (label.confidence * 0.7) + (nearest_tick.confidence * 0.3)
            elif label.source.startswith("cloud"):
                angle = label_angle
                point = point_on_dial(dial, label_angle, 0.92)
                confidence = label.confidence * 0.55
            else:
                continue
        elif label.source.startswith("cloud"):
            angle = label_angle
            point = point_on_dial(dial, label_angle, 0.92)
            confidence = label.confidence * 0.65
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


def _prefer_cloud_result(local: GaugeReading, cloud: GaugeReading) -> bool:
    if cloud.reading is not None and local.reading is None:
        return True
    if cloud.reading is None and local.reading is not None:
        return False
    return cloud.confidence >= local.confidence


def _has_calibration_ticks(calibration: GaugeCalibration | None) -> bool:
    return calibration is not None and bool(calibration.ticks)


def _cloud_context(observation: GaugeObservation) -> dict[str, Any]:
    return {
        "needle_angle": observation.needle.angle if observation.needle else None,
        "dial": observation.dial.to_dict() if observation.dial else None,
    }


def _translate_extraction(extraction: VisionExtraction, offset: Point) -> VisionExtraction:
    return VisionExtraction(
        labels=translate_numeric_labels(extraction.labels, offset),
        ticks=[translate_tick(tick, offset) for tick in extraction.ticks],
        dial=translate_dial(extraction.dial, offset),
        needle=translate_needle(extraction.needle, offset),
        unit=extraction.unit,
    )


def _choose_detection(local: Any | None, cloud: Any | None, *, weak_threshold: float) -> Any | None:
    if cloud is None:
        return local
    if local is None:
        return cloud
    local_confidence = float(getattr(local, "confidence", 0.0))
    cloud_confidence = float(getattr(cloud, "confidence", 0.0))
    if local_confidence < weak_threshold or cloud_confidence > local_confidence + 0.10:
        return cloud
    return local


def _effective_needle_angle(observation: GaugeObservation) -> float:
    needle = observation.needle
    if needle is None:
        return float("nan")
    dial = observation.dial
    if dial is not None and needle.end is not None:
        return needle_angle_at_tick_ring(dial, needle)
    return needle.angle


def _needle_reprojected_to_dial(
    needle: NeedleDetection | None,
    dial: DialGeometry | None,
) -> NeedleDetection | None:
    if needle is None or dial is None or needle.end is None:
        return needle
    reprojected = NeedleDetection(
        angle=needle_angle_at_tick_ring(dial, needle),
        confidence=needle.confidence,
        start=needle.start or dial.center,
        end=needle.end,
        source=needle.source,
    )
    return reprojected
