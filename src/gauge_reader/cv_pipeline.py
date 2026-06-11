from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from .geometry import (
    angle_from_dial_point,
    angle_from_point,
    angular_distance,
    dial_radial_fraction,
    merge_ticks_by_angle,
    needle_angle_at_tick_ring,
    normalize_angle,
    point_on_dial,
)
from .models import DialGeometry, GaugeObservation, NeedleDetection, Point, TickDetection
from .ocr import detect_labels_with_tesseract


class MissingDependencyError(RuntimeError):
    pass


class ImageReadError(RuntimeError):
    pass


class LocalCVPipeline:
    def process(self, image_path: str | Path) -> GaugeObservation:
        cv2, np = _import_cv()
        image = cv2.imread(str(image_path))
        if image is None:
            raise ImageReadError(f"could not read image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        dial = self._estimate_dial(cv2, np, gray, edges)
        needle = self._detect_needle(cv2, np, gray, edges, dial)
        ticks = self._detect_ticks(cv2, np, edges, dial)
        labels = detect_labels_with_tesseract(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return GaugeObservation(
            dial=dial,
            needle=needle,
            ticks=ticks,
            labels=labels,
            reason=None,
        )

    def write_debug_overlay(
        self,
        image_path: str | Path,
        observation: GaugeObservation,
        debug_dir: str | Path,
        lower_angle: float | None = None,
        upper_angle: float | None = None,
    ) -> dict[str, str]:
        cv2, _ = _import_cv()
        image = cv2.imread(str(image_path))
        if image is None:
            return {}
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)

        if observation.dial is not None:
            dial = observation.dial
            boundary_center = _int_point(dial.ellipse_center)
            if dial.is_ellipse:
                cv2.ellipse(
                    image,
                    boundary_center,
                    (int(round(dial.major)), int(round(dial.minor))),
                    -dial.rotation_deg,
                    0,
                    360,
                    (80, 180, 255),
                    2,
                )
            else:
                cv2.circle(image, boundary_center, int(dial.radius), (80, 180, 255), 2)
            cv2.circle(image, _int_point(dial.center), 4, (80, 180, 255), -1)

        if observation.dial is not None and observation.needle is not None:
            dial = observation.dial
            needle = observation.needle
            pivot = needle.start if needle.start is not None else dial.center
            if needle.end is not None:
                tick_angle = needle_angle_at_tick_ring(dial, needle)
                ring_point = point_on_dial(dial, tick_angle, 0.92)
                cv2.line(image, _int_point(pivot), _int_point(needle.end), (0, 0, 180), 2)
                cv2.line(image, _int_point(pivot), _int_point(ring_point), (0, 0, 255), 3)
            else:
                end = point_on_dial(dial, needle.angle, 0.92)
                cv2.line(image, _int_point(dial.center), _int_point(end), (0, 0, 255), 3)

        for tick in observation.ticks:
            if observation.dial is not None:
                outer = point_on_dial(observation.dial, tick.angle, 0.98)
                inner = point_on_dial(observation.dial, tick.angle, 0.86)
                cv2.line(image, _int_point(inner), _int_point(outer), (0, 180, 0), 1)
            elif tick.point is not None:
                cv2.circle(image, _int_point(tick.point), 2, (0, 180, 0), -1)

        for label in observation.labels:
            if label.bbox is not None:
                p1 = (int(label.bbox.x), int(label.bbox.y))
                p2 = (int(label.bbox.x + label.bbox.width), int(label.bbox.y + label.bbox.height))
                cv2.rectangle(image, p1, p2, (255, 0, 180), 1)
                cv2.putText(image, label.text, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 180), 1)
            elif label.location is not None:
                cv2.circle(image, _int_point(label.location), 3, (255, 0, 180), -1)

        if observation.dial is not None:
            for angle, color in [(lower_angle, (255, 180, 0)), (upper_angle, (255, 255, 0))]:
                if angle is None:
                    continue
                end = point_on_dial(observation.dial, angle, 1.0)
                cv2.line(image, _int_point(observation.dial.center), _int_point(end), color, 2)

        tick_count = len(observation.ticks)
        print(f"cv_ticks_detected: {tick_count}", file=sys.stderr)

        output = debug_path / "overlay.jpg"
        cv2.imwrite(str(output), image)
        return {"overlay": str(output)}

    def _estimate_dial(self, cv2: Any, np: Any, gray: Any, edges: Any) -> DialGeometry:
        height, width = gray.shape[:2]
        min_dim = min(width, height)
        ellipse = self._fit_outer_ellipse(cv2, np, edges, width, height)
        if ellipse is not None:
            refined_center, center_confidence = self._refine_center_from_radial_lines(cv2, np, edges, ellipse)
            if refined_center is not None:
                ellipse = DialGeometry(
                    center=refined_center,
                    radius=ellipse.radius,
                    confidence=min(0.95, ellipse.confidence + center_confidence),
                    major_radius=ellipse.major,
                    minor_radius=ellipse.minor,
                    rotation_deg=ellipse.rotation_deg,
                    boundary_center=ellipse.ellipse_center,
                    source="ellipse_radial_refined",
                )
            return ellipse

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, min_dim // 3),
            param1=120,
            param2=30,
            minRadius=max(10, int(min_dim * 0.20)),
            maxRadius=max(12, int(min_dim * 0.55)),
        )
        if circles is not None and len(circles) > 0:
            circle = np.round(circles[0][0]).astype("int")
            return DialGeometry(
                center=Point(float(circle[0]), float(circle[1])),
                radius=float(circle[2]),
                confidence=0.82,
                source="hough_circle",
            )
        return DialGeometry(
            center=Point(width / 2.0, height / 2.0),
            radius=min_dim * 0.45,
            confidence=0.35,
            source="image_center_fallback",
        )

    def _fit_outer_ellipse(self, cv2: Any, np: Any, edges: Any, width: int, height: int) -> DialGeometry | None:
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        min_dim = min(width, height)
        image_center = Point(width / 2.0, height / 2.0)
        best: tuple[float, DialGeometry] | None = None
        for contour in contours:
            if len(contour) < 40:
                continue
            try:
                (cx, cy), (axis_a, axis_b), raw_rotation = cv2.fitEllipse(contour)
            except Exception:
                continue
            major_diameter = max(float(axis_a), float(axis_b))
            minor_diameter = min(float(axis_a), float(axis_b))
            if major_diameter < min_dim * 0.35 or major_diameter > min_dim * 1.15:
                continue
            if minor_diameter < min_dim * 0.18 or minor_diameter > major_diameter:
                continue
            major_radius = major_diameter / 2.0
            minor_radius = minor_diameter / 2.0
            axis_ratio = minor_radius / max(major_radius, 1.0)
            if axis_ratio < 0.35:
                continue
            center = Point(float(cx), float(cy))
            center_prior = max(0.0, 1.0 - _distance(center, image_center) / max(1.0, min_dim * 0.45))
            area_fraction = min(1.0, (major_radius * minor_radius) / max(1.0, (min_dim * 0.45) ** 2))
            perimeter = cv2.arcLength(contour, True)
            perimeter_score = min(1.0, perimeter / max(1.0, math.pi * (major_radius + minor_radius)))
            score = 0.45 * area_fraction + 0.25 * axis_ratio + 0.20 * center_prior + 0.10 * perimeter_score
            rotation = _opencv_ellipse_rotation_to_image_polar(axis_a, axis_b, raw_rotation)
            dial = DialGeometry(
                center=center,
                radius=(major_radius + minor_radius) / 2.0,
                confidence=max(0.45, min(0.86, score)),
                major_radius=major_radius,
                minor_radius=minor_radius,
                rotation_deg=rotation,
                boundary_center=center,
                source="fit_ellipse",
            )
            if best is None or score > best[0]:
                best = (score, dial)
        return best[1] if best is not None else None

    def _refine_center_from_radial_lines(self, cv2: Any, np: Any, edges: Any, dial: DialGeometry) -> tuple[Point | None, float]:
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(20, int(dial.radius * 0.12)),
            minLineLength=max(10, int(dial.minor * 0.08)),
            maxLineGap=max(4, int(dial.radius * 0.04)),
        )
        if lines is None:
            return None, 0.0

        candidates: list[tuple[Point, Point]] = []
        for raw in lines[:, 0, :]:
            start = Point(float(raw[0]), float(raw[1]))
            end = Point(float(raw[2]), float(raw[3]))
            midpoint = Point((start.x + end.x) / 2.0, (start.y + end.y) / 2.0)
            length = _distance(start, end)
            if length < dial.minor * 0.06 or length > dial.major * 0.95:
                continue
            radial_fraction = dial_radial_fraction(dial, midpoint)
            if radial_fraction < 0.16 or radial_fraction > 1.12:
                continue
            image_radial_angle = angle_from_point(dial.center, midpoint)
            segment_angle = angle_from_point(start, end)
            if angular_distance(image_radial_angle, segment_angle) > 25.0 and angular_distance(image_radial_angle + 180.0, segment_angle) > 25.0:
                continue
            candidates.append((start, end))

        if len(candidates) < 3:
            return None, 0.0

        center = _least_squares_line_intersection(np, candidates)
        if center is None:
            return None, 0.0
        center_offset = dial_radial_fraction(dial, center)
        if center_offset > 0.32:
            return None, 0.0
        residuals = [_distance_point_to_segment(center, start, end) for start, end in candidates]
        residual = float(np.median(residuals))
        if residual > dial.minor * 0.12:
            return None, 0.0
        confidence = min(0.12, 0.03 + len(candidates) / 100.0)
        return center, confidence

    def _detect_needle(self, cv2: Any, np: Any, gray: Any, edges: Any, dial: DialGeometry) -> NeedleDetection | None:
        radius = dial.radius
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(30, int(radius * 0.22)),
            minLineLength=max(20, int(radius * 0.25)),
            maxLineGap=max(6, int(radius * 0.05)),
        )
        best: tuple[float, Point, Point] | None = None
        if lines is not None:
            for raw in lines[:, 0, :]:
                start = Point(float(raw[0]), float(raw[1]))
                end = Point(float(raw[2]), float(raw[3]))
                length = _distance(start, end)
                if length <= 0:
                    continue
                center_distance = _distance_point_to_segment(dial.center, start, end)
                far_fraction = max(dial_radial_fraction(dial, start), dial_radial_fraction(dial, end))
                radial_score = min(1.0, far_fraction / 0.78)
                center_score = max(0.0, 1.0 - center_distance / max(1.0, dial.minor * 0.12))
                length_score = min(1.0, length / max(1.0, dial.major * 0.8))
                score = 0.45 * center_score + 0.35 * length_score + 0.20 * radial_score
                if best is None or score > best[0]:
                    best = (score, start, end)

        if best is not None and best[0] >= 0.35:
            score, start, end = best
            needle_tip = start if _distance(dial.center, start) >= _distance(dial.center, end) else end
            return NeedleDetection(
                angle=angle_from_dial_point(dial, needle_tip),
                confidence=max(0.0, min(1.0, score)),
                start=dial.center,
                end=needle_tip,
            )

        return self._detect_needle_by_radial_darkness(np, gray, dial)

    def _detect_needle_by_radial_darkness(self, np: Any, gray: Any, dial: DialGeometry) -> NeedleDetection | None:
        height, width = gray.shape[:2]
        best_angle = None
        best_score = -1.0
        radius_fractions = np.linspace(0.12, 0.86, 80)
        for angle in np.linspace(0.0, 359.5, 720):
            points = [point_on_dial(dial, float(angle), float(fraction)) for fraction in radius_fractions]
            xs = np.rint([point.x for point in points]).astype(int)
            ys = np.rint([point.y for point in points]).astype(int)
            mask = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            if mask.sum() < 20:
                continue
            values = gray[ys[mask], xs[mask]]
            darkness = 1.0 - float(values.mean()) / 255.0
            contrast = float(values.std()) / 255.0
            score = darkness * 0.75 + contrast * 0.25
            if score > best_score:
                best_angle = float(angle)
                best_score = score
        if best_angle is None or best_score < 0.18:
            return None
        return NeedleDetection(
            angle=best_angle,
            confidence=max(0.15, min(0.55, best_score)),
            start=dial.center,
            end=point_on_dial(dial, best_angle, 0.86),
            source="local_radial_scan",
        )

    def _detect_ticks(self, cv2: Any, np: Any, edges: Any, dial: DialGeometry) -> list[TickDetection]:
        radius = dial.radius
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(15, int(radius * 0.12)),
            minLineLength=max(6, int(radius * 0.035)),
            maxLineGap=max(3, int(radius * 0.025)),
        )
        ticks: list[TickDetection] = []
        if lines is None:
            return ticks
        for raw in lines[:, 0, :]:
            start = Point(float(raw[0]), float(raw[1]))
            end = Point(float(raw[2]), float(raw[3]))
            midpoint = Point((start.x + end.x) / 2.0, (start.y + end.y) / 2.0)
            radial_fraction = dial_radial_fraction(dial, midpoint)
            if radial_fraction < 0.58 or radial_fraction > 1.08:
                continue
            length = _distance(start, end)
            if length > dial.major * 0.28:
                continue
            radial_angle = angle_from_dial_point(dial, midpoint)
            image_radial_angle = angle_from_point(dial.center, midpoint)
            segment_angle = angle_from_point(start, end)
            if angular_distance(image_radial_angle, segment_angle) > 30.0 and angular_distance(image_radial_angle + 180.0, segment_angle) > 30.0:
                continue
            ticks.append(
                TickDetection(
                    angle=radial_angle,
                    point=midpoint,
                    confidence=max(0.25, min(0.9, length / max(1.0, dial.minor * 0.12))),
                    length=length,
                    source="local_hough",
                )
            )
        return merge_ticks_by_angle(ticks, tolerance_degrees=2.5)


def _import_cv() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        raise MissingDependencyError(
            "install image dependencies with: python3 -m pip install -e '.[cv,ocr]'"
        ) from exc
    return cv2, np


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _distance_point_to_segment(point: Point, start: Point, end: Point) -> float:
    dx = end.x - start.x
    dy = end.y - start.y
    if dx == 0.0 and dy == 0.0:
        return _distance(point, start)
    t = ((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    projection = Point(start.x + t * dx, start.y + t * dy)
    return _distance(point, projection)


def _int_point(point: Point) -> tuple[int, int]:
    return (int(round(point.x)), int(round(point.y)))


def _opencv_ellipse_rotation_to_image_polar(axis_a: float, axis_b: float, raw_rotation: float) -> float:
    major_axis_rotation = float(raw_rotation)
    if axis_b > axis_a:
        major_axis_rotation += 90.0
    return normalize_angle(-major_axis_rotation)


def _least_squares_line_intersection(np: Any, lines: list[tuple[Point, Point]]) -> Point | None:
    matrix = np.zeros((2, 2), dtype=float)
    vector = np.zeros((2,), dtype=float)
    for start, end in lines:
        dx = end.x - start.x
        dy = end.y - start.y
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        normal = np.array([-dy / length, dx / length], dtype=float)
        point = np.array([start.x, start.y], dtype=float)
        outer = np.outer(normal, normal)
        matrix += outer
        vector += outer @ point
    try:
        solution = np.linalg.solve(matrix, vector)
    except Exception:
        return None
    if not np.isfinite(solution).all():
        return None
    return Point(float(solution[0]), float(solution[1]))
