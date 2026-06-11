from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gauge_reader.geometry import (
    angle_from_dial_point,
    angle_from_point,
    angular_distance,
    dial_radial_fraction,
    point_at_angle,
    point_on_dial,
    sweep_delta,
)
from gauge_reader.interpolation import interpolate_reading
from gauge_reader.models import DialGeometry, Point, ValueTick


class GeometryTests(unittest.TestCase):
    def test_image_space_angles(self) -> None:
        center = Point(100, 100)
        self.assertAlmostEqual(angle_from_point(center, Point(120, 100)), 0)
        self.assertAlmostEqual(angle_from_point(center, Point(100, 80)), 90)
        self.assertAlmostEqual(angle_from_point(center, Point(80, 100)), 180)
        self.assertAlmostEqual(angle_from_point(center, Point(100, 120)), 270)

    def test_point_at_angle_round_trip(self) -> None:
        center = Point(50, 50)
        point = point_at_angle(center, 135, 20)
        self.assertAlmostEqual(angle_from_point(center, point), 135)

    def test_elliptical_dial_angle_round_trip(self) -> None:
        dial = DialGeometry(
            center=Point(100, 100),
            radius=60,
            major_radius=80,
            minor_radius=40,
            rotation_deg=30,
            source="test_ellipse",
        )
        for angle in [0, 45, 90, 180, 270, 330]:
            point = point_on_dial(dial, angle, 0.75)
            self.assertAlmostEqual(angle_from_dial_point(dial, point), angle, places=6)
            self.assertAlmostEqual(dial_radial_fraction(dial, point), 0.75, places=6)

    def test_angular_distance_wraps(self) -> None:
        self.assertAlmostEqual(angular_distance(359, 1), 2)
        self.assertAlmostEqual(angular_distance(10, 350), 20)

    def test_sweep_delta_directions(self) -> None:
        self.assertAlmostEqual(sweep_delta(225, 315, "clockwise"), 270)
        self.assertAlmostEqual(sweep_delta(225, 315, "counterclockwise"), 90)


class InterpolationTests(unittest.TestCase):
    def test_interpolates_across_long_gauge_arc(self) -> None:
        ticks = [
            ValueTick(value=0, angle=225, confidence=1.0),
            ValueTick(value=100, angle=315, confidence=1.0),
        ]
        result = interpolate_reading(90, ticks)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.reading, 50.0)
        self.assertEqual(result.direction, "clockwise")

    def test_chooses_closest_bracketing_known_ticks(self) -> None:
        ticks = [
            ValueTick(value=0, angle=225, confidence=1.0),
            ValueTick(value=50, angle=90, confidence=1.0),
            ValueTick(value=100, angle=315, confidence=1.0),
        ]
        result = interpolate_reading(45, ticks)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.reading, 66.6666666667)
        self.assertEqual(result.lower_tick.value, 50)
        self.assertEqual(result.upper_tick.value, 100)

    def test_returns_none_without_bracket(self) -> None:
        ticks = [ValueTick(value=0, angle=225, confidence=1.0)]
        self.assertIsNone(interpolate_reading(90, ticks))


if __name__ == "__main__":
    unittest.main()
