from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gauge_reader.cloud import CloudVisionAdapter, VisionExtraction, extraction_from_mapping
from gauge_reader.geometry import point_on_dial
from gauge_reader.models import (
    BBox,
    DialGeometry,
    GaugeObservation,
    NeedleDetection,
    NumericLabel,
    Point,
    TickDetection,
)
from gauge_reader.reader import GaugeReader, associate_labels_to_ticks


class FakePipeline:
    def __init__(self, observation: GaugeObservation):
        self.observation = observation

    def process(self, image_path: str) -> GaugeObservation:
        return self.observation


class FakeCloud(CloudVisionAdapter):
    def __init__(self, extraction: VisionExtraction):
        self.extraction = extraction
        self.calls = 0

    def extract(self, image_path: str, context=None) -> VisionExtraction:
        self.calls += 1
        return self.extraction


class ReaderTests(unittest.TestCase):
    def test_manual_calibration_overrides_missing_ocr(self) -> None:
        observation = GaugeObservation(
            dial=DialGeometry(Point(100, 100), 80, confidence=0.9),
            needle=NeedleDetection(angle=90, confidence=0.95),
        )
        reader = GaugeReader(pipeline=FakePipeline(observation), use_cloud=False)
        result = reader.read(
            "unused.jpg",
            calibration={
                "unit": "psi",
                "ticks": [
                    {"value": 0, "angle": 225},
                    {"value": 100, "angle": 315},
                ],
            },
        )
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.reading or 0, 50.0)
        self.assertEqual(result.unit, "psi")
        self.assertEqual(result.lower_tick.value, 0)
        self.assertEqual(result.upper_tick.value, 100)

    def test_associates_numeric_labels_to_nearby_ticks(self) -> None:
        dial = DialGeometry(Point(100, 100), 80, confidence=0.9)
        labels = [
            NumericLabel(value=40, text="40", bbox=BBox(92, 18, 16, 12), confidence=0.9),
        ]
        ticks = [TickDetection(angle=90, confidence=0.8)]
        value_ticks = associate_labels_to_ticks(labels, ticks, dial)
        self.assertEqual(len(value_ticks), 1)
        self.assertEqual(value_ticks[0].value, 40)
        self.assertAlmostEqual(value_ticks[0].angle, 90)

    def test_associates_labels_in_elliptical_dial_space(self) -> None:
        dial = DialGeometry(
            Point(100, 100),
            60,
            confidence=0.9,
            major_radius=90,
            minor_radius=45,
            rotation_deg=25,
            source="test_ellipse",
        )
        labels = [
            NumericLabel(value=20, text="20", center=point_on_dial(dial, 120, 0.72), confidence=0.9),
        ]
        ticks = [TickDetection(angle=120, confidence=0.8)]
        value_ticks = associate_labels_to_ticks(labels, ticks, dial)
        self.assertEqual(len(value_ticks), 1)
        self.assertEqual(value_ticks[0].value, 20)
        self.assertAlmostEqual(value_ticks[0].angle, 120)

    def test_cloud_fallback_can_supply_labels(self) -> None:
        dial = DialGeometry(Point(100, 100), 80, confidence=0.9)
        observation = GaugeObservation(
            dial=dial,
            needle=NeedleDetection(angle=90, confidence=0.95),
        )
        cloud = FakeCloud(
            VisionExtraction(
                labels=[
                    NumericLabel(value=0, text="0", center=Point(43.4, 156.6), confidence=0.9, source="cloud"),
                    NumericLabel(value=50, text="50", center=Point(100, 20), confidence=0.9, source="cloud"),
                    NumericLabel(value=100, text="100", center=Point(156.6, 156.6), confidence=0.9, source="cloud"),
                ]
            )
        )
        reader = GaugeReader(pipeline=FakePipeline(observation), cloud_adapter=cloud, use_cloud=True)
        result = reader.read("unused.jpg")
        self.assertEqual(cloud.calls, 1)
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.reading or 0, 50.0, places=1)
        self.assertEqual(result.reason, "ok_cloud_fallback")

    def test_cloud_fallback_can_supply_missing_needle_with_calibration(self) -> None:
        observation = GaugeObservation(
            dial=DialGeometry(Point(100, 100), 80, confidence=0.9),
            needle=None,
        )
        cloud = FakeCloud(
            VisionExtraction(
                needle=NeedleDetection(
                    angle=90,
                    confidence=0.9,
                    start=Point(100, 100),
                    end=Point(100, 20),
                    source="cloud",
                )
            )
        )
        reader = GaugeReader(pipeline=FakePipeline(observation), cloud_adapter=cloud, use_cloud=True)
        result = reader.read(
            "unused.jpg",
            calibration={
                "ticks": [
                    {"value": 0, "angle": 225},
                    {"value": 100, "angle": 315},
                ],
            },
        )
        self.assertEqual(cloud.calls, 1)
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.reading or 0, 50.0)
        self.assertEqual(result.reason, "ok_cloud_fallback")

    def test_cloud_mapping_parses_dial_and_needle(self) -> None:
        extraction = extraction_from_mapping(
            {
                "dial": {
                    "center": [100, 100],
                    "radius": 80,
                    "bbox": [20, 20, 160, 160],
                    "confidence": 0.8,
                },
                "needle": {
                    "base": [100, 100],
                    "tip": [100, 20],
                    "confidence": 0.9,
                },
            },
            source="test",
        )
        self.assertIsNotNone(extraction.dial)
        self.assertIsNotNone(extraction.needle)
        assert extraction.dial is not None
        assert extraction.needle is not None
        self.assertAlmostEqual(extraction.dial.radius, 80)
        self.assertAlmostEqual(extraction.needle.angle, 90)

    def test_failed_when_needle_missing(self) -> None:
        observation = GaugeObservation(dial=DialGeometry(Point(100, 100), 80, confidence=0.9), needle=None)
        reader = GaugeReader(pipeline=FakePipeline(observation), use_cloud=False)
        result = reader.read("unused.jpg")
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "needle_not_found")


if __name__ == "__main__":
    unittest.main()
