from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gauge_reader.models import DialGeometry, GaugeObservation, NeedleDetection, Point
from gauge_reader.reader import GaugeReader


class FakePipeline:
    def __init__(self, observation: GaugeObservation):
        self.observation = observation

    def process(self, image_path: str) -> GaugeObservation:
        return self.observation


class SyntheticScenarioTests(unittest.TestCase):
    def test_nonzero_sweep_with_midpoint_calibration(self) -> None:
        observation = GaugeObservation(
            dial=DialGeometry(Point(200, 200), 140, confidence=0.9),
            needle=NeedleDetection(angle=45, confidence=0.92),
        )
        reader = GaugeReader(pipeline=FakePipeline(observation), use_cloud=False)
        result = reader.read(
            "synthetic.jpg",
            calibration={
                "unit": "bar",
                "ticks": [
                    {"value": 0, "angle": 225},
                    {"value": 50, "angle": 90},
                    {"value": 100, "angle": 315},
                ],
            },
        )
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.reading or 0, 66.6666666667)
        self.assertEqual(result.unit, "bar")

    def test_endpoint_only_long_arc_is_reported_but_lower_confidence(self) -> None:
        observation = GaugeObservation(
            dial=DialGeometry(Point(200, 200), 140, confidence=0.5),
            needle=NeedleDetection(angle=90, confidence=0.55),
        )
        reader = GaugeReader(pipeline=FakePipeline(observation), use_cloud=False, confidence_threshold=0.8)
        result = reader.read(
            "synthetic.jpg",
            calibration={
                "ticks": [
                    {"value": 0, "angle": 225},
                    {"value": 100, "angle": 315},
                ],
            },
        )
        self.assertEqual(result.status, "low_confidence")
        self.assertAlmostEqual(result.reading or 0, 50.0)


if __name__ == "__main__":
    unittest.main()
