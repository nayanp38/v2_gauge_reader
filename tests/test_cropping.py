from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gauge_reader.cloud import CloudVisionAdapter, VisionExtraction
from gauge_reader.cropping import dial_crop_box, translate_numeric_label
from gauge_reader.models import DialGeometry, GaugeObservation, NumericLabel, Point
from gauge_reader.reader import GaugeReader


class FakeCloud(CloudVisionAdapter):
    def __init__(self, extraction: VisionExtraction):
        self.extraction = extraction
        self.paths: list[str] = []

    def extract(self, image_path: str | Path, context=None) -> VisionExtraction:
        self.paths.append(str(image_path))
        return self.extraction


class CroppingTests(unittest.TestCase):
    def test_dial_crop_box_adds_padding(self) -> None:
        dial = DialGeometry(Point(100, 100), 50, confidence=0.9)

        crop = dial_crop_box(dial, 300, 300, padding_fraction=0.10, min_padding_px=0)

        self.assertIsNotNone(crop)
        assert crop is not None
        self.assertEqual(crop.x, 45)
        self.assertEqual(crop.y, 45)
        self.assertEqual(crop.width, 110)
        self.assertEqual(crop.height, 110)

    def test_dial_crop_box_clips_to_image_bounds(self) -> None:
        dial = DialGeometry(Point(20, 20), 50, confidence=0.9)

        crop = dial_crop_box(dial, 100, 100, padding_fraction=0.0, min_padding_px=0)

        self.assertIsNotNone(crop)
        assert crop is not None
        self.assertEqual(crop.x, 0)
        self.assertEqual(crop.y, 0)
        self.assertEqual(crop.width, 70)
        self.assertEqual(crop.height, 70)

    def test_translate_numeric_label_moves_bbox_and_center(self) -> None:
        label = NumericLabel(value=5, text="5", center=Point(10, 20), confidence=0.8)

        translated = translate_numeric_label(label, Point(100, 200))

        self.assertEqual(translated.center, Point(110, 220))
        self.assertEqual(translated.value, 5)
        self.assertEqual(translated.confidence, 0.8)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "cv2 not installed")
    def test_cloud_crop_coordinates_are_translated_back_to_full_image(self) -> None:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "image.png"
            cv2.imwrite(str(image_path), np.zeros((200, 200, 3), dtype=np.uint8))
            dial = DialGeometry(Point(100, 100), 50, confidence=0.9)
            observation = GaugeObservation(dial=dial, needle=None)
            cloud = FakeCloud(
                VisionExtraction(
                    labels=[
                        NumericLabel(value=5, text="5", center=Point(10, 20), confidence=0.9, source="cloud"),
                    ]
                )
            )
            reader = GaugeReader(cloud_adapter=cloud)

            extraction = reader._extract_cloud(image_path, observation)

        crop = dial_crop_box(dial, 200, 200)
        self.assertIsNotNone(crop)
        assert crop is not None
        self.assertEqual(len(cloud.paths), 1)
        self.assertNotEqual(cloud.paths[0], str(image_path))
        self.assertEqual(extraction.labels[0].center, Point(crop.x + 10, crop.y + 20))


if __name__ == "__main__":
    unittest.main()
