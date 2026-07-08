from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gauge_label_lab.app import LabelStore, has_image_signature, is_supported_image


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01"
    b"\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00"
)


class ImageFilterTests(unittest.TestCase):
    def test_detects_common_image_signatures(self) -> None:
        self.assertTrue(has_image_signature(PNG_BYTES))
        self.assertTrue(has_image_signature(b"\xff\xd8\xff\xe0" + b"0" * 28))
        self.assertTrue(has_image_signature(b"GIF89a" + b"0" * 26))
        self.assertTrue(has_image_signature(b"RIFF0000WEBP" + b"0" * 20))
        self.assertFalse(has_image_signature(b"not an image"))

    def test_requires_supported_extension_and_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = root / "gauge.png"
            image.write_bytes(PNG_BYTES)
            renamed_text = root / "fake.png"
            renamed_text.write_text("not an image", encoding="utf-8")
            unsupported = root / "gauge.txt"
            unsupported.write_bytes(PNG_BYTES)

            self.assertTrue(is_supported_image(image))
            self.assertFalse(is_supported_image(renamed_text))
            self.assertFalse(is_supported_image(unsupported))


class LabelStoreTests(unittest.TestCase):
    def test_labels_current_image_and_writes_overlay_without_moving_original(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            labeled_dir = root / "labeled"
            csv_path = root / "labels.csv"
            input_dir.mkdir()
            (input_dir / "a.png").write_bytes(PNG_BYTES)
            (input_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

            store = LabelStore(input_dir, labeled_dir, csv_path, overlay_writer=fake_overlay_writer)
            store.ensure_ready()

            pending = store.pending_images()
            self.assertEqual([path.name for path in pending], ["a.png"])

            record = store.label_image("a.png", "5")
            self.assertEqual(record.original_filename, "a.png")
            self.assertEqual(record.labeled_filename, "a_overlay.png")
            self.assertEqual(record.value, "5")
            self.assertTrue((input_dir / "a.png").exists())
            self.assertTrue((labeled_dir / "a_overlay.png").exists())
            self.assertEqual(store.pending_images(), [])

            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["original_filename"], "a.png")
            self.assertEqual(rows[0]["labeled_filename"], "a_overlay.png")
            self.assertEqual(rows[0]["value"], "5")

    def test_renames_destination_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            labeled_dir = root / "labeled"
            csv_path = root / "labels.csv"
            input_dir.mkdir()
            labeled_dir.mkdir()
            (input_dir / "a.png").write_bytes(PNG_BYTES)
            (labeled_dir / "a_overlay.png").write_bytes(PNG_BYTES)

            store = LabelStore(input_dir, labeled_dir, csv_path, overlay_writer=fake_overlay_writer)
            store.ensure_ready()
            record = store.label_image("a.png", "4.5")

            self.assertEqual(record.labeled_filename, "a_overlay_0001.png")
            self.assertTrue((labeled_dir / "a_overlay_0001.png").exists())


def fake_overlay_writer(source: Path, work_dir: Path) -> Path:
    overlay = work_dir / f"{source.stem}_debug{source.suffix}"
    overlay.write_bytes(source.read_bytes())
    return overlay


if __name__ == "__main__":
    unittest.main()
