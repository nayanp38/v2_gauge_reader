from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gauge_reader.evaluate import (
    EvaluationPaths,
    discover_samples,
    eval_pipeline_name,
    evaluate_dataset,
    find_synth_dial_value,
    gauge_min_max_from_annotations,
    resolve_eval_pipeline,
    summarize_rows,
)
from gauge_reader.models import GaugeReading


class FakeReader:
    def read(self, image_path: Path, calibration=None, debug_dir: Path | None = None) -> GaugeReading:
        debug = {}
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            overlay = debug_dir / "overlay.jpg"
            overlay.write_text("overlay", encoding="utf-8")
            debug["overlay"] = str(overlay)
        return GaugeReading(
            reading=4.8,
            confidence=0.9,
            status="ok",
            reason="ok",
            needle_angle=123.0,
            debug=debug,
        )


class EvaluateTests(unittest.TestCase):
    def test_finds_synth_dial_value_in_annotation_list(self) -> None:
        payload = [{"category_name": "dial", "synth_dial_value": "5.25"}]
        self.assertEqual(find_synth_dial_value(payload), 5.25)

    def test_finds_gauge_min_max_from_scale_labels(self) -> None:
        payload = [
            {"category_name": "scale-label", "synth_value": "0"},
            {"category_name": "scale-label", "synth_value": 10},
            {"category_name": "dial", "synth_dial_value": 5},
        ]
        self.assertEqual(gauge_min_max_from_annotations(payload), (0.0, 10.0))

    def test_discovers_same_stem_samples_and_oracle_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.png").write_bytes(b"not inspected")
            (root / "a.json").write_text(
                json.dumps(
                    [
                        {"category_name": "dial", "synth_dial_value": 5},
                        {"category_name": "scale-label", "synth_value": 0, "bbox": [10, 20, 8, 12]},
                        {"category_name": "scale-label", "synth_value": 10, "bbox": [100, 20, 8, 12]},
                    ]
                ),
                encoding="utf-8",
            )
            (root / "b.json").write_text(json.dumps([{"synth_dial_value": 3}]), encoding="utf-8")

            samples, discovery = discover_samples(root, use_oracle_calibration=True)

        self.assertEqual(len(samples), 1)
        self.assertEqual(discovery["json_files"], 2)
        self.assertEqual(discovery["missing_image"], 1)
        self.assertEqual(discovery["samples_with_gauge_range"], 1)
        self.assertEqual(samples[0].target, 5)
        self.assertEqual(samples[0].gauge_min, 0)
        self.assertEqual(samples[0].gauge_max, 10)
        self.assertEqual(samples[0].gauge_range, 10)
        self.assertIsNotNone(samples[0].calibration)
        assert samples[0].calibration is not None
        self.assertEqual(len(samples[0].calibration["ticks"]), 2)
        self.assertEqual(samples[0].calibration["ticks"][0]["point"], [14.0, 26.0])

    def test_evaluate_dataset_writes_results_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data = root / "data"
            out = root / "out"
            data.mkdir()
            (data / "a.png").write_bytes(b"not inspected")
            (data / "a.json").write_text(
                json.dumps(
                    [
                        {"category_name": "dial", "synth_dial_value": 5},
                        {"category_name": "scale-label", "synth_value": 0, "bbox": [10, 20, 8, 12]},
                        {"category_name": "scale-label", "synth_value": 10, "bbox": [100, 20, 8, 12]},
                    ]
                ),
                encoding="utf-8",
            )
            paths = EvaluationPaths(
                output_dir=out,
                results_csv=out / "per_sample_results.csv",
                summary_json=out / "summary.json",
            )

            summary = evaluate_dataset(
                data,
                paths,
                use_cloud=False,
                reader_factory=FakeReader,
            )

            with paths.results_csv.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            source_still_exists = (data / "a.png").exists()
            overlay_path = rows[0]["debug_overlay_path"]
            overlay_exists = Path(overlay_path).exists()

        self.assertEqual(summary["counts"]["evaluated"], 1)
        self.assertEqual(summary["counts"]["predicted"], 1)
        self.assertAlmostEqual(summary["error_metrics"]["mae"], 0.2)
        self.assertAlmostEqual(summary["error_metrics"]["range_normalized_mae"], 0.02)
        self.assertAlmostEqual(summary["error_metrics"]["mean_percent_range_error"], 2.0)
        self.assertEqual(summary["counts"]["range_normalized"], 1)
        self.assertEqual(rows[0]["sample_id"], "a")
        self.assertEqual(float(rows[0]["prediction"]), 4.8)
        self.assertEqual(float(rows[0]["gauge_range"]), 10.0)
        self.assertAlmostEqual(float(rows[0]["abs_range_normalized_error"]), 0.02)
        self.assertAlmostEqual(float(rows[0]["percent_range_error"]), 2.0)
        self.assertTrue(source_still_exists)
        self.assertTrue(overlay_exists)
        self.assertIn("labeled/a/overlay.jpg", overlay_path)
        self.assertEqual(summary["config"]["pipeline"], "local")

    def test_evaluate_dataset_reports_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data = root / "data"
            out = root / "out"
            data.mkdir()
            for sample_id, value in [("a", 5), ("b", 6)]:
                (data / f"{sample_id}.png").write_bytes(b"not inspected")
                (data / f"{sample_id}.json").write_text(
                    json.dumps([{"category_name": "dial", "synth_dial_value": value}]),
                    encoding="utf-8",
                )
            paths = EvaluationPaths(
                output_dir=out,
                results_csv=out / "per_sample_results.csv",
                summary_json=out / "summary.json",
            )
            progress = io.StringIO()

            evaluate_dataset(
                data,
                paths,
                use_cloud=False,
                reader_factory=FakeReader,
                show_progress=True,
                progress_stream=progress,
            )

        progress_lines = progress.getvalue().splitlines()
        self.assertEqual(progress_lines[0], "gauge-eval progress: 0/2 completed (0.0%)")
        self.assertIn("gauge-eval progress: 1/2 completed (50.0%) sample=a status=ok", progress_lines)
        self.assertIn("gauge-eval progress: 2/2 completed (100.0%) sample=b status=ok", progress_lines)

    def test_cloud_ocr_pipeline_forces_cloud_ocr(self) -> None:
        use_cloud, force_cloud_ocr = resolve_eval_pipeline(
            "cloud-ocr",
            no_cloud=False,
            force_cloud_ocr=False,
        )

        self.assertTrue(use_cloud)
        self.assertTrue(force_cloud_ocr)
        self.assertEqual(eval_pipeline_name(use_cloud, force_cloud_ocr), "cloud-ocr")

    def test_local_pipeline_disables_cloud(self) -> None:
        use_cloud, force_cloud_ocr = resolve_eval_pipeline(
            "local",
            no_cloud=False,
            force_cloud_ocr=False,
        )

        self.assertFalse(use_cloud)
        self.assertFalse(force_cloud_ocr)
        self.assertEqual(eval_pipeline_name(use_cloud, force_cloud_ocr), "local")

    def test_summarize_rows_tracks_failed_coverage(self) -> None:
        summary = summarize_rows(
            [
                {
                    "status": "failed",
                    "reason": "needle_not_found",
                    "prediction": "",
                    "latency_ms": 10,
                },
                {
                    "status": "ok",
                    "reason": "ok",
                    "prediction": 2,
                    "error": -0.5,
                    "abs_error": 0.5,
                    "squared_error": 0.25,
                    "range_normalized_error": -0.05,
                    "abs_range_normalized_error": 0.05,
                    "squared_range_normalized_error": 0.0025,
                    "percent_range_error": 5.0,
                    "latency_ms": 30,
                },
            ],
            discovery={"samples": 2},
            config={"test": True},
        )
        self.assertEqual(summary["counts"]["coverage"], 0.5)
        self.assertEqual(summary["counts"]["range_normalized"], 1)
        self.assertEqual(summary["counts"]["ok_coverage"], 0.5)
        self.assertEqual(summary["counts"]["status"]["failed"], 1)
        self.assertEqual(summary["error_metrics"]["within_abs_0_5"], 1.0)
        self.assertEqual(summary["error_metrics"]["within_5_percent_range"], 1.0)
        self.assertEqual(summary["error_metrics"]["mean_percent_range_error"], 5.0)
        self.assertEqual(summary["ok_error_metrics"]["within_abs_0_5"], 1.0)
        self.assertEqual(summary["ok_error_metrics"]["range_normalized_mae"], 0.05)
        self.assertEqual(summary["latency_ms"]["median"], 20.0)


if __name__ == "__main__":
    unittest.main()
