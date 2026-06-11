from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gauge_reader.gemini_vlm_adapter import (
    DEFAULT_MODEL,
    normalize_payload,
    parse_gemini_response,
    parse_json_response,
)


class FakeObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class GeminiVLMAdapterTests(unittest.TestCase):
    def test_default_model_is_robotics_er_16(self) -> None:
        self.assertEqual(DEFAULT_MODEL, "gemini-robotics-er-1.6-preview")

    def test_parse_plain_json_response(self) -> None:
        parsed = parse_json_response('{"unit": "psi", "labels": []}')
        self.assertEqual(parsed["unit"], "psi")

    def test_parse_markdown_json_response(self) -> None:
        parsed = parse_json_response(
            '```json\n{"unit": null, "labels": [{"text": "40", "value": 40, "bbox": [1, 2, 3, 4], "confidence": 0.8}]}\n```'
        )
        self.assertEqual(parsed["labels"][0]["value"], 40)

    def test_parse_markdown_array_response(self) -> None:
        parsed = parse_json_response(
            '```json\n[{"box_2d": [1, 2, 3, 4], "label": "10"}, {"box_2d": [5, 6, 7, 8], "label": "20"}]\n```'
        )
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[1]["label"], "20")

    def test_parse_gemini_response_skips_tool_parts_and_converts_er_boxes(self) -> None:
        response = FakeObject(
            candidates=[
                FakeObject(
                    content=FakeObject(
                        parts=[
                            FakeObject(executable_code="print('ignored')"),
                            FakeObject(code_execution_result="ignored"),
                            FakeObject(
                                text='```json\n{"unit": null, "labels": [{"text": "40", "value": 40, "bbox": [100, 200, 300, 500], "confidence": 0.8}]}\n```'
                            ),
                        ]
                    )
                )
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "test.png"
            image_path.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR"
                b"\x00\x00\x03\xe8"
                b"\x00\x00\x01\xf4"
                b"\x08\x02\x00\x00\x00"
            )
            parsed = parse_gemini_response(response, image_path)

        self.assertEqual(parsed["labels"][0]["bbox"], [200.0, 50.0, 300.0, 100.0])

    def test_parse_docs_style_array_from_response_text(self) -> None:
        response = FakeObject(
            text='[{"box_2d": [100, 200, 300, 500], "label": "40"}]'
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "test.png"
            image_path.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR"
                b"\x00\x00\x03\xe8"
                b"\x00\x00\x01\xf4"
                b"\x08\x02\x00\x00\x00"
            )
            parsed = parse_gemini_response(response, image_path)

        self.assertIsNone(parsed["unit"])
        self.assertEqual(parsed["labels"][0]["text"], "40")
        self.assertEqual(parsed["labels"][0]["value"], 40.0)
        self.assertEqual(parsed["labels"][0]["bbox"], [200.0, 50.0, 300.0, 100.0])

    def test_parse_docs_style_object_with_dial_and_needle(self) -> None:
        response = FakeObject(
            text=(
                '{"dial": {"box_2d": [100, 100, 900, 900], "center_point": [500, 500]}, '
                '"needle": {"base_point": [500, 500], "tip_point": [200, 500]}, '
                '"labels": [{"box_2d": [100, 200, 300, 500], "label": "40"}]}'
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "test.png"
            image_path.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR"
                b"\x00\x00\x03\xe8"
                b"\x00\x00\x01\xf4"
                b"\x08\x02\x00\x00\x00"
            )
            parsed = parse_gemini_response(response, image_path)
            normalized = normalize_payload(parsed)

        self.assertEqual(normalized["dial"]["center"], [500.0, 250.0])
        self.assertEqual(normalized["dial"]["radius"], 300.0)
        self.assertEqual(normalized["needle"]["base"], [500.0, 250.0])
        self.assertEqual(normalized["needle"]["tip"], [500.0, 100.0])
        self.assertEqual(normalized["labels"][0]["bbox"], [200.0, 50.0, 300.0, 100.0])

    def test_normalize_payload_filters_bad_labels(self) -> None:
        normalized = normalize_payload(
            {
                "unit": "bar",
                "labels": [
                    {"text": "10", "value": "10", "bbox": [1, 2, 3, 4], "confidence": 2},
                    {"text": "bad", "value": "bad", "bbox": [1, 2, 3, 4], "confidence": 0.8},
                    {"text": "bad", "value": 20, "bbox": [1, 2, 3], "confidence": 0.8},
                ],
            }
        )
        self.assertEqual(normalized["unit"], "bar")
        self.assertEqual(len(normalized["labels"]), 1)
        self.assertEqual(normalized["labels"][0]["confidence"], 1.0)
        self.assertEqual(normalized["ticks"], [])


if __name__ == "__main__":
    unittest.main()
