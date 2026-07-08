from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gemini-robotics-er-1.6-preview"

PROMPT = """Return ONLY JSON for an analog gauge image.
Use Gemini Robotics-ER normalized coordinates:
- points are [y, x] on a 0-1000 scale
- boxes are [ymin, xmin, ymax, xmax] on a 0-1000 scale

Find:
1. the dial/circle bounding box and center point
2. the needle base point near the pivot and needle tip point
3. every visible numeric dial label printed on the gauge face

Ignore logos, serial numbers, units, annotations, reflections, and camera UI text.
Do not return masks, prose, explanations, or markdown fences.
For every numeric label, return a box_2d. If you truly cannot provide a box,
return center_point for that label instead. Do not return label-only entries.

Return this shape:
{
  "dial": {"box_2d": [ymin, xmin, ymax, xmax], "center_point": [y, x]},
  "needle": {"base_point": [y, x], "tip_point": [y, x]},
  "labels": [{"box_2d": [ymin, xmin, ymax, xmax], "label": "40"}]
}

If a field is not visible, use null for dial or needle, and [] for labels.
"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "unit": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "value": {"type": "number"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "center": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "required": ["text", "value", "confidence"],
            },
        },
        "dial": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "center": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "radius": {"type": "number"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": ["center", "radius", "confidence"],
                },
                {"type": "null"},
            ]
        },
        "needle": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "base": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "tip": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": ["tip", "confidence"],
                },
                {"type": "null"},
            ]
        },
    },
    "required": ["unit", "labels"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gemini VLM adapter for gauge-reader cloud fallback."
    )
    parser.add_argument("image", help="Path to a gauge image.")
    parser.add_argument(
        "--model",
        default=os.environ.get("GAUGE_READER_GEMINI_MODEL", DEFAULT_MODEL),
        help="Gemini model to use. Defaults to GAUGE_READER_GEMINI_MODEL or gemini-robotics-er-1.6-preview.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("GAUGE_READER_GEMINI_TEMPERATURE", "0")),
        help="Generation temperature.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=int(os.environ.get("GAUGE_READER_GEMINI_MAX_OUTPUT_TOKENS", "2048")),
        help="Maximum output tokens.",
    )
    parser.add_argument(
        "--diagnostics",
        default=os.environ.get("GAUGE_READER_GEMINI_DIAGNOSTICS"),
        help="Optional path to write raw/parsed response diagnostics for parser debugging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = extract_gauge_labels(
            image_path=Path(args.image),
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            diagnostics_path=Path(args.diagnostics) if args.diagnostics else None,
        )
    except Exception as exc:
        print(f"gemini_vlm_adapter: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(normalize_payload(payload), sort_keys=True))
    return 0


def extract_gauge_labels(
    *,
    image_path: Path,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
    diagnostics_path: Path | None = None,
) -> dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise RuntimeError("install cloud dependencies with: python3 -m pip install -e '.[cloud]'") from exc

    mime_type = _guess_mime_type(image_path)
    image_bytes = image_path.read_bytes()

    client = _build_client(genai)
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            PROMPT,
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    return parse_gemini_response(response, image_path, diagnostics_path=diagnostics_path)


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    labels: list[dict[str, Any]] = []
    for raw in payload.get("labels", []):
        label = _normalize_label(raw)
        if label is not None:
            labels.append(label)
    unit = payload.get("unit")
    return {
        "unit": str(unit).strip() if unit not in {None, ""} else None,
        "labels": labels,
        "dial": _normalize_dial(payload.get("dial")),
        "needle": _normalize_needle(payload.get("needle")),
        "ticks": [],
    }


def parse_json_response(text: str) -> Any:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("empty model response")
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = _parse_json_from_markdown_or_text(cleaned)
    return parsed


def parse_gemini_response(
    response: Any,
    image_path: Path,
    diagnostics_path: Path | None = None,
) -> dict[str, Any]:
    raw_text = _text_from_response(response)
    parsed: Any = None
    payload: dict[str, Any] | None = None
    try:
        parsed = parse_json_response(raw_text)
        payload = _payload_from_model_json(parsed, _read_image_size(image_path))
        _validate_against_response_schema(payload)
        _write_diagnostics(diagnostics_path, raw_text=raw_text, parsed=parsed, payload=payload)
        return payload
    except Exception as exc:
        _write_diagnostics(diagnostics_path, raw_text=raw_text, parsed=parsed, payload=payload, error=exc)
        print(f"gemini_vlm_adapter raw response: {raw_text}", file=sys.stderr)
        raise


def _text_from_response(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        chunks: list[str] = []
        for part in parts:
            if _has_tool_part(part):
                continue
            text = getattr(part, "text", None)
            if text:
                chunks.append(str(text))
        joined = "".join(chunks).strip()
        if joined:
            return joined

    response_text = getattr(response, "text", None)
    if response_text:
        return str(response_text).strip()
    raise ValueError("model response did not include text")


def _has_tool_part(part: Any) -> bool:
    return any(
        getattr(part, name, None) is not None
        for name in (
            "executable_code",
            "executableCode",
            "code_execution_result",
            "codeExecutionResult",
        )
    )


def _parse_json_from_markdown_or_text(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        fenced_json = "\n".join(lines).strip()
        if fenced_json:
            return json.loads(fenced_json)

    starts = [index for index in (cleaned.find("{"), cleaned.find("[")) if index >= 0]
    start = min(starts) if starts else -1
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if start >= 0 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError("model response did not contain JSON")


def _write_diagnostics(
    diagnostics_path: Path | None,
    *,
    raw_text: str,
    parsed: Any,
    payload: dict[str, Any] | None,
    error: Exception | None = None,
) -> None:
    if diagnostics_path is None:
        return
    report = _diagnostics_report(raw_text=raw_text, parsed=parsed, payload=payload, error=error)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def _diagnostics_report(
    *,
    raw_text: str,
    parsed: Any,
    payload: dict[str, Any] | None,
    error: Exception | None,
) -> dict[str, Any]:
    raw_labels = _raw_label_items(parsed)
    raw_label_stats = [_raw_label_stat(item) for item in raw_labels]
    converted_labels = payload.get("labels", []) if isinstance(payload, dict) else []
    report: dict[str, Any] = {
        "raw_text": raw_text,
        "parsed": parsed,
        "payload": payload,
        "counts": {
            "raw_labels": len(raw_labels),
            "raw_labels_with_numeric_text": sum(1 for item in raw_label_stats if item["has_numeric_text"]),
            "raw_labels_with_box": sum(1 for item in raw_label_stats if item["has_box"]),
            "raw_labels_with_center": sum(1 for item in raw_label_stats if item["has_center"]),
            "converted_labels": len(converted_labels),
            "converted_labels_with_bbox": sum(1 for item in converted_labels if _is_bbox(item.get("bbox"))),
            "converted_labels_with_center": sum(1 for item in converted_labels if _is_point(item.get("center"))),
            "dropped_raw_labels": max(0, len(raw_labels) - len(converted_labels)),
        },
        "raw_label_stats": raw_label_stats,
    }
    if error is not None:
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def _raw_label_items(parsed: Any) -> list[Any]:
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        labels = parsed.get("labels")
        if isinstance(labels, list):
            return labels
        if "box_2d" in parsed or "bbox" in parsed:
            return [parsed]
    return []


def _raw_label_stat(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {
            "kind": type(item).__name__,
            "has_numeric_text": _numeric_value_from_text(str(item)) is not None,
            "has_box": False,
            "has_center": False,
            "keys": [],
        }
    text = str(_first_present(item, ("label", "text", "number", "value", "name")) or "")
    return {
        "kind": "object",
        "text": text,
        "has_numeric_text": _numeric_value_from_text(text) is not None,
        "has_box": _is_bbox(
            _first_present(item, ("box_2d", "bbox", "bounding_box", "boundingBox", "box", "box2d"))
        ),
        "has_center": _is_point(
            _first_present(item, ("center_point", "center", "point", "location"))
        ),
        "keys": sorted(str(key) for key in item.keys()),
    }


def _validate_against_response_schema(payload: dict[str, Any]) -> None:
    _validate_object(payload, RESPONSE_SCHEMA, path="$")


def _validate_object(value: Any, schema: dict[str, Any], path: str) -> None:
    if schema.get("anyOf"):
        errors: list[str] = []
        for option in schema["anyOf"]:
            try:
                _validate_object(value, option, path)
                return
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError(f"{path} did not match any allowed schema: {'; '.join(errors)}")

    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        for required in schema.get("required", []):
            if required not in value:
                raise ValueError(f"{path}.{required} is required")
        properties = schema.get("properties", {})
        for key, child in properties.items():
            if key in value:
                _validate_object(value[key], child, f"{path}.{key}")
        return

    if schema_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array")
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(value) < min_items:
            raise ValueError(f"{path} must contain at least {min_items} items")
        if max_items is not None and len(value) > max_items:
            raise ValueError(f"{path} must contain at most {max_items} items")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                _validate_object(item, item_schema, f"{path}[{index}]")
        return

    if schema_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string")
        return

    if schema_type == "number":
        if _coerce_float(value) is None:
            raise ValueError(f"{path} must be a number")
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        number = float(value)
        if minimum is not None and number < minimum:
            raise ValueError(f"{path} must be >= {minimum}")
        if maximum is not None and number > maximum:
            raise ValueError(f"{path} must be <= {maximum}")
        return

    if schema_type == "null":
        if value is not None:
            raise ValueError(f"{path} must be null")
        return


def _payload_from_model_json(parsed: Any, image_size: tuple[int, int]) -> dict[str, Any]:
    if isinstance(parsed, list):
        return _payload_from_er_box_array(parsed, image_size)
    if isinstance(parsed, dict):
        if "labels" in parsed or "dial" in parsed or "needle" in parsed:
            return _payload_from_er_object(parsed, image_size)
        if "box_2d" in parsed or "bbox" in parsed:
            return _payload_from_er_box_array([parsed], image_size)
    raise ValueError("model response must be a JSON array of boxes or an object with labels")


def _payload_from_er_box_array(items: list[Any], image_size: tuple[int, int]) -> dict[str, Any]:
    labels: list[dict[str, Any]] = []
    for item in items:
        label = _label_from_er_item(item, image_size)
        if label is not None:
            labels.append(label)
    return {"unit": None, "labels": labels}


def _label_from_er_item(item: Any, image_size: tuple[int, int]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    text = str(_first_present(item, ("label", "text", "number", "value", "name")) or "").strip()
    value = _coerce_float(_first_present(item, ("value", "number")))
    if value is None:
        value = _numeric_value_from_text(text)
    if value is None:
        return None

    box = _first_present(
        item,
        (
            "box_2d",
            "bbox",
            "bounding_box",
            "boundingBox",
            "box",
            "box2d",
        ),
    )
    center = _first_present(
        item,
        (
            "center_point",
            "center",
            "point",
            "location",
        ),
    )
    if not _is_bbox(box) and not _is_point(center):
        return None

    label: dict[str, Any] = {
        "text": text or str(value),
        "value": value,
        "confidence": _clamp01(
            _coerce_float(item.get("confidence"))
            if _coerce_float(item.get("confidence")) is not None
            else 0.75
        ),
    }
    if _is_bbox(box):
        label["bbox"] = _er_box_to_pixel_bbox([float(part) for part in box], image_size)
    if _is_point(center):
        label["center"] = _er_point_to_pixel_point(center, image_size)
    return label


def _payload_from_er_object(payload: dict[str, Any], image_size: tuple[int, int]) -> dict[str, Any]:
    labels_payload = _payload_from_er_box_array(payload.get("labels", []), image_size)
    converted: dict[str, Any] = {
        "unit": payload.get("unit"),
        "labels": labels_payload["labels"],
        "dial": _dial_from_er_object(payload.get("dial"), image_size),
        "needle": _needle_from_er_object(payload.get("needle"), image_size),
    }
    return converted


def _convert_payload_bboxes_to_pixel_bboxes(payload: dict[str, Any], image_size: tuple[int, int]) -> dict[str, Any]:
    width, height = image_size
    converted = dict(payload)
    labels: list[dict[str, Any]] = []
    for raw in payload.get("labels", []):
        label = dict(raw)
        box = label.get("box_2d", label.get("bbox"))
        if not _is_bbox(box):
            continue
        label["bbox"] = _er_box_to_pixel_bbox([float(item) for item in box], (width, height))
        label.pop("box_2d", None)
        labels.append(label)
    converted["labels"] = labels
    return converted


def _dial_from_er_object(raw: Any, image_size: tuple[int, int]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    box = raw.get("box_2d", raw.get("bbox"))
    center = raw.get("center_point", raw.get("center"))
    pixel_bbox = _er_box_to_pixel_bbox([float(item) for item in box], image_size) if _is_bbox(box) else None
    pixel_center = _er_point_to_pixel_point(center, image_size) if _is_point(center) else None
    if pixel_center is None and pixel_bbox is not None:
        pixel_center = [
            pixel_bbox[0] + pixel_bbox[2] / 2.0,
            pixel_bbox[1] + pixel_bbox[3] / 2.0,
        ]
    if pixel_center is None:
        return None
    radius = _coerce_float(raw.get("radius"))
    if radius is None and pixel_bbox is not None:
        radius = (pixel_bbox[2] + pixel_bbox[3]) / 4.0
    if radius is None or radius <= 0:
        return None
    result: dict[str, Any] = {
        "center": pixel_center,
        "radius": radius,
        "confidence": _clamp01(_coerce_float(raw.get("confidence")) if _coerce_float(raw.get("confidence")) is not None else 0.80),
    }
    if pixel_bbox is not None:
        result["bbox"] = pixel_bbox
    return result


def _needle_from_er_object(raw: Any, image_size: tuple[int, int]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    base = raw.get("base_point", raw.get("base", raw.get("start_point", raw.get("start"))))
    tip = raw.get("tip_point", raw.get("tip", raw.get("end_point", raw.get("end"))))
    pixel_tip = _er_point_to_pixel_point(tip, image_size) if _is_point(tip) else None
    if pixel_tip is None:
        return None
    result: dict[str, Any] = {
        "tip": pixel_tip,
        "confidence": _clamp01(_coerce_float(raw.get("confidence")) if _coerce_float(raw.get("confidence")) is not None else 0.80),
    }
    if _is_point(base):
        result["base"] = _er_point_to_pixel_point(base, image_size)
    return result


def _er_box_to_pixel_bbox(box: list[float], image_size: tuple[int, int]) -> list[float]:
    width, height = image_size
    ymin, xmin, ymax, xmax = box
    ymin, ymax = sorted((_clamp(ymin, 0.0, 1000.0), _clamp(ymax, 0.0, 1000.0)))
    xmin, xmax = sorted((_clamp(xmin, 0.0, 1000.0), _clamp(xmax, 0.0, 1000.0)))
    return [
        xmin / 1000.0 * width,
        ymin / 1000.0 * height,
        (xmax - xmin) / 1000.0 * width,
        (ymax - ymin) / 1000.0 * height,
    ]


def _er_point_to_pixel_point(point: Any, image_size: tuple[int, int]) -> list[float]:
    width, height = image_size
    y, x = [float(item) for item in point]
    return [
        _clamp(x, 0.0, 1000.0) / 1000.0 * width,
        _clamp(y, 0.0, 1000.0) / 1000.0 * height,
    ]


def _read_image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return image.size
    except Exception:
        pass

    with image_path.open("rb") as handle:
        header = handle.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            width, height = struct.unpack(">II", header[16:24])
            return int(width), int(height)
        if header.startswith(b"\xff\xd8"):
            return _read_jpeg_size(image_path)
    raise ValueError(f"could not determine image size for coordinate conversion: {image_path}")


def _read_jpeg_size(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as handle:
        handle.read(2)
        while True:
            marker_start = handle.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                handle.read(3)
                height, width = struct.unpack(">HH", handle.read(4))
                return int(width), int(height)
            segment_length_data = handle.read(2)
            if len(segment_length_data) != 2:
                break
            segment_length = struct.unpack(">H", segment_length_data)[0]
            handle.seek(max(0, segment_length - 2), 1)
    raise ValueError(f"could not determine JPEG size: {image_path}")


def _normalize_label(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    value = _coerce_float(raw.get("value"))
    bbox = raw.get("bbox")
    center = raw.get("center")
    if value is None or (not _is_bbox(bbox) and not _is_point(center)):
        return None
    text = str(raw.get("text", value)).strip()
    confidence = _coerce_float(raw.get("confidence"))
    label: dict[str, Any] = {
        "text": text,
        "value": value,
        "confidence": _clamp01(confidence if confidence is not None else 0.75),
    }
    if _is_bbox(bbox):
        label["bbox"] = [float(item) for item in bbox]
    if _is_point(center):
        label["center"] = [float(item) for item in center]
    return label


def _normalize_dial(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    center = raw.get("center")
    radius = _coerce_float(raw.get("radius"))
    if not _is_point(center) or radius is None or radius <= 0:
        return None
    result: dict[str, Any] = {
        "center": [float(center[0]), float(center[1])],
        "radius": radius,
        "confidence": _clamp01(_coerce_float(raw.get("confidence")) if _coerce_float(raw.get("confidence")) is not None else 0.75),
    }
    if _is_bbox(raw.get("bbox")):
        result["bbox"] = [float(item) for item in raw["bbox"]]
    return result


def _normalize_needle(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    tip = raw.get("tip")
    if not _is_point(tip):
        return None
    result: dict[str, Any] = {
        "tip": [float(tip[0]), float(tip[1])],
        "confidence": _clamp01(_coerce_float(raw.get("confidence")) if _coerce_float(raw.get("confidence")) is not None else 0.75),
    }
    if _is_point(raw.get("base")):
        result["base"] = [float(raw["base"][0]), float(raw["base"][1])]
    return result


def _build_client(genai_module: Any) -> Any:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return genai_module.Client(api_key=api_key)
    return genai_module.Client()


def _guess_mime_type(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type in {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}:
        return mime_type
    if image_path.suffix.lower() == ".avif":
        return "image/avif"
    return "image/jpeg"


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric_value_from_text(text: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", text.replace("O", "0").replace("o", "0"))
    if match is None:
        return None
    return _coerce_float(match.group(0).replace(",", "."))


def _is_bbox(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    return all(_coerce_float(item) is not None for item in value)


def _is_point(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    return all(_coerce_float(item) is not None for item in value)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
