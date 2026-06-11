from __future__ import annotations

import re
from typing import Any

from .models import BBox, NumericLabel

NUMERIC_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def parse_numeric_text(text: str) -> float | None:
    match = NUMERIC_RE.search(text.replace("O", "0").replace("o", "0"))
    if match is None:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def detect_labels_with_tesseract(image: Any) -> list[NumericLabel]:
    """Best-effort local OCR. Returns an empty list when dependencies are absent."""
    try:
        from PIL import Image
        import pytesseract
    except Exception:
        return []

    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(image)
        except Exception:
            return []

    config = "--psm 11 -c tessedit_char_whitelist=0123456789.-,+"
    try:
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    except Exception:
        return []

    labels: list[NumericLabel] = []
    for index, text in enumerate(data.get("text", [])):
        value = parse_numeric_text(str(text))
        if value is None:
            continue
        confidence = _parse_confidence(data.get("conf", ["0"])[index])
        bbox = BBox(
            float(data["left"][index]),
            float(data["top"][index]),
            float(data["width"][index]),
            float(data["height"][index]),
        )
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        labels.append(
            NumericLabel(
                value=value,
                text=str(text),
                bbox=bbox,
                confidence=confidence,
                source="local_tesseract",
            )
        )
    return labels


def _parse_confidence(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric < 0:
        return 0.0
    if numeric > 1.0:
        numeric /= 100.0
    return max(0.0, min(1.0, numeric))
