#!/usr/bin/env python3
"""Overlay COCO RLE segmentations from synth dataset JSON onto the matching image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    "dial": (0, 255, 255),        # yellow
    "casing": (255, 128, 0),      # blue-orange
    "face_plate": (0, 255, 0),    # green
    "scale-label": (255, 0, 255), # magenta
}


def load_annotations(json_path: Path) -> list[dict]:
    with json_path.open() as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "annotations" in data:
        return data["annotations"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported annotation format in {json_path}")


def decode_segmentation(segmentation: dict) -> np.ndarray:
    counts = segmentation["counts"]
    if isinstance(counts, list):
        rle = mask_utils.frPyObjects(segmentation, segmentation["size"][0], segmentation["size"][1])
    else:
        rle = segmentation
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return decoded.astype(bool)


def draw_overlay(image: np.ndarray, annotations: list[dict], alpha: float) -> np.ndarray:
    overlay = image.copy()
    legend_y = 24

    for ann in annotations:
        name = ann.get("category_name", f"id={ann.get('category_id', '?')}")
        color = CATEGORY_COLORS.get(name, (200, 200, 200))
        b, g, r = color

        seg = ann.get("segmentation")
        if seg:
            mask = decode_segmentation(seg)
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            overlay[mask] = (
                overlay[mask].astype(np.float32) * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
            ).astype(np.uint8)

        bbox = ann.get("bbox")
        if bbox:
            x, y, w, h = [int(round(v)) for v in bbox]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

        keypoints = ann.get("keypoints")
        if keypoints:
            for i in range(0, len(keypoints), 3):
                x, y, visible = keypoints[i : i + 3]
                if visible > 0:
                    cv2.circle(overlay, (int(x), int(y)), 6, (0, 0, 255), -1)
                    cv2.circle(overlay, (int(x), int(y)), 6, (255, 255, 255), 1)

        label = name
        if "synth_dial_value" in ann:
            label = f"{name} ({ann['synth_dial_value']:.2f})"
        cv2.putText(
            overlay,
            label,
            (12, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            (12, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
        legend_y += 22

    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", type=Path, help="Annotation JSON file")
    parser.add_argument("image", type=Path, nargs="?", help="Image file (defaults to same stem as JSON)")
    parser.add_argument("-o", "--output", type=Path, help="Output image path")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay opacity (0-1)")
    parser.add_argument("--show", action="store_true", help="Open a preview window")
    args = parser.parse_args()

    image_path = args.image or args.json.with_suffix(".png")
    output_path = args.output or args.json.with_name(args.json.stem + "_segments.jpg")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    annotations = load_annotations(args.json)
    result = draw_overlay(image, annotations, alpha=args.alpha)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    print(f"Wrote {len(annotations)} segments to {output_path}")

    if args.show:
        cv2.imshow("segments", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
