from __future__ import annotations

import argparse
import json
import sys

from .reader import GaugeReader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read a single-needle radial analogue gauge.")
    parser.add_argument("image", help="Path to the gauge image.")
    parser.add_argument("--calibration", help="Optional calibration JSON path.")
    parser.add_argument("--debug", help="Directory for debug overlays.")
    parser.add_argument(
        "--no-cloud",
        action="store_true",
        help="Disable cloud OCR/VLM fallback even when GAUGE_READER_CLOUD_COMMAND is configured.",
    )
    parser.add_argument(
        "--force-cloud-ocr",
        action="store_true",
        help="Always use the cloud adapter for numeric OCR labels instead of local Tesseract labels.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.65,
        help="Minimum confidence for status=ok.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.no_cloud and args.force_cloud_ocr:
        parser.error("--force-cloud-ocr cannot be used with --no-cloud")
    reader = GaugeReader(
        use_cloud=not args.no_cloud,
        force_cloud_ocr=args.force_cloud_ocr,
        confidence_threshold=args.confidence_threshold,
    )
    result = reader.read(
        args.image,
        calibration=args.calibration,
        debug_dir=args.debug,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.status in {"ok", "low_confidence"} else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
