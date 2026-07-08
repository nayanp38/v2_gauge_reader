from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

from .models import GaugeReading
from .reader import GaugeReader


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
    ".avif",
    ".heic",
    ".heif",
}

EVAL_PIPELINES = ("hybrid", "local", "cloud-ocr")

RESULT_FIELDS = [
    "sample_id",
    "image_path",
    "json_path",
    "target",
    "gauge_min",
    "gauge_max",
    "gauge_range",
    "target_fraction",
    "prediction",
    "error",
    "abs_error",
    "squared_error",
    "range_normalized_error",
    "abs_range_normalized_error",
    "squared_range_normalized_error",
    "percent_range_error",
    "status",
    "reason",
    "confidence",
    "needle_angle",
    "lower_tick_value",
    "lower_tick_angle",
    "upper_tick_value",
    "upper_tick_angle",
    "latency_ms",
    "used_oracle_calibration",
    "debug_overlay_path",
    "debug_ocr_crop_path",
]


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    image_path: Path
    json_path: Path
    target: float
    gauge_min: float | None = None
    gauge_max: float | None = None
    calibration: dict[str, Any] | None = None

    @property
    def gauge_range(self) -> float | None:
        if self.gauge_min is None or self.gauge_max is None:
            return None
        gauge_range = self.gauge_max - self.gauge_min
        return gauge_range if gauge_range > 0 else None


@dataclass(frozen=True)
class EvaluationPaths:
    output_dir: Path
    results_csv: Path
    summary_json: Path
    overlay_dir: Path | None = None


def discover_samples(
    data_dir: Path,
    *,
    use_oracle_calibration: bool = False,
    limit: int | None = None,
) -> tuple[list[EvaluationSample], dict[str, int]]:
    data_dir = data_dir.expanduser().resolve()
    counters: Counter[str] = Counter()
    samples: list[EvaluationSample] = []

    for json_path in sorted(data_dir.glob("*.json")):
        counters["json_files"] += 1
        image_path = matching_image_path(json_path)
        if image_path is None:
            counters["missing_image"] += 1
            continue

        annotations = load_json(json_path)
        target = find_synth_dial_value(annotations)
        if target is None:
            counters["missing_synth_dial_value"] += 1
            continue

        gauge_min_max = gauge_min_max_from_annotations(annotations)
        if gauge_min_max is None:
            counters["missing_gauge_range"] += 1
            gauge_min = None
            gauge_max = None
        else:
            counters["samples_with_gauge_range"] += 1
            gauge_min, gauge_max = gauge_min_max

        calibration = calibration_from_annotations(annotations) if use_oracle_calibration else None
        if use_oracle_calibration and not calibration:
            counters["missing_oracle_calibration"] += 1

        samples.append(
            EvaluationSample(
                sample_id=json_path.stem,
                image_path=image_path,
                json_path=json_path,
                target=target,
                gauge_min=gauge_min,
                gauge_max=gauge_max,
                calibration=calibration,
            )
        )
        counters["samples"] += 1
        if limit is not None and len(samples) >= limit:
            break

    return samples, dict(counters)


def matching_image_path(json_path: Path) -> Path | None:
    for extension in sorted(IMAGE_EXTENSIONS):
        candidate = json_path.with_suffix(extension)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_synth_dial_value(payload: Any) -> float | None:
    for value in walk_values_for_key(payload, "synth_dial_value"):
        number = coerce_float(value)
        if number is not None:
            return number
    return None


def gauge_min_max_from_annotations(payload: Any) -> tuple[float, float] | None:
    values: list[float] = []
    for annotation in annotations_from_payload(payload):
        category = str(annotation.get("category_name", "")).lower().replace("_", "-")
        if category not in {"scale-label", "scale-labels", "tick-label", "tick-labels"}:
            continue
        value = coerce_float(annotation.get("synth_value"))
        if value is not None:
            values.append(value)
    if len(values) < 2:
        return None
    gauge_min = min(values)
    gauge_max = max(values)
    if gauge_max <= gauge_min:
        return None
    return gauge_min, gauge_max


def calibration_from_annotations(payload: Any) -> dict[str, Any] | None:
    ticks: list[dict[str, Any]] = []
    for annotation in annotations_from_payload(payload):
        category = str(annotation.get("category_name", "")).lower().replace("_", "-")
        if category not in {"scale-label", "scale-labels", "tick-label", "tick-labels"}:
            continue
        value = coerce_float(annotation.get("synth_value"))
        bbox = annotation.get("bbox")
        if value is None or not is_bbox(bbox):
            continue
        x, y, width, height = [float(part) for part in bbox]
        ticks.append(
            {
                "value": value,
                "point": [x + width / 2.0, y + height / 2.0],
                "confidence": 1.0,
            }
        )
    if len(ticks) < 2:
        return None
    return {"ticks": ticks}


def annotations_from_payload(payload: Any) -> list[dict[str, Any]]:
    annotations = payload if isinstance(payload, list) else payload.get("annotations", []) if isinstance(payload, dict) else []
    return [annotation for annotation in annotations if isinstance(annotation, dict)]


def walk_values_for_key(payload: Any, key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(payload, dict):
        if key in payload:
            values.append(payload[key])
        for child in payload.values():
            values.extend(walk_values_for_key(child, key))
    elif isinstance(payload, list):
        for child in payload:
            values.extend(walk_values_for_key(child, key))
    return values


def evaluate_dataset(
    data_dir: Path,
    paths: EvaluationPaths,
    *,
    use_cloud: bool = True,
    force_cloud_ocr: bool = False,
    confidence_threshold: float = 0.65,
    use_oracle_calibration: bool = False,
    limit: int | None = None,
    reader_factory: Callable[[], GaugeReader] | None = None,
    show_progress: bool = False,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    samples, discovery = discover_samples(
        data_dir,
        use_oracle_calibration=use_oracle_calibration,
        limit=limit,
    )
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = paths.overlay_dir or paths.output_dir / "labeled"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    reader = reader_factory() if reader_factory is not None else GaugeReader(
        use_cloud=use_cloud,
        force_cloud_ocr=force_cloud_ocr,
        confidence_threshold=confidence_threshold,
    )

    progress_output = progress_stream or sys.stderr
    if show_progress:
        print_eval_progress(0, len(samples), stream=progress_output)

    for index, sample in enumerate(samples, start=1):
        row = evaluate_sample(
            reader,
            sample,
            use_oracle_calibration=use_oracle_calibration,
            debug_root=overlay_dir,
        )
        rows.append(row)
        if show_progress:
            print_eval_progress(
                index,
                len(samples),
                sample_id=sample.sample_id,
                status=str(row.get("status", "")),
                stream=progress_output,
            )

    write_results_csv(paths.results_csv, rows)
    summary = summarize_rows(rows, discovery=discovery, config={
        "data_dir": str(data_dir.expanduser().resolve()),
        "output_dir": str(paths.output_dir),
        "pipeline": eval_pipeline_name(use_cloud, force_cloud_ocr),
        "use_cloud": use_cloud,
        "force_cloud_ocr": force_cloud_ocr,
        "cloud_adapter": type(getattr(reader, "cloud_adapter", None)).__name__,
        "confidence_threshold": confidence_threshold,
        "use_oracle_calibration": use_oracle_calibration,
        "limit": limit,
        "overlay_dir": str(overlay_dir),
    })
    with paths.summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def print_eval_progress(
    completed: int,
    total: int,
    *,
    sample_id: str | None = None,
    status: str | None = None,
    stream: TextIO,
) -> None:
    percent = 100.0 if total == 0 else (completed / total) * 100.0
    message = f"gauge-eval progress: {completed}/{total} completed ({percent:.1f}%)"
    if sample_id:
        message += f" sample={sample_id}"
    if status:
        message += f" status={status}"
    print(message, file=stream, flush=True)


def evaluate_sample(
    reader: GaugeReader,
    sample: EvaluationSample,
    *,
    use_oracle_calibration: bool,
    debug_root: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    debug_dir = debug_root / sample.sample_id if debug_root is not None else None
    try:
        reading = read_with_optional_debug(
            reader,
            sample.image_path,
            calibration=sample.calibration if use_oracle_calibration else None,
            debug_dir=debug_dir,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return row_from_reading(sample, reading, latency_ms, use_oracle_calibration)
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "sample_id": sample.sample_id,
            "image_path": str(sample.image_path),
            "json_path": str(sample.json_path),
            "target": sample.target,
            "gauge_min": value_or_blank(sample.gauge_min),
            "gauge_max": value_or_blank(sample.gauge_max),
            "gauge_range": value_or_blank(sample.gauge_range),
            "target_fraction": value_or_blank(target_fraction(sample)),
            "prediction": "",
            "error": "",
            "abs_error": "",
            "squared_error": "",
            "range_normalized_error": "",
            "abs_range_normalized_error": "",
            "squared_range_normalized_error": "",
            "percent_range_error": "",
            "status": "exception",
            "reason": f"{type(exc).__name__}: {exc}",
            "confidence": 0.0,
            "needle_angle": "",
            "lower_tick_value": "",
            "lower_tick_angle": "",
            "upper_tick_value": "",
            "upper_tick_angle": "",
            "latency_ms": latency_ms,
            "used_oracle_calibration": bool(use_oracle_calibration and sample.calibration),
            "debug_overlay_path": "",
            "debug_ocr_crop_path": "",
        }


def read_with_optional_debug(
    reader: GaugeReader,
    image_path: Path,
    *,
    calibration: dict[str, Any] | None,
    debug_dir: Path | None,
) -> GaugeReading:
    if debug_dir is None:
        return reader.read(image_path, calibration=calibration)
    try:
        parameters = inspect.signature(reader.read).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_debug = "debug_dir" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if not accepts_debug:
        return reader.read(image_path, calibration=calibration)
    return reader.read(image_path, calibration=calibration, debug_dir=debug_dir)


def row_from_reading(
    sample: EvaluationSample,
    reading: GaugeReading,
    latency_ms: float,
    use_oracle_calibration: bool,
) -> dict[str, Any]:
    prediction = reading.reading
    error = prediction - sample.target if prediction is not None else None
    abs_error = abs(error) if error is not None else None
    squared_error = error * error if error is not None else None
    range_error = normalized_error(error, sample.gauge_range)
    abs_range_error = abs(range_error) if range_error is not None else None
    squared_range_error = range_error * range_error if range_error is not None else None
    percent_range_error = abs_range_error * 100.0 if abs_range_error is not None else None
    return {
        "sample_id": sample.sample_id,
        "image_path": str(sample.image_path),
        "json_path": str(sample.json_path),
        "target": sample.target,
        "gauge_min": value_or_blank(sample.gauge_min),
        "gauge_max": value_or_blank(sample.gauge_max),
        "gauge_range": value_or_blank(sample.gauge_range),
        "target_fraction": value_or_blank(target_fraction(sample)),
        "prediction": value_or_blank(prediction),
        "error": value_or_blank(error),
        "abs_error": value_or_blank(abs_error),
        "squared_error": value_or_blank(squared_error),
        "range_normalized_error": value_or_blank(range_error),
        "abs_range_normalized_error": value_or_blank(abs_range_error),
        "squared_range_normalized_error": value_or_blank(squared_range_error),
        "percent_range_error": value_or_blank(percent_range_error),
        "status": reading.status,
        "reason": reading.reason,
        "confidence": reading.confidence,
        "needle_angle": value_or_blank(reading.needle_angle),
        "lower_tick_value": value_or_blank(reading.lower_tick.value if reading.lower_tick else None),
        "lower_tick_angle": value_or_blank(reading.lower_tick.angle if reading.lower_tick else None),
        "upper_tick_value": value_or_blank(reading.upper_tick.value if reading.upper_tick else None),
        "upper_tick_angle": value_or_blank(reading.upper_tick.angle if reading.upper_tick else None),
        "latency_ms": latency_ms,
        "used_oracle_calibration": bool(use_oracle_calibration and sample.calibration),
        "debug_overlay_path": reading.debug.get("overlay", ""),
        "debug_ocr_crop_path": reading.debug.get("ocr_crop", ""),
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    discovery: dict[str, int],
    config: dict[str, Any],
) -> dict[str, Any]:
    predicted_rows = [row for row in rows if coerce_float(row.get("prediction")) is not None]
    ok_rows = [row for row in predicted_rows if row.get("status") == "ok"]
    range_rows = [row for row in predicted_rows if coerce_float(row.get("abs_range_normalized_error")) is not None]
    ok_range_rows = [row for row in ok_rows if coerce_float(row.get("abs_range_normalized_error")) is not None]
    latencies = [float(row["latency_ms"]) for row in rows if coerce_float(row.get("latency_ms")) is not None]

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "discovery": discovery,
        "counts": {
            "evaluated": len(rows),
            "predicted": len(predicted_rows),
            "coverage": len(predicted_rows) / len(rows) if rows else 0.0,
            "range_normalized": len(range_rows),
            "range_normalized_coverage": len(range_rows) / len(rows) if rows else 0.0,
            "ok_predicted": len(ok_rows),
            "ok_coverage": len(ok_rows) / len(rows) if rows else 0.0,
            "ok_range_normalized": len(ok_range_rows),
            "status": dict(Counter(str(row["status"]) for row in rows)),
            "reason": dict(Counter(str(row["reason"]) for row in rows)),
        },
        "error_metrics": error_metrics_for_rows(predicted_rows),
        "ok_error_metrics": error_metrics_for_rows(ok_rows),
        "latency_ms": {
            "mean": mean(latencies),
            "median": median(latencies),
            "p95": percentile(latencies, 95),
            "max": max(latencies) if latencies else None,
        },
    }


def error_metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    abs_errors = [float(row["abs_error"]) for row in rows if coerce_float(row.get("abs_error")) is not None]
    errors = [float(row["error"]) for row in rows if coerce_float(row.get("error")) is not None]
    squared_errors = [float(row["squared_error"]) for row in rows if coerce_float(row.get("squared_error")) is not None]
    abs_range_errors = [
        float(row["abs_range_normalized_error"])
        for row in rows
        if coerce_float(row.get("abs_range_normalized_error")) is not None
    ]
    range_errors = [
        float(row["range_normalized_error"])
        for row in rows
        if coerce_float(row.get("range_normalized_error")) is not None
    ]
    squared_range_errors = [
        float(row["squared_range_normalized_error"])
        for row in rows
        if coerce_float(row.get("squared_range_normalized_error")) is not None
    ]
    percent_range_errors = [
        float(row["percent_range_error"])
        for row in rows
        if coerce_float(row.get("percent_range_error")) is not None
    ]
    return {
        "mae": mean(abs_errors),
        "median_abs_error": median(abs_errors),
        "rmse": math.sqrt(mean(squared_errors)) if squared_errors else None,
        "mean_error_bias": mean(errors),
        "max_abs_error": max(abs_errors) if abs_errors else None,
        "range_normalized_mae": mean(abs_range_errors),
        "range_normalized_median_abs_error": median(abs_range_errors),
        "range_normalized_rmse": math.sqrt(mean(squared_range_errors)) if squared_range_errors else None,
        "range_normalized_mean_bias": mean(range_errors),
        "max_abs_range_normalized_error": max(abs_range_errors) if abs_range_errors else None,
        "mean_percent_range_error": mean(percent_range_errors),
        "median_percent_range_error": median(percent_range_errors),
        "max_percent_range_error": max(percent_range_errors) if percent_range_errors else None,
        "within_abs_0_1": within(abs_errors, 0.1),
        "within_abs_0_25": within(abs_errors, 0.25),
        "within_abs_0_5": within(abs_errors, 0.5),
        "within_abs_1_0": within(abs_errors, 1.0),
        "within_1_percent_range": within(percent_range_errors, 1.0),
        "within_2_percent_range": within(percent_range_errors, 2.0),
        "within_5_percent_range": within(percent_range_errors, 5.0),
        "within_10_percent_range": within(percent_range_errors, 10.0),
    }


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def default_paths(output_dir: Path | None) -> EvaluationPaths:
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = Path("eval_runs") / stamp
    output_dir = output_dir.expanduser().resolve()
    return EvaluationPaths(
        output_dir=output_dir,
        results_csv=output_dir / "per_sample_results.csv",
        summary_json=output_dir / "summary.json",
        overlay_dir=output_dir / "labeled",
    )


def coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def normalized_error(error: float | None, gauge_range: float | None) -> float | None:
    if error is None or gauge_range is None or gauge_range <= 0:
        return None
    return error / gauge_range


def target_fraction(sample: EvaluationSample) -> float | None:
    gauge_range = sample.gauge_range
    if sample.gauge_min is None or gauge_range is None:
        return None
    return (sample.target - sample.gauge_min) / gauge_range


def is_bbox(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 4 and all(coerce_float(part) is not None for part in value)


def value_or_blank(value: Any) -> Any:
    return "" if value is None else value


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def within(abs_errors: list[float], threshold: float) -> float | None:
    if not abs_errors:
        return None
    return sum(1 for error in abs_errors if error <= threshold) / len(abs_errors)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate gauge-reader predictions on a synthetic labelled dataset.")
    parser.add_argument("data_dir", help="Folder containing same-stem .png/.json gauge samples.")
    parser.add_argument("--output", help="Output folder for summary.json and per_sample_results.csv.")
    parser.add_argument("--limit", type=int, help="Maximum number of samples to evaluate.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-sample progress messages.",
    )
    parser.add_argument(
        "--pipeline",
        choices=EVAL_PIPELINES,
        help=(
            "Evaluation pipeline preset: hybrid uses local CV with cloud fallback, "
            "local disables cloud, and cloud-ocr forces cloud numeric OCR."
        ),
    )
    parser.add_argument("--no-cloud", action="store_true", help="Disable cloud fallback during evaluation.")
    parser.add_argument(
        "--force-cloud-ocr",
        action="store_true",
        help="Always use the cloud adapter for numeric OCR labels instead of local Tesseract labels.",
    )
    parser.add_argument(
        "--oracle-calibration",
        action="store_true",
        help="Use synth scale-label annotations as calibration ticks to isolate needle/dial geometry error.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.65,
        help="GaugeReader confidence threshold.",
    )
    return parser


def resolve_eval_pipeline(
    pipeline: str | None,
    *,
    no_cloud: bool,
    force_cloud_ocr: bool,
) -> tuple[bool, bool]:
    if pipeline == "local":
        use_cloud = False
        force_ocr = False
    elif pipeline == "cloud-ocr":
        use_cloud = True
        force_ocr = True
    else:
        use_cloud = True
        force_ocr = False

    if no_cloud:
        use_cloud = False
    if force_cloud_ocr:
        force_ocr = True
    return use_cloud, force_ocr


def eval_pipeline_name(use_cloud: bool, force_cloud_ocr: bool) -> str:
    if force_cloud_ocr:
        return "cloud-ocr"
    if not use_cloud:
        return "local"
    return "hybrid"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    use_cloud, force_cloud_ocr = resolve_eval_pipeline(
        args.pipeline,
        no_cloud=args.no_cloud,
        force_cloud_ocr=args.force_cloud_ocr,
    )
    if not use_cloud and force_cloud_ocr:
        parser.error("--force-cloud-ocr cannot be used with --no-cloud")
    paths = default_paths(Path(args.output) if args.output else None)
    summary = evaluate_dataset(
        Path(args.data_dir),
        paths,
        use_cloud=use_cloud,
        force_cloud_ocr=force_cloud_ocr,
        confidence_threshold=args.confidence_threshold,
        use_oracle_calibration=args.oracle_calibration,
        limit=args.limit,
        show_progress=not args.no_progress,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Per-sample results: {paths.results_csv}")
    print(f"Summary: {paths.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
