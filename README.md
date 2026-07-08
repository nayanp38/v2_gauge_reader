# Hybrid Analogue Gauge Reader

This project reads single-needle radial analogue gauges by localizing the needle and interpolating between nearby known dial values. It keeps the main pipeline simple and inspectable:

1. Estimate dial geometry.
2. Localize the needle angle.
3. Detect tick marks and numeric labels.
4. Associate labels to ticks.
5. Interpolate the needle position between the closest known values.
6. Fall back to manual calibration or a cloud OCR/VLM adapter when local extraction is weak.

For camera-perspective distortion, the local CV path first tries to fit the dial boundary as an ellipse. Tick, label, and needle angles are then measured in ellipse-normalized dial space, and the effective needle pivot is refined from the convergence of radial tick/needle-like line segments when enough candidates are visible. This is more stable than assuming the photographed gauge is a centered circle.

After estimating the dial, OCR runs on a padded crop around the dial instead of the full image. Cloud OCR/VLM calls also receive this crop when a dial was found, and all returned coordinates are translated back into the original image before interpolation and debug overlays.

## Install

Core geometry and tests use only the Python standard library. For real image reading, install the CV extras:

```bash
python3 -m pip install -e ".[cv,ocr]"
```

`pytesseract` also requires the Tesseract binary to be installed on the machine. Cloud fallback is optional:

```bash
python3 -m pip install -e ".[cloud]"
```

## CLI

```bash
gauge-read path/to/gauge.jpg --debug out/
```

When debug output is enabled, `ocr_crop.jpg` shows the cropped dial image used for OCR.

The CLI prints JSON:

```json
{
  "reading": 42.1,
  "unit": "psi",
  "confidence": 0.82,
  "status": "ok",
  "reason": "ok",
  "needle_angle": 117.3,
  "lower_tick": {"value": 40, "angle": 125.0, "source": "ocr"},
  "upper_tick": {"value": 50, "angle": 101.0, "source": "ocr"},
  "debug": {"overlay": "out/overlay.jpg"}
}
```

## Manual Calibration

When OCR is unreliable, provide a calibration JSON file:

```json
{
  "unit": "psi",
  "ticks": [
    {"value": 0, "angle": 225},
    {"value": 50, "angle": 90},
    {"value": 100, "angle": 315}
  ]
}
```

Tick entries can use `angle`, `angle_deg`, or `point`:

```json
{"value": 50, "point": [320, 120]}
```

Point-based calibration is converted to an angle using the detected dial center.

## Cloud Fallback

Install the cloud extra and set a Gemini API key:

```bash
python3 -m pip install -e ".[cloud]"
export GEMINI_API_KEY='your-api-key'
```

To force numeric label OCR through the built-in Gemini Robotics-ER adapter even when local OCR produces labels, add:

```bash
gauge-read gauge.jpg --debug out/ --force-cloud-ocr
```

This still uses local CV for dial, needle, and tick geometry when available. Manual calibration ticks override OCR for interpolation, but `--force-cloud-ocr` still calls the cloud adapter so you can inspect Gemini output in debug/eval runs.

When cloud OCR appears to miss boxes, dump the raw Gemini response and parser counts:

```bash
export GAUGE_READER_GEMINI_DIAGNOSTICS='out/gemini_diagnostics.json'
gauge-read gauge.jpg --debug out/ --force-cloud-ocr
```

If `raw_labels` is greater than `converted_labels`, Gemini found labels but the response shape did not include usable geometry for every label.

You can choose a model with:

```bash
export GAUGE_READER_GEMINI_MODEL='gemini-robotics-er-1.6-preview'
```

The adapter focuses on visible numeric dial labels and returns JSON to the main reader:

```bash
gauge-vlm-gemini gauge.jpg
```

Expected output:

```json
{
  "unit": "psi",
  "labels": [
    {"text": "40", "value": 40, "bbox": [100, 120, 32, 18], "confidence": 0.92}
  ],
  "ticks": [
    {"angle": 122.5, "confidence": 0.8}
  ]
}
```

You can also provide your own command by setting `GAUGE_READER_CLOUD_COMMAND` to any executable that prints the same JSON shape. The command adapter takes precedence over the built-in Gemini adapter.

## Python API

```python
from gauge_reader import GaugeReader

reader = GaugeReader()
result = reader.read("gauge.jpg", calibration=None, debug_dir="out")
print(result.to_dict())
```

To force cloud OCR labels from Python:

```python
reader = GaugeReader(force_cloud_ocr=True)
```

## Dataset Evaluation

From the repo root, evaluate a folder of same-stem `.png`/`.json` pairs whose JSON contains `synth_dial_value`:

```bash
python3 evaluate_dataset.py /Users/nayan.patel/Downloads/archive/sample_synth_datasets/ds6.0/data --output eval_runs/ds6_zero_shot --no-cloud
```

If your global `python3` does not have the CV extras installed, use the repo venv:

```bash
.venv/bin/python evaluate_dataset.py /Users/nayan.patel/Downloads/archive/sample_synth_datasets/ds6.0/data --output eval_runs/ds6_zero_shot --no-cloud
```

The evaluator writes `summary.json`, `per_sample_results.csv`, and per-sample model-view overlays under `OUTPUT/labeled/<sample_id>/overlay.jpg`. Source dataset images remain in their original folder. To isolate needle/dial geometry from OCR quality, use the synthetic `scale-label` annotations as calibration ticks:

While running from the CLI, progress is printed to stderr as `completed/total` after each sample. Add `--no-progress` to silence it.

For cross-gauge comparisons, the evaluator also reports range-normalized error when the JSON contains at least two `scale-label`/`tick-label` `synth_value`s. `percent_range_error` is `abs(prediction - target) / (gauge_max - gauge_min) * 100`, which is usually more comparable than raw-unit error across gauges with different scales.

```bash
python3 evaluate_dataset.py /Users/nayan.patel/Downloads/archive/sample_synth_datasets/ds6.0/data --output eval_runs/ds6_oracle --no-cloud --oracle-calibration
```

To benchmark the cloud OCR path directly, include `--force-cloud-ocr` and leave cloud enabled:

```bash
.venv/bin/python evaluate_dataset.py /Users/nayan.patel/Downloads/archive/sample_synth_datasets/ds6.0/data --output eval_runs/ds6_cloud_ocr --force-cloud-ocr
```

Equivalent named eval pipeline:

```bash
.venv/bin/python evaluate_dataset.py /Users/nayan.patel/Downloads/archive/sample_synth_datasets/ds6.0/data --output eval_runs/ds6_cloud_ocr --pipeline cloud-ocr
```

## Current Scope

V1 targets single-needle radial gauges with visible numeric markings. Multi-needle gauges, rectangular/linear gauges, and digital sub-displays are intentionally out of scope.

## Gauge Label Lab

From the repo root, run the web labelling tool without installing the separate package:

```bash
python3 label_lab.py --input imgs --labeled labeled_gauge_images --csv gauge_labels.csv
```

Then open `http://127.0.0.1:8765`.

When a label is saved, the original image stays in the input folder. The labeled folder receives a separate `*_overlay` image generated from the gauge reader debug view, and the CSV row points `labeled_path` at that overlay.
