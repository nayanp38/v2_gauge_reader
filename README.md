# Hybrid Analogue Gauge Reader

This project reads single-needle radial analogue gauges by localizing the needle and interpolating between nearby known dial values. It keeps the main pipeline simple and inspectable:

1. Estimate dial geometry.
2. Localize the needle angle.
3. Detect tick marks and numeric labels.
4. Associate labels to ticks.
5. Interpolate the needle position between the closest known values.
6. Fall back to manual calibration or a cloud OCR/VLM adapter when local extraction is weak.

For camera-perspective distortion, the local CV path first tries to fit the dial boundary as an ellipse. Tick, label, and needle angles are then measured in ellipse-normalized dial space, and the effective needle pivot is refined from the convergence of radial tick/needle-like line segments when enough candidates are visible. This is more stable than assuming the photographed gauge is a centered circle.

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

The simplest cloud integration is command-based. Set `GAUGE_READER_CLOUD_COMMAND` to a command that returns JSON:

```bash
export GAUGE_READER_CLOUD_COMMAND='python3 my_vlm_adapter.py {image}'
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

If `GAUGE_READER_CLOUD_COMMAND` is unset, the reader stays local.

## Python API

```python
from gauge_reader import GaugeReader

reader = GaugeReader()
result = reader.read("gauge.jpg", calibration=None, debug_dir="out")
print(result.to_dict())
```

## Current Scope

V1 targets single-needle radial gauges with visible numeric markings. Multi-needle gauges, rectangular/linear gauges, and digital sub-displays are intentionally out of scope.
