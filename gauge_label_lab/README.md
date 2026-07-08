# Gauge Label Lab

A small web-based labelling tool for gauge images. It scans an input folder, serves one pending image at a time, records a gauge value in one central CSV, keeps the source image in place, writes a separate model-view overlay into the labelled output folder, and advances to the next image.

Only files with supported image extensions and image-like headers are opened.

## Run

From the repository root, no install is required:

```bash
python3 label_lab.py --input imgs --labeled labeled_gauge_images --csv gauge_labels.csv
```

Or install this package directly:

```bash
cd gauge_label_lab
python3 -m pip install -e .
gauge-label-lab --input ../imgs --labeled ../imgs_labeled --csv ../labels.csv
```

Then open:

```text
http://127.0.0.1:8765
```

## CSV

The CSV is created automatically with:

```csv
original_filename,labeled_filename,value,labeled_at_utc,source_path,labeled_path
```

`labeled_filename` and `labeled_path` refer to the generated `*_overlay` image in the labelled folder, not a moved source image.

## Supported Images

`.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`, `.tif`, `.tiff`, `.avif`, `.heic`, and `.heif`.
