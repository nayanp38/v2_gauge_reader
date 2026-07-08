from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import posixpath
import shutil
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote, urlparse


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".avif",
    ".heic",
    ".heif",
}

CSV_FIELDS = [
    "original_filename",
    "labeled_filename",
    "value",
    "labeled_at_utc",
    "source_path",
    "labeled_path",
]


@dataclass(frozen=True)
class LabelRecord:
    original_filename: str
    labeled_filename: str
    value: str
    labeled_at_utc: str
    source_path: str
    labeled_path: str

    def to_row(self) -> dict[str, str]:
        return {
            "original_filename": self.original_filename,
            "labeled_filename": self.labeled_filename,
            "value": self.value,
            "labeled_at_utc": self.labeled_at_utc,
            "source_path": self.source_path,
            "labeled_path": self.labeled_path,
        }


class LabelStore:
    def __init__(
        self,
        input_dir: Path,
        labeled_dir: Path,
        csv_path: Path,
        overlay_writer: Callable[[Path, Path], Path] | None = None,
    ):
        self.input_dir = input_dir.expanduser().resolve()
        self.labeled_dir = labeled_dir.expanduser().resolve()
        self.csv_path = csv_path.expanduser().resolve()
        self.overlay_writer = overlay_writer or write_model_overlay
        self._lock = threading.Lock()

    def ensure_ready(self) -> None:
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            raise FileNotFoundError(f"input folder does not exist: {self.input_dir}")
        self.labeled_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()

    def pending_images(self) -> list[Path]:
        labeled_sources = self._labeled_source_paths()
        images = [
            path
            for path in self.input_dir.iterdir()
            if path.is_file() and is_supported_image(path)
            and str(path.resolve()) not in labeled_sources
        ]
        return sorted(images, key=lambda path: path.name.casefold())

    def current_image(self) -> Path | None:
        images = self.pending_images()
        return images[0] if images else None

    def state(self) -> dict[str, Any]:
        images = self.pending_images()
        current = images[0] if images else None
        labeled_count = self._labeled_count()
        return {
            "input_dir": str(self.input_dir),
            "labeled_dir": str(self.labeled_dir),
            "csv_path": str(self.csv_path),
            "remaining": len(images),
            "labeled_count": labeled_count,
            "current": image_payload(current) if current is not None else None,
        }

    def image_path_for_name(self, filename: str) -> Path:
        safe_name = Path(filename).name
        path = (self.input_dir / safe_name).resolve()
        if path.parent != self.input_dir:
            raise ValueError("invalid image path")
        if not path.exists() or not path.is_file() or not is_supported_image(path):
            raise FileNotFoundError(f"image is not pending or not supported: {safe_name}")
        if str(path.resolve()) in self._labeled_source_paths():
            raise FileNotFoundError(f"image is already labeled: {safe_name}")
        return path

    def label_image(self, filename: str, value: str) -> LabelRecord:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("label value is required")

        with self._lock:
            source = self.image_path_for_name(filename)
            self.labeled_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="gauge_label_overlay_") as tmpdir:
                overlay_source = self.overlay_writer(source, Path(tmpdir))
                overlay_suffix = overlay_source.suffix or source.suffix
                destination = unique_destination(
                    self.labeled_dir,
                    f"{source.stem}_overlay{overlay_suffix}",
                )
                shutil.copy2(overlay_source, destination)

            record = LabelRecord(
                original_filename=source.name,
                labeled_filename=destination.name,
                value=cleaned_value,
                labeled_at_utc=datetime.now(timezone.utc).isoformat(),
                source_path=str(source),
                labeled_path=str(destination),
            )
            self._append_record(record)
            return record

    def _append_record(self, record: LabelRecord) -> None:
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writerow(record.to_row())

    def _labeled_count(self) -> int:
        if not self.csv_path.exists():
            return 0
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return sum(1 for _ in reader)

    def _labeled_source_paths(self) -> set[str]:
        if not self.csv_path.exists():
            return set()

        labeled: set[str] = set()
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source_path = row.get("source_path")
                if source_path:
                    labeled.add(str(Path(source_path).expanduser().resolve()))
                    continue

                original_filename = row.get("original_filename")
                if original_filename:
                    labeled.add(str((self.input_dir / Path(original_filename).name).resolve()))
        return labeled


def write_model_overlay(source: Path, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = work_dir / "debug"
    try:
        from gauge_reader import GaugeReader

        reading = GaugeReader(use_cloud=False).read(source, debug_dir=debug_dir)
        overlay = Path(reading.debug.get("overlay", ""))
        if overlay.exists() and overlay.is_file():
            return overlay
    except Exception:
        pass

    fallback = work_dir / f"{source.stem}_overlay{source.suffix}"
    shutil.copy2(source, fallback)
    return fallback


def image_payload(path: Path) -> dict[str, str]:
    return {
        "filename": path.name,
        "url": f"/image/{quote(path.name)}",
    }


def is_supported_image(path: Path) -> bool:
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    try:
        header = path.read_bytes()[:32]
    except OSError:
        return False
    return has_image_signature(header)


def has_image_signature(header: bytes) -> bool:
    if header.startswith(b"\xff\xd8\xff"):
        return True
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if header.startswith((b"GIF87a", b"GIF89a")):
        return True
    if header.startswith(b"BM"):
        return True
    if header.startswith((b"II*\x00", b"MM\x00*")):
        return True
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return True
    if len(header) >= 16 and header[4:8] == b"ftyp":
        brands = header[8:32]
        return any(brand in brands for brand in (b"avif", b"avis", b"heic", b"heif", b"mif1", b"msf1"))
    return False


def unique_destination(folder: Path, filename: str) -> Path:
    candidate = folder / Path(filename).name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        renamed = folder / f"{stem}_{counter:04d}{suffix}"
        if not renamed.exists():
            return renamed
        counter += 1


class LabelRequestHandler(BaseHTTPRequestHandler):
    store: LabelStore

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = posixpath.normpath(parsed.path)
        if path == "/":
            self._send_html(INDEX_HTML)
            return
        if path == "/api/state":
            self._send_json(self.store.state())
            return
        if path.startswith("/image/"):
            self._send_image(unquote(path.removeprefix("/image/")))
            return
        self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/label":
            self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self._read_json()
            filename = str(payload.get("filename", ""))
            value = str(payload.get("value", ""))
            record = self.store.label_image(filename, value)
            self._send_json({"record": record.to_row(), "state": self.store.state()})
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body or "{}")
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_image(self, filename: str) -> None:
        try:
            path = self.store.image_path_for_name(filename)
            body = path.read_bytes()
        except (ValueError, FileNotFoundError, OSError):
            self._send_json({"error": "image_not_found"}, HTTPStatus.NOT_FOUND)
            return

        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def make_handler(store: LabelStore) -> type[LabelRequestHandler]:
    class BoundLabelRequestHandler(LabelRequestHandler):
        pass

    BoundLabelRequestHandler.store = store
    return BoundLabelRequestHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a web labelling tool for gauge images.")
    parser.add_argument("--input", required=True, help="Folder containing unlabelled gauge images.")
    parser.add_argument("--labeled", required=True, help="Folder where generated label overlays are written.")
    parser.add_argument("--csv", required=True, help="Central CSV path for labels.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = LabelStore(Path(args.input), Path(args.labeled), Path(args.csv))
    store.ensure_ready()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    host, port = server.server_address
    print(f"Gauge Label Lab running at http://{host}:{port}")
    print(f"Input: {store.input_dir}")
    print(f"Labeled: {store.labeled_dir}")
    print(f"CSV: {store.csv_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Gauge Label Lab")
    finally:
        server.server_close()
    return 0


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gauge Label Lab</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #647087;
      --line: #d8deea;
      --accent: #0f766e;
      --accent-dark: #0b5f59;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
      font-weight: 700;
      letter-spacing: 0;
    }
    .stats {
      display: flex;
      align-items: center;
      gap: 16px;
      color: var(--muted);
      font-size: 14px;
      white-space: nowrap;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 0;
      min-height: 0;
    }
    .viewer {
      min-height: 0;
      padding: 18px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: auto;
    }
    .image-frame {
      width: 100%;
      height: calc(100vh - 100px);
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--line);
      background: #101828;
      border-radius: 8px;
      overflow: hidden;
    }
    .image-frame img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: block;
    }
    .empty {
      color: #ffffff;
      font-size: 18px;
    }
    aside {
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .filename {
      font-size: 14px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    label {
      display: block;
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 8px;
    }
    input {
      width: 100%;
      height: 42px;
      padding: 8px 10px;
      font-size: 18px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--text);
    }
    button {
      width: 100%;
      height: 42px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    .status {
      min-height: 22px;
      font-size: 14px;
      color: var(--muted);
    }
    .status.error { color: var(--danger); }
    .paths {
      margin-top: auto;
      padding-top: 16px;
      border-top: 1px solid var(--line);
      display: grid;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    @media (max-width: 850px) {
      main { grid-template-columns: 1fr; }
      aside { border-left: 0; border-top: 1px solid var(--line); }
      .image-frame { height: 56vh; }
      header { align-items: flex-start; flex-direction: column; }
      .stats { flex-wrap: wrap; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1>Gauge Label Lab</h1>
      <div class="stats">
        <span id="remaining">Remaining: 0</span>
        <span id="labeled">Labeled: 0</span>
      </div>
    </header>
    <main>
      <section class="viewer">
        <div class="image-frame" id="frame">
          <div class="empty">Loading</div>
        </div>
      </section>
      <aside>
        <div>
          <label>Image</label>
          <div class="filename" id="filename">-</div>
        </div>
        <form id="form">
          <label for="value">Gauge Value</label>
          <input id="value" name="value" autocomplete="off" inputmode="decimal" autofocus>
          <div style="height: 12px"></div>
          <button id="submit" type="submit">Save Label</button>
        </form>
        <div class="status" id="status"></div>
        <div class="paths">
          <div id="inputPath"></div>
          <div id="csvPath"></div>
        </div>
      </aside>
    </main>
  </div>
  <script>
    let current = null;

    async function loadState() {
      const response = await fetch('/api/state', {cache: 'no-store'});
      const state = await response.json();
      current = state.current;
      document.getElementById('remaining').textContent = `Remaining: ${state.remaining}`;
      document.getElementById('labeled').textContent = `Labeled: ${state.labeled_count}`;
      document.getElementById('inputPath').textContent = `Input: ${state.input_dir}`;
      document.getElementById('csvPath').textContent = `CSV: ${state.csv_path}`;

      const frame = document.getElementById('frame');
      const filename = document.getElementById('filename');
      const value = document.getElementById('value');
      const submit = document.getElementById('submit');
      frame.innerHTML = '';
      value.value = '';

      if (!current) {
        frame.innerHTML = '<div class="empty">All images labeled</div>';
        filename.textContent = '-';
        value.disabled = true;
        submit.disabled = true;
        setStatus('');
        return;
      }

      const img = document.createElement('img');
      img.src = current.url;
      img.alt = current.filename;
      frame.appendChild(img);
      filename.textContent = current.filename;
      value.disabled = false;
      submit.disabled = false;
      value.focus();
      setStatus('');
    }

    function setStatus(message, isError = false) {
      const status = document.getElementById('status');
      status.textContent = message;
      status.className = isError ? 'status error' : 'status';
    }

    document.getElementById('form').addEventListener('submit', async (event) => {
      event.preventDefault();
      if (!current) return;
      const value = document.getElementById('value').value.trim();
      if (!value) {
        setStatus('Enter a value before saving.', true);
        return;
      }
      document.getElementById('submit').disabled = true;
      setStatus('Saving...');
      const response = await fetch('/api/label', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({filename: current.filename, value})
      });
      const payload = await response.json();
      if (!response.ok) {
        setStatus(payload.error || 'Unable to save label.', true);
        document.getElementById('submit').disabled = false;
        return;
      }
      await loadState();
    });

    loadState().catch((error) => setStatus(error.message, true));
  </script>
</body>
</html>
"""
