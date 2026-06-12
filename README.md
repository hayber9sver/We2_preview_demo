# himax_preview

A local **Python** live-preview tool for the **Seeed Grove Vision AI Module V2** (Himax WiseEye2 / WE2).
It talks to the board's factory **SenseCraft / SSCMA** firmware over USB serial, decodes the JPEG
camera frames, and draws the AI inference results (detection boxes or hand-pose skeletons) on a live
preview — everything the Himax web toolkit does in the browser, but **without a browser** (no Web Serial).

> Runtime-only: this tool **never flashes firmware** and never writes to flash. It only sends
> runtime AT commands (`AT+INVOKE`, `AT+SAMPLE`, `AT+SENSOR`, `AT+TSCORE`, `AT+BREAK`).

![hand pose demo](docs/demo.png)

## How it works

The board's firmware speaks the SSCMA AT protocol at **921600 baud**. We send a start command and
read back JSON packets framed as `\r{ ... }\n`, each carrying a base64 JPEG plus inference results.
Inference runs on a downscaled frame internally, but the firmware re-scales the output coordinates
back to the returned image resolution, so overlays align at any resolution with no extra scaling.
Full reverse-engineering notes (commands, framing, schema) are in [`PROTOCOL_NOTES.md`](PROTOCOL_NOTES.md).

## Use as a module (for other projects)

The reusable logic lives in **`himax_vision.py`**; `himax_preview.py` is just an example CLI on
top of it. Import `HimaxVision` from your own project:

```python
from himax_vision import HimaxVision

with HimaxVision(port="/dev/ttyACM0", resolution=640, show_preview=True,
                 collect_inference=True, save_log=True, log_path="run.log",
                 rotate=180, tscore=60) as cam:
    print(cam.model)                 # {'name': 'Gesture Detection', 'classes': ['paper','rock','scissors'], ...}
    for frame in cam.stream():       # one Frame per inference event
        for d in frame.detections:   # list[Detection]
            print(d.label, d.score, d.box, d.keypoints)   # e.g. "scissors 74 (cx,cy,w,h) None"
        if frame.image is not None:  # BGR numpy array (preview image), or None
            ...
```

For a **pose / keypoint model** (e.g. Hand Pose), each detection carries `kind == "pose"` and a
`keypoints` list of `(x, y, score, kp_id)` tuples (plus the hand `box`):

```python
from himax_vision import HimaxVision

with HimaxVision(port="/dev/ttyACM0", resolution=640, collect_inference=True,
                 tscore=40, rotate=180) as cam:
    for frame in cam.stream():
        for d in frame.detections:
            if d.kind == "pose":
                cx, cy, w, h = d.box                 # hand bounding box (center-based)
                print("hand", d.score, "at", (cx, cy), "->", len(d.keypoints), "keypoints")
                for x, y, score, kp_id in d.keypoints:
                    print(f"  kp{kp_id}: ({x},{y}) {score}")
            else:                                    # kind == "box" (object detection)
                print(d.label, d.score, d.box)
```

> The same code handles both model types: object detectors fill `kind="box"` (no keypoints),
> pose models fill `kind="pose"` (box + keypoints). Set `show_preview=True` (or `annotate=True`
> with `return_image=True`) to also get the drawn skeleton/box on `frame.image`.

**Inputs** (constructor): `port`, `resolution` (240/480/640), `show_preview` (open a window),
`collect_inference` (parse results), `save_log` + `log_path`. Extras: `rotate`, `tscore`,
`sample` (raw camera), `return_image` (get `Frame.image` without a window), `annotate`.

**Outputs**: `cam.model` (dict), and per `Frame`: `.image` (preview, BGR or `None`),
`.detections` (`list[Detection]` with `kind`/`label`/`score`/`box` center-based/`keypoints`),
`.resolution`, `.perf`, `.raw`. The window's `q` key (or `cam.stop()` / leaving the `with`) ends
the stream. Runtime-only — the module never flashes.

## Requirements

- Linux (developed on an Orange Pi / arm64), Python 3.8+
- A Grove Vision AI Module V2 on **`/dev/ttyACM0`** running its factory SenseCraft firmware
- User in the `dialout` group (for serial access)

```bash
pip install --user pyserial opencv-python numpy
```

> If your system `python3` doesn't have the deps (e.g. it's an older 3.10 while you installed
> them under 3.12), run the tool with the matching interpreter, e.g. `python3.12 himax_preview.py`.

## Quick start

```bash
# Hand-pose detection + skeleton, high-res, upright
python3 himax_preview.py --res 640 --tscore 40 --draw-pose --rotate 180

# Object-detection model: draw boxes instead
python3 himax_preview.py --tscore 40 --draw-inference --rotate 180

# Pure camera preview, full resolution, no inference
python3 himax_preview.py --sample --res 640 --rotate 180

# Plain camera preview (model's resolution)
python3 himax_preview.py --rotate 180
```

Press **`q`** in the preview window to quit.

## Options

| Flag | Description |
|------|-------------|
| `--port PORT` | Serial port (default `/dev/ttyACM0`) |
| `--baud N` | Baud rate (default `921600`) |
| `--res {240,480,640}` | Camera resolution (`640` = 640×480, the sensor max) |
| `--sample` | Pure camera stream (`AT+SAMPLE`, no model/inference) |
| `--tscore 0-100` | Confidence threshold. **Board default 83 is very high and hides most detections — use ~40.** |
| `--draw-pose` | Overlay hand-pose detections (box + 21 keypoints + skeleton) |
| `--draw-inference` | Overlay object-detection boxes (`data.boxes`). Class labels (e.g. `rock`/`paper`/`scissors`) are auto-fetched from the model via `AT+INFO?` |
| `--rotate {0,90,180,270}` | Rotate preview clockwise (use `180` if upside-down) |
| `--frames N` | Capture N frames then stop (`0` = unlimited) |
| `--duration S` | Stop after S seconds |
| `--no-window` | Headless: save annotated PNGs + raw JPEGs + `summary.json` instead of showing a window |
| `--save-dir DIR` | Output directory for `--no-window` |
| `--list-raw` | Dump raw serial bytes to stdout (debugging) |
| `--log-level LEVEL` | `DEBUG`/`INFO`/`WARNING`/`ERROR` (default `INFO`) |

A run log is always written to `himax_preview.log`.

### Headless capture (no display)

```bash
python3 himax_preview.py --tscore 40 --draw-pose --frames 30 \
        --no-window --save-dir captures --rotate 180 --log-level DEBUG
```

Writes `captures/frame_NNN.png` (annotated), `frame_NNN.jpg` (raw), and `summary.json`.

## Troubleshooting

- **`Device or resource busy` / no data** — another program holds the port. The SenseCraft web tool
  in Chromium grabs `/dev/ttyACM0`; close it before running this tool (`fuser /dev/ttyACM0` shows the holder).
- **Inference runs but nothing is detected** — the confidence threshold is too high. Pass `--tscore 40`.
  The board default is **83**, which filters out almost everything.
- **Image is upside-down** — add `--rotate 180` (or `90`/`270` for other mountings).
- **`--draw-pose` shows nothing** — that model must emit `keypoints` (e.g. the *Hand Pose* model) and
  you need a hand in frame. For object detectors use `--draw-inference` (which draws `boxes`).
- **One `WARNING: dropped 1 malformed packet` at startup** — normal. The first frame after the stream
  starts can be partial; it's dropped and the stream continues.

## Notes

- Detection boxes are `data.boxes = [ [cx,cy,w,h,score,target], ... ]` — **center-based** (cx,cy = box center).
- The keypoint schema is `data.keypoints = [ [cx,cy,w,h,score,target], [ [x,y,score,id] × 21 ] ]`
  (bbox also center-based; 21 MediaPipe-style hand landmarks).
- Class names for box labels are read from the model metadata via `AT+INFO?` automatically.
- Resolution options come from `AT+SENSORS?`; set with `AT+SENSOR=<id>,<enable>,<opt_id>`. The
  resolution must be set immediately before the stream command or it reverts.

## License

Provided as-is for use with the Grove Vision AI Module V2. The reverse-engineered protocol details
are derived from Himax/Seeed's publicly distributed web toolkit and SSCMA firmware behavior.
