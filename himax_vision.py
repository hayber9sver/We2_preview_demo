#!/usr/bin/env python3
"""himax_vision — reusable module for the Seeed Grove Vision AI Module V2 (Himax WiseEye2).

Talks to the board's factory SenseCraft / SSCMA firmware over USB serial, decodes the camera
frames and AI inference results, and (optionally) shows a live preview window. Runtime-only:
it NEVER flashes firmware.

------------------------------------------------------------------ I/O contract -------------
INPUT  (HimaxVision constructor):
    port              : USB serial port, e.g. "/dev/ttyACM0"
    resolution        : 240 | 480 | 640  (None = leave board as-is; 640 = 640x480, the max)
    show_preview      : bool — open a live cv2 window (press 'q' to stop the stream)
    collect_inference : bool — parse the model output into Frame.detections
    save_log          : bool — write a run log
    log_path          : where to write the log (None -> "himax_vision.log" when save_log=True)
  (extras: rotate=0/90/180/270, tscore=0..100 confidence, sample=True for raw-camera mode,
   return_image=True to get Frame.image even without a window, annotate to draw overlays.)

OUTPUT:
    .model                      -> dict: {name, category, algorithm, classes}     (the MODEL)
    for frame in .stream():     -> a Frame for every inference event:
        frame.image             -> BGR numpy array (the PREVIEW image) or None
        frame.detections        -> list[Detection]                       (INFERENCE results)
        frame.resolution, .perf, .raw
----------------------------------------------------------------------------------------------

Example:
    from himax_vision import HimaxVision
    with HimaxVision(port="/dev/ttyACM0", resolution=640, show_preview=True,
                     collect_inference=True, save_log=True, log_path="run.log",
                     rotate=180, tscore=60) as cam:
        print(cam.model)                       # {'name': 'Gesture Detection', 'classes': [...]}
        for frame in cam.stream():
            for d in frame.detections:
                print(d.label, d.score, d.box)  # e.g. "scissors 74 (cx,cy,w,h)"
"""

import base64
import binascii
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

log = logging.getLogger("himax_vision")
log.addHandler(logging.NullHandler())  # library: silent unless the user enables logging

# ---- SSCMA protocol constants (reverse-engineered; see PROTOCOL_NOTES.md) ----
DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 921600
INVOKE_CMD = b"AT+INVOKE=-1,0,0\r"   # continuous inference-with-image stream
SAMPLE_CMD = b"AT+SAMPLE=-1\r"       # continuous raw-camera stream (no model)
STOP_CMD = b"AT+BREAK\r"             # stop the stream loop (runtime-only, never writes flash)
SENSOR_ID = 1
RES_OPT = {240: 0, 480: 1, 640: 2}   # pixels -> sensor opt_id (640 = 640x480, the max)

B_LF, B_CR, B_OBRACE, B_CBRACE = 0x0A, 0x0D, 0x7B, 0x7D

PALETTE = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255),
]
# 21-point MediaPipe-style hand skeleton (for pose/keypoint models).
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


# ============================================================ data types ====================
@dataclass
class Detection:
    """One model detection. `box` is CENTER-based (cx, cy, w, h) in the image's resolution."""
    kind: str                                  # "box" (object detection) | "pose" (keypoints)
    class_id: int
    label: str                                 # class name if known, else str(class_id)
    score: float
    box: Tuple[float, float, float, float]     # cx, cy, w, h
    keypoints: Optional[List[Tuple]] = None    # [(x, y, score, kp_id), ...] for pose models


@dataclass
class Frame:
    index: int
    detections: List[Detection]
    resolution: Optional[Tuple[int, int]]
    image: Optional["object"] = None           # BGR numpy array, or None
    jpeg: Optional[bytes] = None               # the raw decoded JPEG bytes, or None
    perf: Optional[list] = None                # [prep, infer, post] ms
    raw: Optional[dict] = None                 # full data packet (image field stripped)


# ============================================================ helpers ========================
class _PacketFramer:
    """Extracts complete JSON packets framed as \\r{ ... }\\n from the byte stream."""

    def __init__(self):
        self.cache = bytearray()
        self.has_start = False
        self.last = 0

    def feed(self, chunk) -> Iterator[bytes]:
        for b in chunk:
            if b == B_OBRACE:
                if self.last == B_CR:
                    self.has_start = True
                    self.cache = bytearray([b])
                elif self.has_start:
                    self.cache.append(b)
            elif b == B_LF:
                if self.last == B_CBRACE and self.has_start:
                    self.has_start = False
                    pkt = bytes(self.cache)
                    self.cache = bytearray()
                    self.last = b
                    yield pkt
                    continue
                elif self.has_start:
                    self.cache.append(b)
            elif self.has_start:
                self.cache.append(b)
            self.last = b
            if len(self.cache) > 8 * 1024 * 1024:
                self.cache = bytearray()
                self.has_start = False


def apply_rotation(frame, deg):
    if not deg:
        return frame
    import cv2
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(deg)
    return frame if code is None else cv2.rotate(frame, code)


def rotate_point(x, y, w, h, deg):
    """Map a point on a WxH image to where it lands after apply_rotation(deg)."""
    if deg == 90:
        return (h - 1 - y, x)
    if deg == 180:
        return (w - 1 - x, h - 1 - y)
    if deg == 270:
        return (y, w - 1 - x)
    return (x, y)


def format_detections(detections: List[Detection]) -> str:
    """Compact one-line summary of detections (for logging)."""
    parts = []
    for d in detections:
        cx, cy, w, h = d.box
        s = "%s %s box=[%g,%g,%g,%g]" % (d.label, d.score, cx, cy, w, h)
        if d.keypoints is not None:
            s += " keypoints=%d" % len(d.keypoints)
        parts.append(s)
    return "; ".join(parts) if parts else "(none)"


# ============================================================ main class =====================
class HimaxVision:
    def __init__(self, port=DEFAULT_PORT, resolution=None, show_preview=False,
                 collect_inference=True, save_log=False, log_path=None,
                 rotate=0, tscore=None, sample=False, baud=DEFAULT_BAUD,
                 return_image=None, annotate=None, window_name="Himax Vision"):
        self.port = port
        self.resolution = resolution
        self.show_preview = show_preview
        self.collect_inference = collect_inference
        self.save_log = save_log
        self.log_path = log_path
        self.rotate = rotate
        self.tscore = tscore
        self.sample = sample
        self.baud = baud
        self.window_name = window_name
        # produce Frame.image when showing a window, or when explicitly requested
        self.return_image = bool(show_preview if return_image is None else return_image)
        self.want_image = self.show_preview or self.return_image
        # draw overlays onto the image; defaults to "whenever we have both an image and results"
        self.annotate = (self.want_image and self.collect_inference) if annotate is None else annotate

        self._ser = None
        self._framer = _PacketFramer()
        self._model: Dict = {}
        self._classes: List[str] = []
        self._opened = False
        self._stop = False
        self.stats = {"frames": 0, "detections_total": 0, "dropped": 0,
                      "first_resolution": None}
        if self.save_log:
            self._setup_logging()

    # ---- logging -------------------------------------------------------------------------
    def _setup_logging(self):
        path = self.log_path or "himax_vision.log"
        log.setLevel(logging.INFO)
        # remove our previous real handlers (keep the NullHandler harmless)
        for h in list(log.handlers):
            if not isinstance(h, logging.NullHandler):
                log.removeHandler(h)
        fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
        try:
            fh = logging.FileHandler(path, mode="w")
            fh.setFormatter(fmt)
            log.addHandler(fh)
            self.log_path = path
        except Exception:
            log.addHandler(logging.StreamHandler())
            log.exception("could not open log file %s", path)

    # ---- properties ----------------------------------------------------------------------
    @property
    def model(self) -> Dict:
        """Loaded model metadata: {name, category, algorithm, classes}."""
        return dict(self._model)

    @property
    def classes(self) -> List[str]:
        return list(self._classes)

    # ---- low-level -----------------------------------------------------------------------
    def _send(self, data, label):
        self._ser.write(data)
        self._ser.flush()
        log.info("sent %s: %r", label, data)

    def _fetch_model(self):
        """Query AT+INFO? and decode the model metadata + class names. Best-effort."""
        try:
            self._send(STOP_CMD, "STOP(pre-INFO)")
            time.sleep(0.3)
            self._ser.reset_input_buffer()
            for attempt in (1, 2):
                self._send(b"AT+INFO?\r", "INFO?(try %d)" % attempt)
                buf, t = "", time.time()
                while time.time() - t < 1.2:
                    chunk = self._ser.read(4096)
                    if chunk:
                        buf += chunk.decode("utf-8", "replace")
                    for ln in buf.replace("\r", "\n").split("\n"):
                        ln = ln.strip()
                        if not (ln.startswith("{") and '"INFO?"' in ln and ln.endswith("}")):
                            continue
                        try:
                            d = json.loads(ln).get("data")
                        except ValueError:
                            continue
                        if isinstance(d, dict) and d.get("info"):
                            meta = json.loads(base64.b64decode(d["info"]).decode("utf-8", "replace"))
                            self._model = {k: meta.get(k) for k in
                                           ("name", "category", "algorithm", "classes")}
                            self._classes = meta.get("classes") or []
                            log.info("model '%s' classes: %s",
                                     self._model.get("name", "?"), self._classes)
                            return
                self._ser.reset_input_buffer()
            log.warning("AT+INFO? returned no model info")
        except Exception:
            log.exception("could not fetch model info (AT+INFO?)")

    def _set_sensor(self, res):
        opt = RES_OPT.get(res)
        if opt is None:
            log.warning("unknown resolution %s; leaving sensor unchanged", res)
            return
        self._send(b"AT+SENSOR=%d,1,%d\r" % (SENSOR_ID, opt), "SENSOR(res=%d)" % res)
        time.sleep(0.5)
        self._ser.read(8192)
        self._ser.reset_input_buffer()

    # ---- lifecycle -----------------------------------------------------------------------
    def open(self):
        """Open the port, read the model, set resolution, and start the stream."""
        import serial
        self._ser = serial.Serial(self.port, self.baud, timeout=0.2)
        log.info("opened %s @ %d", self.port, self.baud)
        self._fetch_model()
        self._send(STOP_CMD, "STOP(pre-clean)")
        time.sleep(0.2)
        self._ser.reset_input_buffer()
        # resolution MUST be set immediately before the stream command or it reverts
        if self.resolution:
            self._set_sensor(self.resolution)
        self._send(SAMPLE_CMD if self.sample else INVOKE_CMD,
                   "START(%s)" % ("SAMPLE" if self.sample else "INVOKE"))
        if not self.sample and self.tscore is not None:
            time.sleep(0.8)
            self._send(b"AT+TSCORE=%d\r" % self.tscore, "TSCORE(%d)" % self.tscore)
            self._ser.reset_input_buffer()
        self._opened = True
        self._stop = False
        return self

    def _name(self, cid):
        return self._classes[cid] if 0 <= cid < len(self._classes) else str(cid)

    def _parse_detections(self, data) -> List[Detection]:
        dets = []
        for b in data.get("boxes") or []:
            if isinstance(b, (list, tuple)) and len(b) >= 6:
                cid = int(b[5])
                dets.append(Detection("box", cid, self._name(cid), b[4],
                                      (b[0], b[1], b[2], b[3]), None))
        for d in data.get("keypoints") or []:
            if (isinstance(d, (list, tuple)) and len(d) >= 1
                    and isinstance(d[0], (list, tuple)) and len(d[0]) >= 6):
                bb = d[0]
                cid = int(bb[5])
                pts = [tuple(p) for p in (d[1] if len(d) > 1 and isinstance(d[1], (list, tuple)) else [])]
                dets.append(Detection("pose", cid, self._name(cid), bb[4],
                                      (bb[0], bb[1], bb[2], bb[3]), pts))
        return dets

    def _decode(self, b64):
        if not b64:
            return None, None
        import cv2
        import numpy as np
        try:
            raw = base64.b64decode(b64)
        except (binascii.Error, ValueError):
            log.warning("dropped 1 frame: partial base64 (len=%s)", len(b64))
            self.stats["dropped"] += 1
            return None, None
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            log.warning("dropped 1 frame: partial/invalid JPEG (raw_len=%s)", len(raw))
            self.stats["dropped"] += 1
            return None, None
        return frame, raw

    def _render(self, frame, detections):
        """Draw detections on the native frame, rotate, then draw upright labels."""
        import cv2
        ow, oh = frame.shape[1], frame.shape[0]
        labels = []
        for d in detections:
            cx, cy, w, h = d.box
            x1, y1 = int(round(cx - w / 2.0)), int(round(cy - h / 2.0))
            x2, y2 = int(round(cx + w / 2.0)), int(round(cy + h / 2.0))
            color = PALETTE[d.class_id % len(PALETTE)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            labels.append(("%s %s" % (d.label, d.score), x1, y1, color))
            if d.keypoints:
                pts = [(int(round(p[0])), int(round(p[1]))) for p in d.keypoints]
                for a, b in HAND_EDGES:
                    if a < len(pts) and b < len(pts):
                        cv2.line(frame, pts[a], pts[b], (255, 255, 0), 1, cv2.LINE_AA)
                for p in pts:
                    cv2.circle(frame, p, 3, (0, 0, 255), -1, cv2.LINE_AA)
        frame = apply_rotation(frame, self.rotate)
        for text, lx, ly, color in labels:
            rx, ry = rotate_point(lx, ly, ow, oh, self.rotate)
            cv2.putText(frame, text, (int(rx), max(12, int(ry) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return frame

    def stream(self, max_frames=None, duration=None) -> Iterator[Frame]:
        """Yield a Frame for each inference event until max_frames/duration/'q'/close()."""
        if not self._opened:
            self.open()
        t0 = time.time()
        try:
            while not self._stop:
                if duration and time.time() - t0 >= duration:
                    break
                if max_frames and self.stats["frames"] >= max_frames:
                    break
                chunk = self._ser.read(4096)
                if not chunk:
                    continue
                for packet in self._framer.feed(chunk):
                    try:
                        obj = json.loads(packet.decode("utf-8", "replace"))
                    except ValueError:
                        log.warning("dropped 1 malformed packet (len=%d)", len(packet))
                        self.stats["dropped"] += 1
                        continue
                    if obj.get("type") != 1 or obj.get("name") not in ("INVOKE", "SAMPLE"):
                        continue
                    raw = obj.get("data")
                    data = raw if isinstance(raw, dict) else {}
                    resolution = tuple(data["resolution"]) if data.get("resolution") else None
                    if resolution and self.stats["first_resolution"] is None:
                        self.stats["first_resolution"] = list(resolution)

                    detections = self._parse_detections(data) if self.collect_inference else []
                    if detections:
                        self.stats["detections_total"] += len(detections)
                        log.info("frame %d inference: %s",
                                 self.stats["frames"], format_detections(detections))

                    image, jpeg = (None, None)
                    if self.want_image:
                        image, jpeg = self._decode(data.get("image"))
                        if image is not None:
                            image = (self._render(image, detections) if self.annotate
                                     else apply_rotation(image, self.rotate))
                            if self.show_preview:
                                import cv2
                                cv2.imshow(self.window_name, image)
                                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                                    self._stop = True

                    frame = Frame(index=self.stats["frames"], detections=detections,
                                  resolution=resolution, image=image, jpeg=jpeg,
                                  perf=data.get("perf"),
                                  raw={k: v for k, v in data.items() if k != "image"})
                    self.stats["frames"] += 1
                    yield frame
                    if self._stop:
                        return
        finally:
            pass  # leave the port open; close() (or the context manager) tears down

    def stop(self):
        self._stop = True

    def close(self):
        if self._ser is not None:
            try:
                self._ser.write(STOP_CMD)
                self._ser.flush()
            except Exception:
                pass
            try:
                self._ser.close()
            except Exception:
                pass
            log.info("closed %s", self.port)
            self._ser = None
        self._opened = False
        if self.show_preview:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False
