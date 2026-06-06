#!/usr/bin/env python3
"""
himax_preview.py — local Python port of the Himax/SenseCraft web toolkit live preview.

Reads camera frames + AI inference results from a Grove Vision AI Module V2 over USB
serial (factory SenseCraft firmware speaking the SSCMA AT protocol), decodes the JPEG
frames, draws detected bounding boxes, and shows a live preview (or saves annotated
frames headless for automated testing).

We ONLY talk to the board at runtime. This script NEVER flashes, never writes flash,
never sends xmodem / we2_image_gen / boot-changing commands. The only AT commands used:
    START stream : AT+INVOKE=-1,0,0\\r
    STOP  stream : AT+BREAK\\r
(both runtime-only; AT+BREAK just stops the current invoke loop.)

==================== REVERSE-ENGINEERED PROTOCOL (see PROTOCOL_NOTES.md) ====================
Source: Himax_AI_web_toolkit  assets/index-legacy.8f5c53d6.js (transport/framing)
                              assets/index-legacy.51f14f00.js (box drawing).

* BAUD: 921600              [JS-CONFIRMED: serial.open({baudRate:921600})]
* START: "AT+INVOKE=-1,0,0\\r"   [JS-CONFIRMED: invoke(e)=>`AT+INVOKE=${e},0,0\\r`, called invoke(-1)]
      AT+INVOKE=<n_times>,<diff>,<result_only>
        n_times=-1 -> forever, diff=0 -> every frame, result_only=0 -> include image.
* STOP: "AT+BREAK\\r"        [JS-CONFIRMED: break()=>"AT+BREAK\\r"]
* FRAMING: responses are JSON wrapped as  \\r{ ... }\\n .  A packet starts at a '{' (0x7B)
      immediately preceded by '\\r' (0x0D), and ends at the '}' (0x7D) immediately followed
      by '\\n' (0x0A). Bytes accumulate across reads (packets may be chunked). Non-matching
      boot text / AT echoes are ignored by the state machine.   [JS-CONFIRMED]
* SCHEMA: {"type":int,"name":str,"code":int,"data":{...}}
      type==1 & name=="INVOKE" -> a live inference event.
      data.image  : base64 JPEG (web does W.src="data:image/jpg;base64,"+image)  [JS-CONFIRMED]
      data.boxes  : list of [x, y, w, h, score, target]                          [JS-CONFIRMED]
                    x,y = TOP-LEFT corner; w,h = width,height
                    (web uses canvas strokeRect(x,y,w,h) -> first two args are top-left).
                    target = class id; label = model.classes[target] if available.
      data.resolution / data.perf : optional SSCMA fields (perf=[prep,infer,post] ms).
* SCALING: web upscales small (<640x480) frames by 3x and multiplies box coords by 3 to match.
      We draw on the decoded JPEG at its NATIVE size, so boxes map 1:1 (no scaling). If we
      upscale for viewing we apply the same factor to the boxes.
============================================================================================
"""

import argparse
import base64
import binascii
import json
import logging
import os
import sys
import time

LOG_PATH = "/home/orangepi/himax_py_preview/himax_preview.log"

# ---- Reverse-engineered protocol constants (edit here if firmware differs) ----
DEFAULT_BAUD = 921600
INVOKE_CMD = b"AT+INVOKE=-1,0,0\r"  # continuous inference-with-image stream (model output)
SAMPLE_CMD = b"AT+SAMPLE=-1\r"      # continuous RAW camera stream, no model (pure preview)
STOP_CMD = b"AT+BREAK\r"            # stop the stream loop (runtime-only, no flash write)
DEFAULT_PORT = "/dev/ttyACM0"      # HARD CONSTRAINT: always /dev/ttyACM0

# Camera sensor resolution options (from AT+SENSORS? on this board). Set via
# AT+SENSOR=<id>,<enable>,<opt_id>\r . Confirmed live: 1,1,2 -> 640x480.
SENSOR_ID = 1
RES_OPT = {240: 0, 480: 1, 640: 2}  # pixels -> opt_id ; 640 means 640x480, max for this sensor
# NOTE: --res changes the CAMERA. In --sample mode you get that resolution directly. In
# INVOKE (model) mode the returned image is the model's input size regardless of sensor.

# Framing byte codes
B_LF = 0x0A   # '\n'
B_CR = 0x0D   # '\r'
B_OBRACE = 0x7B  # '{'
B_CBRACE = 0x7D  # '}'

# BGR color palette for class ids (mirrors web's per-class coloring intent).
PALETTE = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255),
    (0, 255, 128), (255, 128, 0),
]

log = logging.getLogger("himax")


def setup_logging(level_name):
    level = getattr(logging, level_name.upper(), logging.INFO)
    log.setLevel(level)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    try:
        fh = logging.FileHandler(LOG_PATH, mode="w")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except Exception:  # pragma: no cover
        log.exception("could not open log file %s (continuing with stderr only)", LOG_PATH)
    log.debug("logging initialized at level %s -> %s", level_name, LOG_PATH)


# ----------------------------------------------------------------------------
# Packet framing state machine — byte-for-byte port of the JS readLoop.
# Emits the raw bytes of each complete JSON packet (without the \r ... \n wrapper).
# ----------------------------------------------------------------------------
class PacketFramer:
    def __init__(self):
        self.cache = bytearray()
        self.has_start = False
        self.last = 0

    def feed(self, chunk):
        """Feed raw bytes; yield complete JSON packets (bytes) as they finish."""
        for b in chunk:
            if b == B_OBRACE:
                if self.last == B_CR:
                    self.has_start = True
                    self.cache = bytearray()
                    self.cache.append(b)
                elif self.has_start:
                    self.cache.append(b)
            elif b == B_LF:
                if self.last == B_CBRACE and self.has_start:
                    self.has_start = False
                    packet = bytes(self.cache)
                    self.cache = bytearray()
                    self.last = b
                    yield packet
                    continue
                elif self.has_start:
                    self.cache.append(b)
            else:
                if self.has_start:
                    self.cache.append(b)
            self.last = b
            # Guard against runaway accumulation on a desynced stream.
            if len(self.cache) > 8 * 1024 * 1024:
                log.warning("framer cache exceeded 8MB without terminator; resetting")
                self.cache = bytearray()
                self.has_start = False


# ----------------------------------------------------------------------------
# Decode + annotate
# ----------------------------------------------------------------------------
def decode_image(b64_str, stats):
    """base64 -> cv2 BGR frame. Returns (frame, raw_jpeg_bytes) or (None, raw)."""
    import cv2
    import numpy as np

    try:
        raw = base64.b64decode(b64_str)
    except (binascii.Error, ValueError):
        log.exception("base64 decode failed (len=%s)", len(b64_str) if b64_str else 0)
        stats["errors"].append("base64_decode")
        return None, None
    try:
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        log.exception("cv2.imdecode threw")
        stats["errors"].append("imdecode_exception")
        return None, raw
    if frame is None:
        log.error("cv2.imdecode returned None (not a valid JPEG?) raw_len=%s", len(raw))
        stats["errors"].append("imdecode_none")
        return None, raw
    stats["image_decode_ok"] += 1
    return frame, raw


def apply_rotation(frame, deg):
    """Rotate frame by 0/90/180/270 degrees (clockwise). Returns rotated frame."""
    if not deg:
        return frame
    import cv2
    code = {90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(deg)
    if code is None:
        return frame
    return cv2.rotate(frame, code)


def rotate_point(x, y, w, h, deg):
    """Map a point from a WxH image to where it lands after apply_rotation(deg).
    w,h are the ORIGINAL (pre-rotation) frame width/height."""
    if deg == 90:    # ROTATE_90_CLOCKWISE -> new image is h x w
        return (h - 1 - y, x)
    if deg == 180:
        return (w - 1 - x, h - 1 - y)
    if deg == 270:   # ROTATE_90_COUNTERCLOCKWISE
        return (y, w - 1 - x)
    return (x, y)


# 21-point hand skeleton (MediaPipe-style landmark ordering): wrist=0, thumb=1-4,
# index=5-8, middle=9-12, ring=13-16, pinky=17-20.
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4),
              (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16),
              (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def draw_pose(frame, detections, stats, labels=None):
    """Draw hand-pose detections in place. Each detection is
    [ [cx,cy,w,h,score,target], [ [x,y,score,kp_id], ...21 ] ]  (bbox is CENTER-based).
    Text labels are appended to `labels` (text, x, y, color) in native coords so the caller
    can render them upright AFTER rotation."""
    import cv2

    if not detections:
        return frame
    h_img, w_img = frame.shape[:2]
    for idx, det in enumerate(detections):
        try:
            if not isinstance(det, (list, tuple)) or len(det) < 2:
                log.debug("skip malformed pose det[%d]: %r", idx, det)
                continue
            box, pts = det[0], det[1]
            cx, cy, bw, bh, score = box[0], box[1], box[2], box[3], box[4]
            x1 = int(round(cx - bw / 2.0)); y1 = int(round(cy - bh / 2.0))
            x2 = int(round(cx + bw / 2.0)); y2 = int(round(cy + bh / 2.0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if labels is not None:
                labels.append(("hand %s" % score, x1, y1, (0, 255, 0)))
            coords = []
            for p in pts:
                px, py = int(round(p[0])), int(round(p[1]))
                coords.append((px, py))
            # skeleton lines first, then joints on top
            for a, b in HAND_EDGES:
                if a < len(coords) and b < len(coords):
                    cv2.line(frame, coords[a], coords[b], (255, 255, 0), 1, cv2.LINE_AA)
            for (px, py) in coords:
                cv2.circle(frame, (px, py), 3, (0, 0, 255), -1, cv2.LINE_AA)
            stats["boxes_total"] += 1
        except Exception:
            log.exception("pose parse/draw failed on det[%d]=%r", idx, det)
            stats["errors"].append("pose_draw")
    return frame


def draw_boxes(frame, boxes, classes, stats, labels=None):
    """Draw [x,y,w,h,score,target] boxes (x,y top-left) onto frame in place. Text labels are
    appended to `labels` (text, x, y, color) so they can be rendered upright after rotation."""
    import cv2

    if not boxes:
        return frame
    h_img, w_img = frame.shape[:2]
    for idx, box in enumerate(boxes):
        try:
            if not isinstance(box, (list, tuple)) or len(box) < 6:
                log.debug("skip malformed box[%d]: %r", idx, box)
                continue
            x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            score = box[4]
            target = int(box[5])
            x1, y1 = int(round(x)), int(round(y))
            x2, y2 = int(round(x + w)), int(round(y + h))
            # clamp to frame
            x1 = max(0, min(x1, w_img - 1)); x2 = max(0, min(x2, w_img - 1))
            y1 = max(0, min(y1, h_img - 1)); y2 = max(0, min(y2, h_img - 1))
            color = PALETTE[target % len(PALETTE)]
            if classes and 0 <= target < len(classes):
                label = "%s: %s" % (classes[target], score)
            else:
                label = "%d: %s" % (target, score)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if labels is not None:
                labels.append((label, x1, y1, color))
            stats["boxes_total"] += 1
        except Exception:
            log.exception("box parse/draw failed on box[%d]=%r", idx, box)
            stats["errors"].append("box_draw")
    return frame


# ----------------------------------------------------------------------------
# Serial helpers
# ----------------------------------------------------------------------------
def open_serial(port, baud):
    import serial
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=0.2)
        log.info("opened serial %s @ %d", port, baud)
        return ser
    except serial.SerialException:
        log.exception("FAILED to open serial port %s @ %d "
                      "(busy? permission? not present?)", port, baud)
        raise
    except Exception:
        log.exception("unexpected error opening serial %s", port)
        raise


def send_cmd(ser, data, label):
    try:
        n = ser.write(data)
        ser.flush()
        log.info("sent %s: %r (%d bytes)", label, data, n)
    except Exception:
        log.exception("FAILED writing %s command %r", label, data)
        raise


def set_sensor_resolution(ser, res):
    """Switch the camera sensor resolution at runtime (no flash write)."""
    opt = RES_OPT.get(res)
    if opt is None:
        log.warning("unknown --res %s, leaving sensor unchanged", res)
        return
    cmd = b"AT+SENSOR=%d,1,%d\r" % (SENSOR_ID, opt)
    try:
        send_cmd(ser, cmd, "SENSOR(res=%d)" % res)
        time.sleep(0.5)
        reply = ser.read(8192).decode("utf-8", "replace")
        if "wrong arg" in reply.lower() or '"code": 5' in reply:
            log.error("sensor did not accept res=%d; reply=%r", res, reply[:200])
        else:
            log.info("sensor set to %dpx (opt_id=%d)", res, opt)
        ser.reset_input_buffer()
    except Exception:
        log.exception("failed setting sensor resolution res=%d (continuing)", res)


def start_stream(ser, args):
    # Stop any prior stream loop and drain first.
    try:
        send_cmd(ser, STOP_CMD, "STOP(pre-clean)")
        time.sleep(0.2)
        ser.reset_input_buffer()
        log.debug("input buffer cleared before start")
    except Exception:
        log.exception("pre-clean before start failed (non-fatal, continuing)")
    # Switch camera resolution AFTER stopping, immediately before starting the stream
    # (the resolution must be the last thing set before SAMPLE/INVOKE or it won't apply).
    if args.res:
        set_sensor_resolution(ser, args.res)
    start_cmd = SAMPLE_CMD if args.sample else INVOKE_CMD
    send_cmd(ser, start_cmd, "START(%s)" % ("SAMPLE" if args.sample else "INVOKE"))
    # Confidence threshold (tscore) lives in the ALGORITHM config, which only exists once
    # INVOKE is running — so it MUST be set AFTER the stream starts. The board default (83)
    # is very high and filters out most detections; lower it (e.g. 40) to see results.
    if not args.sample and args.tscore is not None:
        time.sleep(0.8)
        try:
            send_cmd(ser, b"AT+TSCORE=%d\r" % args.tscore, "TSCORE(%d)" % args.tscore)
            ser.reset_input_buffer()
        except Exception:
            log.exception("failed setting tscore=%d (continuing)", args.tscore)


def stop_stream(ser):
    try:
        send_cmd(ser, STOP_CMD, "STOP")
    except Exception:
        log.exception("failed sending STOP on shutdown (ignored)")


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def run(args):
    stats = {
        "packets_seen": 0,
        "json_parse_ok": 0,
        "image_decode_ok": 0,
        "boxes_total": 0,
        "errors": [],
        "first_resolution": None,
        "baud": args.baud,
        "elapsed_s": 0.0,
    }
    t0 = time.time()
    classes = []  # we don't query AT+MODEL?; class names usually unavailable -> show ids

    if args.save_dir:
        try:
            os.makedirs(args.save_dir, exist_ok=True)
        except Exception:
            log.exception("could not create save-dir %s", args.save_dir)

    ser = open_serial(args.port, args.baud)
    framer = PacketFramer()
    window_name = "Himax Preview (q to quit)"
    gui_ok = not args.no_window
    saved = 0

    try:
        start_stream(ser, args)
        log.info("entering read loop (stream=%s, mode=%s, res=%s, target_frames=%s, duration=%s)",
                 "SAMPLE" if args.sample else "INVOKE",
                 "headless" if args.no_window else "live",
                 args.res or "default", args.frames, args.duration)

        while True:
            # ---- stop conditions ----
            if args.duration and (time.time() - t0) >= args.duration:
                log.info("duration %ss reached, stopping", args.duration)
                break
            if args.frames and saved >= args.frames:
                log.info("captured target %d frames, stopping", args.frames)
                break

            # ---- raw read ----
            try:
                chunk = ser.read(4096)
            except Exception:
                log.exception("serial read failed")
                stats["errors"].append("serial_read")
                break
            if not chunk:
                # timeout tick
                log.debug("read timeout (no bytes)")
                continue
            if args.list_raw:
                try:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                except Exception:
                    log.exception("list-raw dump failed")
                continue

            # ---- frame extraction ----
            for packet in framer.feed(chunk):
                stats["packets_seen"] += 1
                try:
                    obj = json.loads(packet.decode("utf-8", "replace"))
                    stats["json_parse_ok"] += 1
                except Exception as e:
                    # Occasional truncated/partial packet (typically the first frame right
                    # after AT+INVOKE, when the board was mid-transmission). Streaming noise:
                    # drop this frame quietly and keep going — NOT a fatal error.
                    stats["errors"].append("json_parse")
                    log.warning("dropped 1 malformed packet (len=%d, %s); continuing",
                                len(packet), e.__class__.__name__)
                    continue

                ptype = obj.get("type")
                pname = obj.get("name")
                raw_data = obj.get("data")
                data = raw_data if isinstance(raw_data, dict) else {}
                log.debug("packet type=%s name=%s code=%s data=%s",
                          ptype, pname, obj.get("code"),
                          list(data.keys()) if data else raw_data)

                # We care about the INVOKE inference event (type==1). Be lenient: accept
                # any packet that carries an image.
                if "resolution" in data and stats["first_resolution"] is None:
                    stats["first_resolution"] = data.get("resolution")

                img_b64 = data.get("image")
                if not img_b64:
                    # Could be a command reply / non-image event; nothing to draw.
                    continue

                frame, raw_jpeg = decode_image(img_b64, stats)
                if frame is None:
                    continue
                if stats["first_resolution"] is None:
                    stats["first_resolution"] = [frame.shape[1], frame.shape[0]]

                # Draw shapes on the NATIVE (un-rotated) frame so the inference coords line up,
                # collecting text labels separately, THEN rotate the frame and draw the labels
                # upright at their rotation-mapped positions.
                n_det = 0
                labels = []
                ow, oh = frame.shape[1], frame.shape[0]  # native dims before rotation
                if args.draw_inference:
                    boxes = data.get("boxes") or []
                    draw_boxes(frame, boxes, classes, stats, labels)
                    n_det += len(boxes)
                if args.draw_pose:
                    kps = data.get("keypoints") or []
                    draw_pose(frame, kps, stats, labels)
                    n_det += len(kps)

                frame = apply_rotation(frame, args.rotate)

                if labels:
                    import cv2
                    for text, lx, ly, color in labels:
                        rx, ry = rotate_point(lx, ly, ow, oh, args.rotate)
                        cv2.putText(frame, text, (int(rx), max(12, int(ry) - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # ---- output ----
                if args.no_window:
                    try:
                        import cv2
                        fpng = os.path.join(args.save_dir, "frame_%03d.png" % saved)
                        cv2.imwrite(fpng, frame)
                        if raw_jpeg is not None:
                            with open(os.path.join(args.save_dir,
                                                   "frame_%03d.jpg" % saved), "wb") as f:
                                f.write(raw_jpeg)
                        log.info("saved %s (detections=%d, size=%dx%d)",
                                 fpng, n_det, frame.shape[1], frame.shape[0])
                    except Exception:
                        log.exception("failed saving frame %d", saved)
                        stats["errors"].append("save_frame")
                    saved += 1
                    if args.frames and saved >= args.frames:
                        break
                else:
                    if gui_ok:
                        try:
                            import cv2
                            cv2.imshow(window_name, frame)
                            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                                log.info("q pressed, quitting")
                                raise KeyboardInterrupt
                        except KeyboardInterrupt:
                            raise
                        except Exception:
                            log.exception("cv2 GUI call failed — no display? "
                                          "Re-run with --no-window --save-dir DIR.")
                            gui_ok = False
                    saved += 1  # count displayed frames too

    except KeyboardInterrupt:
        log.info("interrupted by user")
    except Exception:
        log.exception("fatal error in run loop")
        stats["errors"].append("fatal")
    finally:
        stats["elapsed_s"] = round(time.time() - t0, 3)
        stop_stream(ser)
        try:
            ser.close()
            log.info("serial port closed")
        except Exception:
            log.exception("error closing serial port")
        if not args.no_window:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
        # summary
        log.info("SUMMARY: packets=%d json_ok=%d img_ok=%d boxes=%d errors=%d res=%s elapsed=%ss",
                 stats["packets_seen"], stats["json_parse_ok"], stats["image_decode_ok"],
                 stats["boxes_total"], len(stats["errors"]), stats["first_resolution"],
                 stats["elapsed_s"])
        if args.save_dir:
            try:
                with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
                    json.dump(stats, f, indent=2)
                log.info("wrote %s", os.path.join(args.save_dir, "summary.json"))
            except Exception:
                log.exception("could not write summary.json")
    return stats


def build_parser():
    p = argparse.ArgumentParser(
        description="Local live preview for Grove Vision AI Module V2 over USB serial "
                    "(SSCMA/SenseCraft AT protocol). Runtime-only; never flashes.")
    p.add_argument("--port", default=DEFAULT_PORT, help="serial port (default %(default)s)")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD,
                   help="baud rate (default %(default)s, reverse-engineered from toolkit)")
    p.add_argument("--frames", type=int, default=0,
                   help="capture N frames then stop (0=unlimited). Headless test uses this.")
    p.add_argument("--duration", type=float, default=0.0,
                   help="stop after S seconds (0=unlimited)")
    p.add_argument("--no-window", action="store_true",
                   help="headless: no GUI, save annotated PNGs + raw jpeg + summary.json")
    p.add_argument("--save-dir", default="",
                   help="directory to write frame_NNN.png / .jpg / summary.json")
    p.add_argument("--res", type=int, default=None, choices=[240, 480, 640],
                   help="camera resolution: 240(=240x240) 480(=480x480) 640(=640x480, max). "
                        "Most useful with --sample; in model mode the image stays model-size.")
    p.add_argument("--sample", action="store_true",
                   help="pure camera preview (AT+SAMPLE, no model/inference) — full sensor "
                        "resolution, no boxes/keypoints")
    p.add_argument("--tscore", type=int, default=None, metavar="0-100",
                   help="confidence threshold for inference (INVOKE mode). Board default 83 "
                        "is very high and hides most detections; try 40. Set after stream starts.")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help="rotate preview clockwise (use 180 if the image is upside-down)")
    p.add_argument("--draw-inference", action="store_true",
                   help="overlay AI detection boxes (data.boxes; for object-detection models)")
    p.add_argument("--draw-pose", action="store_true",
                   help="overlay hand-pose detections (data.keypoints: bbox + 21 landmarks + "
                        "skeleton). Pair with --tscore 40 so detections actually appear.")
    p.add_argument("--list-raw", action="store_true",
                   help="dump raw serial bytes to stdout for debugging (no decode)")
    p.add_argument("--log-level", default="INFO",
                   help="DEBUG/INFO/WARNING/ERROR (default INFO)")
    return p


def main():
    args = build_parser().parse_args()
    setup_logging(args.log_level)
    log.info("himax_preview start: port=%s baud=%d frames=%s no_window=%s save_dir=%s",
             args.port, args.baud, args.frames, args.no_window, args.save_dir)
    if args.no_window and not args.save_dir:
        log.error("--no-window requires --save-dir DIR to write output")
        sys.exit(2)
    if args.frames and not args.no_window:
        log.info("--frames given without --no-window: will run live and count frames")
    try:
        run(args)
    except Exception:
        log.exception("unrecoverable error")
        sys.exit(1)


if __name__ == "__main__":
    main()
