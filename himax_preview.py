#!/usr/bin/env python3
"""himax_preview.py — example CLI built on top of the `himax_vision` module.

This is just a thin command-line wrapper that wires argparse flags to HimaxVision and either
shows a live window or saves annotated frames headless. The actual logic (serial, SSCMA
protocol, decode, inference parsing, drawing) lives in `himax_vision.py` — import that to use
the board from your own project. Runtime-only; never flashes.

Examples:
    # live preview with gesture/RPS detection
    python3 himax_preview.py --res 640 --draw-inference --tscore 60 --rotate 180
    # hand-pose skeleton
    python3 himax_preview.py --draw-pose --tscore 40 --rotate 180
    # pure camera, no inference
    python3 himax_preview.py --sample --res 640 --rotate 180
    # headless capture (no display) -> annotated PNGs + summary.json
    python3 himax_preview.py --draw-inference --tscore 60 --frames 30 \\
            --no-window --save-dir captures --rotate 180
"""

import argparse
import json
import logging
import os
import sys

from himax_vision import HimaxVision, DEFAULT_PORT, DEFAULT_BAUD

LOG_PATH = "/home/orangepi/himax_py_preview/himax_preview.log"


def build_parser():
    p = argparse.ArgumentParser(
        description="Live preview / capture for the Grove Vision AI Module V2 "
                    "(thin CLI over the himax_vision module). Runtime-only; never flashes.")
    p.add_argument("--port", default=DEFAULT_PORT, help="serial port (default %(default)s)")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="baud (default %(default)s)")
    p.add_argument("--res", type=int, default=None, choices=[240, 480, 640],
                   help="camera resolution (640 = 640x480, the max)")
    p.add_argument("--sample", action="store_true",
                   help="pure camera stream (AT+SAMPLE, no inference)")
    p.add_argument("--tscore", type=int, default=None, metavar="0-100",
                   help="confidence threshold (board default 83 hides most; try 40-60)")
    p.add_argument("--draw-inference", action="store_true",
                   help="overlay detection boxes (class names auto-fetched from the model)")
    p.add_argument("--draw-pose", action="store_true",
                   help="overlay hand-pose box + keypoints + skeleton")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help="rotate preview clockwise (use 180 if upside-down)")
    p.add_argument("--at-rotate", type=int, default=None, choices=[0, 1, 2, 3],
                   help="send AT+ROTATE=N before streaming - firmware-side, "
                        "rotates only the AI's input (0/1/2/3 = 0/90/180/270deg); "
                        "the preview image itself is unaffected (use --rotate for that). "
                        "Custom sscma_cam_mic command, not stock SSCMA.")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (0=unlimited)")
    p.add_argument("--duration", type=float, default=0.0, help="stop after S seconds")
    p.add_argument("--no-window", action="store_true",
                   help="headless: save annotated PNGs + raw jpeg + summary.json")
    p.add_argument("--save-dir", default="", help="output dir for --no-window")
    p.add_argument("--list-raw", action="store_true",
                   help="dump raw serial bytes to stdout (debug; bypasses the module)")
    p.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    return p


def list_raw(args):
    """Minimal raw-serial dump for debugging (doesn't use the module)."""
    import serial
    import time
    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    ser.write(b"AT+BREAK\r"); ser.flush(); time.sleep(0.2); ser.reset_input_buffer()
    ser.write(b"AT+SAMPLE=-1\r" if args.sample else b"AT+INVOKE=-1,0,0\r"); ser.flush()
    try:
        while True:
            chunk = ser.read(4096)
            if chunk:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
    except KeyboardInterrupt:
        pass
    finally:
        ser.write(b"AT+BREAK\r"); ser.flush(); ser.close()


def main():
    args = build_parser().parse_args()
    if args.no_window and not args.save_dir:
        print("--no-window requires --save-dir DIR", file=sys.stderr)
        sys.exit(2)
    if args.list_raw:
        list_raw(args)
        return

    cam = HimaxVision(
        port=args.port,
        baud=args.baud,
        resolution=args.res,
        sample=args.sample,
        tscore=args.tscore,
        rotate=args.rotate,
        show_preview=not args.no_window,
        collect_inference=True,
        annotate=(args.draw_inference or args.draw_pose),
        return_image=bool(args.save_dir),          # need the image to save it headless
        save_log=True,
        log_path=LOG_PATH,
        pre_commands=([b"AT+ROTATE=%d\r" % args.at_rotate] if args.at_rotate is not None else None),
    )
    # honor --log-level on the module's logger (also echo to stderr)
    lvl = getattr(logging, args.log_level.upper(), logging.INFO)
    mlog = logging.getLogger("himax_vision")
    mlog.setLevel(lvl)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    mlog.addHandler(sh)

    max_frames = args.frames or None
    duration = args.duration or None

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    try:
        with cam:
            print("model:", cam.model)
            saved = 0
            for frame in cam.stream(max_frames=max_frames, duration=duration):
                if args.save_dir and frame.image is not None:
                    import cv2
                    cv2.imwrite(os.path.join(args.save_dir, "frame_%03d.png" % saved), frame.image)
                    if frame.jpeg is not None:
                        with open(os.path.join(args.save_dir, "frame_%03d.jpg" % saved), "wb") as f:
                            f.write(frame.jpeg)
                    saved += 1
    except KeyboardInterrupt:
        pass

    print("SUMMARY:", cam.stats)
    if args.save_dir:
        summary = dict(cam.stats, baud=args.baud, resolution=args.res or "default")
        with open(os.path.join(args.save_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
