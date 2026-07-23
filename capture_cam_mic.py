#!/usr/bin/env python3
"""
Capture JPEG frames + PCM audio from sscma_cam_mic over the AT-command UART,
saving frames as frame_00001.jpg... and audio as one audio.wav.

Usage:
    python3 capture_cam_mic.py --port /dev/ttyACM0 --resolution 1 --rate 48000 --duration 10 --outdir ./capture

    # camera running (inference only, no preview images saved) + audio:
    python3 capture_cam_mic.py --no-preview --resolution 1 --rate 48000 --duration 10 --outdir ./capture

    # audio only, camera left idle:
    python3 capture_cam_mic.py --no-camera --rate 48000 --duration 10 --outdir ./capture

--resolution: 2=640x480, 1=320x240, 0=160x112  (matches AT+SENSOR=1,1,<opt>)
--rate: 8000 / 16000 / 32000 / 48000            (matches AT+ASR=<rate>)
--duration: seconds to record before sending AT+BREAK and exiting
--no-preview: still runs the camera + NPU inference (AT+INVOKE result_only=1),
              but no image is streamed back, so no frame_*.jpg files are saved
--no-camera: skip AT+SENSOR/AT+INVOKE entirely, audio-only capture
"""
import argparse
import base64
import json
import os
import re
import sys
import time
import wave

import serial

PACKET_RE = re.compile(rb'\{"type": 1, "name": "\w+".*?\}\}')


def send(ser, cmd: str):
    ser.write((cmd + "\r").encode())
    ser.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--resolution", type=int, choices=[0, 1, 2], default=1,
                     help="0=160x112 1=320x240 2=640x480")
    ap.add_argument("--rate", type=int, choices=[8000, 16000, 32000, 48000], default=48000)
    ap.add_argument("--duration", type=float, default=10.0, help="seconds to capture")
    ap.add_argument("--outdir", default="./capture")
    ap.add_argument("--no-preview", action="store_true",
                     help="run camera+inference but don't stream/save images (AT+INVOKE result_only=1)")
    ap.add_argument("--no-camera", action="store_true",
                     help="skip the camera entirely, audio-only")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    frame_dir = args.outdir

    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    time.sleep(2.5)  # let the board settle after opening the port
    ser.reset_input_buffer()

    wav_path = os.path.join(args.outdir, "audio.wav")
    wf = wave.open(wav_path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)  # int16
    wf.setframerate(args.rate)

    frame_count = 0
    result_count = 0
    audio_bytes = 0
    buf = b""

    def handle_packet(raw: bytes):
        nonlocal frame_count, result_count, audio_bytes
        try:
            obj = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            return
        d = obj.get("data", {})
        if not isinstance(d, dict):
            return
        if "image" in d:
            frame_count += 1
            path = os.path.join(frame_dir, f"frame_{frame_count:05d}.jpg")
            with open(path, "wb") as f:
                f.write(base64.b64decode(d["image"]))
        elif obj.get("name") in ("INVOKE", "SAMPLE"):
            result_count += 1  # result-only packet, no image field
        if "audio" in d:
            pcm = base64.b64decode(d["audio"])
            wf.writeframes(pcm)
            audio_bytes += len(pcm)

    try:
        send(ser, "AT+BREAK"); time.sleep(0.2)

        if not args.no_camera:
            print(f"[setup] camera resolution opt={args.resolution}"
                  + (" (no preview - inference only)" if args.no_preview else ""))
            send(ser, f"AT+SENSOR=1,1,{args.resolution}"); time.sleep(0.3)
            result_only = 1 if args.no_preview else 0
            send(ser, f"AT+INVOKE=-1,0,{result_only}"); time.sleep(0.2)
        else:
            print("[setup] camera skipped (--no-camera)")

        print(f"[setup] audio rate={args.rate} Hz")
        send(ser, f"AT+ASR={args.rate}"); time.sleep(0.3)
        send(ser, "AT+ASAMPLE=-1")

        print(f"[capture] recording for {args.duration:.1f}s ... (Ctrl+C to stop early)")
        t_end = time.time() + args.duration
        while time.time() < t_end:
            chunk = ser.read(4096)
            if chunk:
                buf += chunk
                while True:
                    m = PACKET_RE.search(buf)
                    if not m:
                        break
                    handle_packet(m.group(0))
                    buf = buf[m.end():]
            print(f"\r[capture] frames={frame_count} results={result_count} "
                  f"audio={audio_bytes/1024:.1f} KiB", end="", flush=True)
        print()
    except KeyboardInterrupt:
        print("\n[capture] stopped early by user")
    finally:
        send(ser, "AT+BREAK")
        time.sleep(0.3)
        wf.close()
        ser.close()

    if frame_count:
        print(f"[done] {frame_count} frames -> {frame_dir}/frame_*.jpg")
    if result_count:
        print(f"[done] {result_count} inference results received (no images saved, --no-preview)")
    print(f"[done] audio -> {wav_path} ({audio_bytes/1024:.1f} KiB raw PCM)")


if __name__ == "__main__":
    sys.exit(main())
