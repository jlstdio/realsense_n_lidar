#!/usr/bin/env python3
# coding=utf-8
"""Receive servo commands from Jetson and drive the robot arm.

Default protocol (UDP):
- JSON: {"idx": 0, "angle": 90, "duration": 1000}
- CSV:  0,90,1000

idx is 0-5 and is mapped to servo id 1-6.
angle is 0-180 degrees.
duration is the movement time in milliseconds.
"""

import argparse
import json
import os
import socket
import sys
import time
from typing import Tuple

# Ensure local Arm_Lib package is importable when script is run from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Arm_Lib import Arm_Device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot arm UDP receiver for Jetson commands")
    parser.add_argument("--bind", default="0.0.0.0", help="Local bind address")
    parser.add_argument("--port", type=int, default=5005, help="UDP listen port")
    parser.add_argument(
        "--allowed-ip",
        default="192.168.10.1",
        help="Only accept commands from this sender IP",
    )
    return parser.parse_args()


def parse_payload(raw: bytes) -> Tuple[int, int, int]:
    text = raw.decode("utf-8", errors="strict").strip()
    if not text:
        raise ValueError("empty payload")

    if text.startswith("{"):
        msg = json.loads(text)
        idx = int(msg["idx"])
        angle = int(msg["angle"])
        duration = int(msg["duration"])
        return idx, angle, duration

    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError("payload must be JSON or csv: idx,angle,duration")

    idx = int(parts[0])
    angle = int(parts[1])
    duration = int(parts[2])
    return idx, angle, duration


def validate_command(idx: int, angle: int, duration: int) -> None:
    if not 0 <= idx <= 5:
        raise ValueError("idx must be in range 0..5")
    if not 0 <= angle <= 180:
        raise ValueError("angle must be in range 0..180")
    if duration < 0:
        raise ValueError("duration must be >= 0")


def main() -> None:
    args = parse_args()

    arm = Arm_Device()
    time.sleep(0.1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))

    print(f"[INFO] Listening UDP on {args.bind}:{args.port}")
    print(f"[INFO] Allowed sender IP: {args.allowed_ip}")

    try:
        while True:
            data, addr = sock.recvfrom(2048)
            sender_ip, sender_port = addr

            if sender_ip != args.allowed_ip:
                print(f"[WARN] Ignored packet from unauthorized IP: {sender_ip}:{sender_port}")
                continue

            try:
                idx, angle, duration = parse_payload(data)
                validate_command(idx, angle, duration)

                servo_id = idx + 1  # User idx 0-5 -> library servo id 1-6
                arm.Arm_serial_servo_write(servo_id, angle, duration)

                print(
                    f"[OK] {sender_ip}:{sender_port} idx={idx} servo_id={servo_id} "
                    f"angle={angle} duration={duration}"
                )
            except Exception as exc:
                print(f"[ERROR] Failed to handle packet from {sender_ip}:{sender_port}: {exc}")
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    finally:
        sock.close()
        del arm


if __name__ == "__main__":
    main()
