#!/usr/bin/env python3
# coding=utf-8
"""Send robot-arm commands to UDP receiver.

Supported payload formats (same as receiver):
- JSON: {"idx": 0, "angle": 90, "duration": 1000}
- CSV:  0,90,1000

idx: 0-5
angle: 0-180
duration: milliseconds (>= 0)
"""

import argparse
import json
import socket
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson UDP sender for robot arm")
    parser.add_argument("--target-ip", required=True, help="Receiver host IP")
    parser.add_argument("--port", type=int, default=5005, help="Receiver UDP port")

    parser.add_argument("--idx", type=int, required=True, help="Servo index 0..5")
    parser.add_argument("--angle", type=int, required=True, help="Servo angle 0..180")
    parser.add_argument("--duration", type=int, required=True, help="Move duration (ms)")

    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Payload format",
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Send continuously until Ctrl+C",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Loop interval in seconds (used with --loop)",
    )

    return parser.parse_args()


def validate(idx: int, angle: int, duration: int) -> None:
    if not 0 <= idx <= 5:
        raise ValueError("idx must be in range 0..5")
    if not 0 <= angle <= 180:
        raise ValueError("angle must be in range 0..180")
    if duration < 0:
        raise ValueError("duration must be >= 0")


def build_payload(idx: int, angle: int, duration: int, fmt: str) -> bytes:
    if fmt == "json":
        msg = {"idx": idx, "angle": angle, "duration": duration}
        return json.dumps(msg, separators=(",", ":")).encode("utf-8")

    return f"{idx},{angle},{duration}".encode("utf-8")


def main() -> None:
    args = parse_args()
    validate(args.idx, args.angle, args.duration)

    payload = build_payload(args.idx, args.angle, args.duration, args.format)
    target = (args.target_ip, args.port)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        if args.loop:
            print(
                f"[INFO] Sending to {args.target_ip}:{args.port} every {args.interval}s "
                f"payload={payload.decode('utf-8')}"
            )
            while True:
                sock.sendto(payload, target)
                time.sleep(args.interval)
        else:
            sock.sendto(payload, target)
            print(f"[OK] Sent to {args.target_ip}:{args.port} payload={payload.decode('utf-8')}")
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
