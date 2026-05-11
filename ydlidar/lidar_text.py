from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import serial
from PIL import Image, ImageDraw

START_SCAN_COMMAND = b"\xA5\x60"
STOP_SCAN_COMMAND = b"\xA5\x65"
PACKET_HEADER = b"\xAA\x55"
DEFAULT_BAUDRATE = 230400
DEFAULT_DURATION_SECONDS = 10.0


def load_pyplot(backend: str):
    matplotlib.use(backend, force=True)
    import matplotlib.pyplot as plt

    return plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture YDLIDAR G2 raw data for 10 seconds.")
    parser.add_argument("--port", default=detect_default_port(), help="Serial port path. Defaults to detected device.")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="Serial baudrate for YDLIDAR G2.")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS, help="Capture duration in seconds.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "output"),
        help="Directory where JSON, PNG, and GIF outputs will be written.",
    )
    parser.add_argument("--live", action="store_true", help="Open a real-time viewer instead of saving capture files.")
    parser.add_argument(
        "--live-history-packets",
        type=int,
        default=12,
        help="How many recent packets to keep visible in live mode.",
    )
    parser.add_argument(
        "--live-range-mm",
        type=float,
        default=4000.0,
        help="Half-width and half-height of the live view in millimeters.",
    )
    parser.add_argument(
        "--revolutions-per-frame",
        type=int,
        default=3,
        help="Number of completed revolutions to merge into one GIF frame/update.",
    )
    return parser


def detect_default_port() -> str:
    by_id_dir = Path("/dev/serial/by-id")
    if by_id_dir.exists():
        for candidate in sorted(by_id_dir.iterdir()):
            name = candidate.name.lower()
            if any(token in name for token in ("ydlidar", "cp210", "silicon", "uart")):
                return str(candidate.resolve())

    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        matches = sorted(Path("/dev").glob(pattern.split("/dev/")[1]))
        if matches:
            return str(matches[0])

    return "/dev/ttyUSB0"


def checksum_is_valid(packet: bytes) -> bool:
    if len(packet) < 10:
        return False

    lsn = packet[3]
    if len(packet) != 10 + (3 * lsn):
        return False

    checksum = packet[8] | (packet[9] << 8)
    running = (packet[0] | (packet[1] << 8)) ^ (packet[2] | (packet[3] << 8))
    running ^= packet[4] | (packet[5] << 8)
    running ^= packet[6] | (packet[7] << 8)

    for point_index in range(lsn):
        offset = 10 + (point_index * 3)
        running ^= packet[offset]
        running ^= packet[offset + 1] | (packet[offset + 2] << 8)

    return running == checksum


def first_level_angle(lsb: int, msb: int) -> float:
    return ((lsb | (msb << 8)) >> 1) / 64.0


def diff_angle(end_angle: float, start_angle: float) -> float:
    if end_angle < start_angle:
        end_angle += 360.0
    return end_angle - start_angle


def interpolated_angle(index: int, angle_delta: float, sample_count: int, start_angle: float) -> float:
    if sample_count <= 1:
        return start_angle

    angle = (angle_delta / (sample_count - 1)) * index + start_angle
    if angle >= 360.0:
        angle -= 360.0
    return angle


def corrected_angle(angle_deg: float, distance_mm: int) -> float:
    if distance_mm == 0:
        return 0.0

    angle = angle_deg + math.degrees(math.atan2(21.8 * (155.3 - distance_mm), 155.3 * distance_mm))
    if angle < 0.0:
        angle += 360.0
    return angle


def extract_packets(buffer: bytearray) -> tuple[list[bytes], bytearray]:
    packets: list[bytes] = []

    while True:
        header_index = buffer.find(PACKET_HEADER)
        if header_index < 0:
            return packets, bytearray(buffer[-1:])

        if header_index > 0:
            del buffer[:header_index]

        if len(buffer) < 10:
            return packets, buffer

        sample_count = buffer[3]
        packet_length = 10 + (3 * sample_count)

        if len(buffer) < packet_length:
            return packets, buffer

        packets.append(bytes(buffer[:packet_length]))
        del buffer[:packet_length]


def parse_packet(packet: bytes, packet_index: int, elapsed_seconds: float) -> dict[str, Any]:
    sample_count = packet[3]
    start_angle = first_level_angle(packet[4], packet[5])
    end_angle = first_level_angle(packet[6], packet[7])
    angle_delta = diff_angle(end_angle, start_angle)
    points: list[dict[str, Any]] = []

    for sample_index in range(sample_count):
        offset = 10 + (sample_index * 3)
        intensity = packet[offset] + ((packet[offset + 1] & 0b11) << 8)
        distance_mm = (packet[offset + 1] >> 2) | (packet[offset + 2] << 6)

        if sample_index == 0:
            base_angle = start_angle
        elif sample_index == sample_count - 1:
            base_angle = end_angle
        else:
            base_angle = interpolated_angle(sample_index, angle_delta, sample_count, start_angle)

        point_angle = corrected_angle(base_angle, distance_mm)
        x_mm = distance_mm * math.cos(math.radians(point_angle))
        y_mm = -distance_mm * math.sin(math.radians(point_angle))

        points.append(
            {
                "sample_index": sample_index,
                "angle_deg": round(point_angle, 4),
                "distance_mm": distance_mm,
                "intensity": intensity,
                "x_mm": round(x_mm, 4),
                "y_mm": round(y_mm, 4),
            }
        )

    return {
        "packet_index": packet_index,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "raw_hex": packet.hex(),
        "frequency_hz": round((packet[2] >> 1) / 10.0, 2),
        "packet_type": packet[2] & 0b1,
        "sample_count": sample_count,
        "start_angle_deg": round(start_angle, 4),
        "end_angle_deg": round(end_angle, 4),
        "checksum_valid": checksum_is_valid(packet),
        "points": points,
    }


def capture_packets(port: str, baudrate: int, duration_seconds: float) -> tuple[str, list[dict[str, Any]], int]:
    packet_records: list[dict[str, Any]] = []
    buffer = bytearray()
    total_bytes_read = 0

    with serial.Serial(port=port, baudrate=baudrate, timeout=0.2) as lidar_serial:
        lidar_serial.reset_input_buffer()
        lidar_serial.reset_output_buffer()
        lidar_serial.write(START_SCAN_COMMAND)
        lidar_serial.flush()

        response_descriptor = lidar_serial.read(7)
        started_at = time.monotonic()
        capture_deadline = started_at + duration_seconds
        stop_capture = False

        try:
            while time.monotonic() < capture_deadline:
                chunk = lidar_serial.read(4096)
                if not chunk:
                    continue

                total_bytes_read += len(chunk)
                buffer.extend(chunk)
                extracted_packets, buffer = extract_packets(buffer)

                for packet in extracted_packets:
                    elapsed_seconds = time.monotonic() - started_at
                    packet_records.append(parse_packet(packet, len(packet_records), elapsed_seconds))
        finally:
            lidar_serial.write(STOP_SCAN_COMMAND)
            lidar_serial.flush()

    return response_descriptor.hex(), packet_records, total_bytes_read


def valid_points_from_packet(packet_record: dict[str, Any]) -> list[dict[str, Any]]:
    if not packet_record["checksum_valid"]:
        return []

    return [point for point in packet_record["points"] if point["distance_mm"] > 0]


def flatten_points(packet_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for packet_record in packet_records:
        if not packet_record["checksum_valid"]:
            continue

        for point in packet_record["points"]:
            if point["distance_mm"] <= 0:
                continue
            points.append(point)
    return points


def chunk_points_for_animation(
    packet_records: list[dict[str, Any]],
    max_frames: int = 180,
    history_packets: int = 12,
) -> list[list[dict[str, Any]]]:
    valid_packets = [packet for packet in packet_records if packet["checksum_valid"]]
    if not valid_packets:
        return []

    packets_per_frame = max(1, math.ceil(len(valid_packets) / max_frames))
    frames: list[list[dict[str, Any]]] = []

    for start_index in range(0, len(valid_packets), packets_per_frame):
        history_start = max(0, start_index - ((history_packets - 1) * packets_per_frame))
        packet_slice = valid_packets[history_start : start_index + packets_per_frame]
        frame_points: list[dict[str, Any]] = []

        for packet in packet_slice:
            for point in packet["points"]:
                if point["distance_mm"] <= 0:
                    continue
                frame_points.append(point)

        frames.append(frame_points)

    return frames


def compute_plot_bounds(points: list[dict[str, Any]], padding_mm: float = 200.0) -> tuple[float, float, float, float]:
    if not points:
        return -1000.0, 1000.0, -1000.0, 1000.0

    x_values = [point["x_mm"] for point in points]
    y_values = [point["y_mm"] for point in points]
    min_x = min(x_values) - padding_mm
    max_x = max(x_values) + padding_mm
    min_y = min(y_values) - padding_mm
    max_y = max(y_values) + padding_mm

    if math.isclose(min_x, max_x):
        min_x -= 500.0
        max_x += 500.0
    if math.isclose(min_y, max_y):
        min_y -= 500.0
        max_y += 500.0

    return min_x, max_x, min_y, max_y


def map_point_to_pixel(
    x_mm: float,
    y_mm: float,
    bounds: tuple[float, float, float, float],
    image_size: int,
    margin_px: int,
) -> tuple[int, int]:
    min_x, max_x, min_y, max_y = bounds
    usable_size = image_size - (margin_px * 2)
    x_ratio = (x_mm - min_x) / (max_x - min_x)
    y_ratio = (y_mm - min_y) / (max_y - min_y)
    x_pixel = margin_px + int(x_ratio * usable_size)
    y_pixel = image_size - margin_px - int(y_ratio * usable_size)
    return x_pixel, y_pixel


def intensity_to_color(intensity: int, max_intensity: int) -> tuple[int, int, int]:
    if max_intensity <= 0:
        return 70, 90, 180

    normalized = max(0.0, min(1.0, intensity / max_intensity))
    red = int(80 + (175 * normalized))
    green = int(70 + (140 * (1.0 - abs(normalized - 0.5) * 2.0)))
    blue = int(220 - (170 * normalized))
    return red, green, blue


def render_animation_frame(
    points: list[dict[str, Any]],
    bounds: tuple[float, float, float, float],
    image_size: int = 800,
    margin_px: int = 50,
    frame_label: str | None = None,
) -> Image.Image:
    image = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(image)
    grid_color = (220, 220, 220)
    axis_color = (50, 50, 50)
    max_intensity = max((point["intensity"] for point in points), default=1)

    for step in range(5):
        offset = margin_px + int(((image_size - (margin_px * 2)) / 4) * step)
        draw.line((offset, margin_px, offset, image_size - margin_px), fill=grid_color, width=1)
        draw.line((margin_px, offset, image_size - margin_px, offset), fill=grid_color, width=1)

    x_axis_start = map_point_to_pixel(bounds[0], 0.0, bounds, image_size, margin_px)
    x_axis_end = map_point_to_pixel(bounds[1], 0.0, bounds, image_size, margin_px)
    y_axis_start = map_point_to_pixel(0.0, bounds[2], bounds, image_size, margin_px)
    y_axis_end = map_point_to_pixel(0.0, bounds[3], bounds, image_size, margin_px)
    draw.line((x_axis_start[0], x_axis_start[1], x_axis_end[0], x_axis_end[1]), fill=axis_color, width=2)
    draw.line((y_axis_start[0], y_axis_start[1], y_axis_end[0], y_axis_end[1]), fill=axis_color, width=2)

    for point in points:
        pixel_x, pixel_y = map_point_to_pixel(point["x_mm"], point["y_mm"], bounds, image_size, margin_px)
        color = intensity_to_color(point["intensity"], max_intensity)
        draw.ellipse((pixel_x - 2, pixel_y - 2, pixel_x + 2, pixel_y + 2), fill=color)

    label = "YDLIDAR G2 live update"
    if frame_label:
        label = f"{label}  {frame_label}"
    draw.text((margin_px, 16), label, fill=(20, 20, 20))
    return image


def save_gif(packet_records: list[dict[str, Any]], gif_path: Path) -> None:
    all_points = flatten_points(packet_records)
    frame_points = chunk_points_for_animation(packet_records)

    if not frame_points:
        frame_points = [[]]

    bounds = compute_plot_bounds(all_points)
    frames = [
        render_animation_frame(points, bounds, frame_label=f"frame {index + 1}/{len(frame_points)}")
        for index, points in enumerate(frame_points)
    ]
    frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=40,
        loop=0,
        optimize=False,
        disposal=2,
    )


def save_gif_from_revolutions(
    revolution_frames: list[list[dict[str, Any]]],
    gif_path: Path,
    frame_durations_seconds: list[float] | None = None,
) -> None:
    if not revolution_frames:
        revolution_frames = [[]]

    all_points = [point for frame in revolution_frames for point in frame]
    bounds = compute_plot_bounds(all_points)
    frames = [
        render_animation_frame(points, bounds, frame_label=f"rev {index + 1}/{len(revolution_frames)}")
        for index, points in enumerate(revolution_frames)
    ]

    if frame_durations_seconds and len(frame_durations_seconds) == len(frames):
        durations_ms = [max(10, int(round(seconds * 1000.0))) for seconds in frame_durations_seconds]
    else:
        durations_ms = [40] * len(frames)

    frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def save_gif_atomic(packet_records: list[dict[str, Any]], gif_path: Path) -> None:
    temp_path = gif_path.with_suffix(gif_path.suffix + ".tmp")
    save_gif(packet_records, temp_path)
    temp_path.replace(gif_path)


def save_gif_from_revolutions_atomic(
    revolution_frames: list[list[dict[str, Any]]],
    gif_path: Path,
    frame_durations_seconds: list[float] | None = None,
) -> None:
    temp_path = gif_path.with_suffix(gif_path.suffix + ".tmp")
    save_gif_from_revolutions(revolution_frames, temp_path, frame_durations_seconds=frame_durations_seconds)
    temp_path.replace(gif_path)


def is_rotation_wrap(previous_start_angle: float, current_start_angle: float, wrap_threshold_deg: float = 45.0) -> bool:
    return current_start_angle + wrap_threshold_deg < previous_start_angle


def save_plot(points: list[dict[str, Any]], figure_path: Path) -> None:
    plt = load_pyplot("Agg")
    plt.figure(figsize=(8, 8))

    if points:
        x_values = [point["x_mm"] for point in points]
        y_values = [point["y_mm"] for point in points]
        intensity_values = [point["intensity"] for point in points]
        plt.scatter(x_values, y_values, c=intensity_values, s=5, cmap="viridis", alpha=0.8)
    else:
        plt.text(0.5, 0.5, "No valid scan points captured", ha="center", va="center", transform=plt.gca().transAxes)

    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.title("YDLIDAR G2 2D scan")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()


def ensure_live_display_available() -> None:
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return

    raise RuntimeError(
        "No GUI display is available in this session. Run locally on the desktop or reconnect with ssh -X/ssh -Y and a working X server."
    )


def run_live_viewer(port: str, baudrate: int, history_packets: int, view_range_mm: float) -> int:
    ensure_live_display_available()
    plt = load_pyplot("TkAgg")

    plt.ion()
    figure, axis = plt.subplots(figsize=(8, 8))
    scatter = axis.scatter([], [], s=8, c=[], cmap="viridis", vmin=0, vmax=1023)
    axis.set_title("YDLIDAR G2 live view")
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.grid(True, alpha=0.3)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-view_range_mm, view_range_mm)
    axis.set_ylim(-view_range_mm, view_range_mm)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axvline(0.0, color="black", linewidth=0.8)
    status_text = axis.text(0.02, 0.98, "connecting...", transform=axis.transAxes, va="top")

    packet_window: deque[list[dict[str, Any]]] = deque(maxlen=max(1, history_packets))
    packet_count = 0
    started_at = time.monotonic()
    buffer = bytearray()

    with serial.Serial(port=port, baudrate=baudrate, timeout=0.05) as lidar_serial:
        lidar_serial.reset_input_buffer()
        lidar_serial.reset_output_buffer()
        lidar_serial.write(START_SCAN_COMMAND)
        lidar_serial.flush()
        response_descriptor = lidar_serial.read(7)

        try:
            while plt.fignum_exists(figure.number):
                chunk = lidar_serial.read(4096)
                if not chunk:
                    plt.pause(0.001)
                    continue

                buffer.extend(chunk)
                extracted_packets, buffer = extract_packets(buffer)
                if not extracted_packets:
                    plt.pause(0.001)
                    continue

                for packet in extracted_packets:
                    packet_record = parse_packet(packet, packet_count, time.monotonic() - started_at)
                    packet_count += 1
                    packet_window.append(valid_points_from_packet(packet_record))

                current_points = [point for packet_points in packet_window for point in packet_points]
                if current_points:
                    x_values = [point["x_mm"] for point in current_points]
                    y_values = [point["y_mm"] for point in current_points]
                    intensity_values = [point["intensity"] for point in current_points]
                    scatter.set_offsets(list(zip(x_values, y_values)))
                    scatter.set_array(intensity_values)
                else:
                    scatter.set_offsets([])
                    scatter.set_array([])

                elapsed = time.monotonic() - started_at
                status_text.set_text(
                    f"packets={packet_count} points={len(current_points)} hz~{packet_count / max(elapsed, 1e-6):.1f} descriptor={response_descriptor.hex()}"
                )
                figure.canvas.draw_idle()
                figure.canvas.flush_events()
                plt.pause(0.001)
        finally:
            lidar_serial.write(STOP_SCAN_COMMAND)
            lidar_serial.flush()

    plt.ioff()
    plt.close(figure)
    return 0


def save_outputs(
    output_dir: Path,
    port: str,
    baudrate: int,
    duration_seconds: float,
    response_descriptor_hex: str,
    total_bytes_read: int,
    packet_records: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"g2_capture_{timestamp}.json"
    figure_path = output_dir / f"g2_capture_{timestamp}.png"
    gif_path = output_dir / f"g2_capture_{timestamp}.gif"

    valid_packet_count = sum(1 for packet in packet_records if packet["checksum_valid"])
    points = flatten_points(packet_records)

    payload = {
        "device": "YDLIDAR G2",
        "port": port,
        "baudrate": baudrate,
        "duration_seconds": duration_seconds,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "response_descriptor_hex": response_descriptor_hex,
        "total_bytes_read": total_bytes_read,
        "packet_count": len(packet_records),
        "valid_packet_count": valid_packet_count,
        "point_count": len(points),
        "packets": packet_records,
    }

    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)

    save_plot(points, figure_path)
    save_gif_atomic(packet_records, gif_path)
    return json_path, figure_path, gif_path


def capture_and_save_with_realtime_gif(
    output_dir: Path,
    port: str,
    baudrate: int,
    duration_seconds: float,
    revolutions_per_frame: int,
) -> tuple[str, int, int, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"g2_capture_{timestamp}.json"
    figure_path = output_dir / f"g2_capture_{timestamp}.png"
    gif_path = output_dir / f"g2_capture_{timestamp}.gif"

    packet_records: list[dict[str, Any]] = []
    buffer = bytearray()
    total_bytes_read = 0
    previous_start_angle: float | None = None
    revolution_count = 0
    current_revolution_points: list[dict[str, Any]] = []
    completed_revolution_frames: list[list[dict[str, Any]]] = []
    completed_revolution_durations_seconds: list[float] = []
    last_revolution_completed_at = 0.0
    effective_revolutions_per_frame = max(1, revolutions_per_frame)
    pending_frame_points: list[dict[str, Any]] = []
    pending_frame_duration_seconds = 0.0
    pending_frame_revolutions = 0

    save_gif_from_revolutions_atomic([], gif_path, frame_durations_seconds=None)

    with serial.Serial(port=port, baudrate=baudrate, timeout=0.2) as lidar_serial:
        lidar_serial.reset_input_buffer()
        lidar_serial.reset_output_buffer()
        lidar_serial.write(START_SCAN_COMMAND)
        lidar_serial.flush()

        response_descriptor = lidar_serial.read(7)
        started_at = time.monotonic()
        capture_deadline = started_at + duration_seconds
        stop_capture = False

        try:
            while time.monotonic() < capture_deadline:
                chunk = lidar_serial.read(4096)
                if not chunk:
                    continue

                total_bytes_read += len(chunk)
                buffer.extend(chunk)
                extracted_packets, buffer = extract_packets(buffer)

                for packet in extracted_packets:
                    now = time.monotonic()
                    if now >= capture_deadline:
                        stop_capture = True
                        break

                    elapsed_seconds = now - started_at
                    packet_record = parse_packet(packet, len(packet_records), elapsed_seconds)
                    packet_records.append(packet_record)

                    current_start_angle = float(packet_record["start_angle_deg"])
                    if previous_start_angle is not None and is_rotation_wrap(previous_start_angle, current_start_angle):
                        if current_revolution_points:
                            revolution_count += 1
                            revolution_period_seconds = max(0.01, elapsed_seconds - last_revolution_completed_at)
                            last_revolution_completed_at = elapsed_seconds
                            pending_frame_points.extend(current_revolution_points)
                            pending_frame_duration_seconds += revolution_period_seconds
                            pending_frame_revolutions += 1

                            if pending_frame_revolutions >= effective_revolutions_per_frame:
                                completed_revolution_frames.append(pending_frame_points)
                                completed_revolution_durations_seconds.append(pending_frame_duration_seconds)
                                save_gif_from_revolutions_atomic(
                                    completed_revolution_frames,
                                    gif_path,
                                    frame_durations_seconds=completed_revolution_durations_seconds,
                                )
                                print(
                                    f"GIF updated: frame={len(completed_revolution_frames)} merged_revolutions={pending_frame_revolutions} elapsed={elapsed_seconds:.2f}s frame_period={pending_frame_duration_seconds:.3f}s path={gif_path}",
                                    flush=True,
                                )
                                pending_frame_points = []
                                pending_frame_duration_seconds = 0.0
                                pending_frame_revolutions = 0
                        current_revolution_points = []

                    current_revolution_points.extend(valid_points_from_packet(packet_record))
                    previous_start_angle = current_start_angle

                if stop_capture:
                    break
        finally:
            lidar_serial.write(STOP_SCAN_COMMAND)
            lidar_serial.flush()

    points = flatten_points(packet_records)
    valid_packet_count = sum(1 for packet in packet_records if packet["checksum_valid"])
    payload = {
        "device": "YDLIDAR G2",
        "port": port,
        "baudrate": baudrate,
        "duration_seconds": duration_seconds,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "response_descriptor_hex": response_descriptor.hex(),
        "total_bytes_read": total_bytes_read,
        "packet_count": len(packet_records),
        "valid_packet_count": valid_packet_count,
        "point_count": len(points),
        "packets": packet_records,
    }

    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)

    if pending_frame_points:
        completed_revolution_frames.append(pending_frame_points)
        completed_revolution_durations_seconds.append(max(0.01, pending_frame_duration_seconds))

    save_plot(points, figure_path)
    save_gif_from_revolutions_atomic(
        completed_revolution_frames,
        gif_path,
        frame_durations_seconds=completed_revolution_durations_seconds,
    )
    return response_descriptor.hex(), len(packet_records), valid_packet_count, json_path, figure_path, gif_path


def main() -> int:
    arguments = build_parser().parse_args()
    output_dir = Path(arguments.output_dir).resolve()

    if arguments.live:
        return run_live_viewer(
            port=arguments.port,
            baudrate=arguments.baudrate,
            history_packets=arguments.live_history_packets,
            view_range_mm=arguments.live_range_mm,
        )

    response_descriptor_hex, packet_count, valid_packet_count, json_path, figure_path, gif_path = (
        capture_and_save_with_realtime_gif(
            output_dir=output_dir,
            port=arguments.port,
            baudrate=arguments.baudrate,
            duration_seconds=arguments.duration,
            revolutions_per_frame=arguments.revolutions_per_frame,
        )
    )

    print(f"Response descriptor: {response_descriptor_hex}")
    print(f"Port: {arguments.port}")
    print(f"Packets captured: {packet_count}")
    print(f"Valid packets: {valid_packet_count}")
    print(f"JSON: {json_path}")
    print(f"Figure: {figure_path}")
    print(f"GIF: {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())