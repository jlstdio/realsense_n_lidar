from __future__ import annotations

import argparse
from datetime import datetime
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw
import serial

START_SCAN_COMMAND = b"\xA5\x60"
STOP_SCAN_COMMAND = b"\xA5\x65"
PACKET_HEADER = b"\xAA\x55"
DEFAULT_BAUDRATE = 230400


@dataclass
class IcpResult:
    rotation: np.ndarray
    translation: np.ndarray
    rmse: float
    matched_count: int


@dataclass
class Pose2D:
    x_mm: float = 0.0
    y_mm: float = 0.0
    yaw_rad: float = 0.0


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YDLIDAR G2 ICP-based 2D SLAM-like odometry: 3s warmup + 30s tracking"
    )
    parser.add_argument("--port", default=detect_default_port(), help="Serial port path.")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="Serial baudrate.")
    parser.add_argument("--warmup-seconds", type=float, default=3.0, help="Warmup duration before origin reset.")
    parser.add_argument("--track-seconds", type=float, default=30.0, help="Tracking duration after warmup.")
    parser.add_argument("--min-range-mm", type=float, default=120.0, help="Minimum valid lidar range.")
    parser.add_argument("--max-range-mm", type=float, default=4000.0, help="Maximum valid lidar range.")
    parser.add_argument("--icp-max-points", type=int, default=280, help="Max points used per scan in ICP.")
    parser.add_argument("--icp-iters", type=int, default=18, help="Max ICP iterations per scan.")
    parser.add_argument("--icp-match-th-mm", type=float, default=220.0, help="Max correspondence distance.")
    parser.add_argument(
        "--map-output-dir",
        default=str(Path(__file__).resolve().parent / "ydlidar" / "output"),
        help="Directory where SLAM map PNG is saved.",
    )
    parser.add_argument("--map-max-points", type=int, default=120000, help="Max map points retained in memory.")
    parser.add_argument("--gif-frame-step", type=int, default=2, help="Use every N-th trajectory sample for GIF frame.")
    parser.add_argument("--gif-frame-ms", type=int, default=90, help="Duration per GIF frame in milliseconds.")
    return parser


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


def packet_points(packet: bytes, min_range_mm: float, max_range_mm: float) -> tuple[float, np.ndarray]:
    sample_count = packet[3]
    start_angle = first_level_angle(packet[4], packet[5])
    end_angle = first_level_angle(packet[6], packet[7])
    angle_delta = diff_angle(end_angle, start_angle)

    points = []
    for sample_index in range(sample_count):
        offset = 10 + (sample_index * 3)
        distance_mm = (packet[offset + 1] >> 2) | (packet[offset + 2] << 6)
        if distance_mm <= 0:
            continue
        if distance_mm < min_range_mm or distance_mm > max_range_mm:
            continue

        if sample_index == 0:
            base_angle = start_angle
        elif sample_index == sample_count - 1:
            base_angle = end_angle
        else:
            base_angle = interpolated_angle(sample_index, angle_delta, sample_count, start_angle)

        point_angle = corrected_angle(base_angle, distance_mm)
        x_mm = distance_mm * math.cos(math.radians(point_angle))
        y_mm = -distance_mm * math.sin(math.radians(point_angle))
        points.append((x_mm, y_mm))

    if not points:
        return start_angle, np.empty((0, 2), dtype=np.float64)

    return start_angle, np.asarray(points, dtype=np.float64)


def is_rotation_wrap(previous_start_angle: float, current_start_angle: float, wrap_threshold_deg: float = 45.0) -> bool:
    return current_start_angle + wrap_threshold_deg < previous_start_angle


def downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points

    step = max(1, len(points) // max_points)
    sampled = points[::step]
    if len(sampled) > max_points:
        sampled = sampled[:max_points]
    return sampled


def best_fit_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid

    h_mat = source_centered.T @ target_centered
    u_mat, _, v_t = np.linalg.svd(h_mat)
    rotation = v_t.T @ u_mat.T

    if np.linalg.det(rotation) < 0:
        v_t[-1, :] *= -1
        rotation = v_t.T @ u_mat.T

    translation = target_centroid - (rotation @ source_centroid)
    return rotation, translation


def nearest_neighbor_indices(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = source[:, np.newaxis, :] - target[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    nearest_idx = np.argmin(dists, axis=1)
    nearest_dist = dists[np.arange(len(source)), nearest_idx]
    return nearest_idx, nearest_dist


def icp_align(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_iterations: int,
    match_threshold_mm: float,
) -> IcpResult | None:
    if len(source_points) < 20 or len(target_points) < 20:
        return None

    transformed = source_points.copy()
    total_rotation = np.eye(2, dtype=np.float64)
    total_translation = np.zeros(2, dtype=np.float64)
    prev_rmse = float("inf")
    matched_count = 0

    for _ in range(max_iterations):
        nn_idx, nn_dist = nearest_neighbor_indices(transformed, target_points)
        mask = nn_dist < match_threshold_mm
        if np.count_nonzero(mask) < 12:
            return None

        matched_src = transformed[mask]
        matched_tgt = target_points[nn_idx[mask]]
        rotation_step, translation_step = best_fit_transform(matched_src, matched_tgt)

        transformed = (rotation_step @ transformed.T).T + translation_step

        total_rotation = rotation_step @ total_rotation
        total_translation = (rotation_step @ total_translation) + translation_step

        rmse = float(np.sqrt(np.mean(np.square(nn_dist[mask]))))
        matched_count = int(np.count_nonzero(mask))
        if abs(prev_rmse - rmse) < 0.3:
            break
        prev_rmse = rmse

    return IcpResult(total_rotation, total_translation, prev_rmse, matched_count)


class IncrementalSlam:
    def __init__(self, max_points: int, icp_iters: int, match_threshold_mm: float):
        self.pose = Pose2D()
        self._previous_scan: np.ndarray | None = None
        self.max_points = max_points
        self.icp_iters = icp_iters
        self.match_threshold_mm = match_threshold_mm

    def update(self, current_scan: np.ndarray) -> IcpResult | None:
        current_scan = downsample_points(current_scan, self.max_points)
        if len(current_scan) < 20:
            return None

        if self._previous_scan is None:
            self._previous_scan = current_scan
            return None

        icp_result = icp_align(
            source_points=current_scan,
            target_points=self._previous_scan,
            max_iterations=self.icp_iters,
            match_threshold_mm=self.match_threshold_mm,
        )

        if icp_result is None:
            self._previous_scan = current_scan
            return None

        # ICP gives transform from current->previous frame. Invert for sensor motion previous->current.
        motion_rotation = icp_result.rotation.T
        motion_translation = -(motion_rotation @ icp_result.translation)

        delta_yaw = math.atan2(motion_rotation[1, 0], motion_rotation[0, 0])
        cos_yaw = math.cos(self.pose.yaw_rad)
        sin_yaw = math.sin(self.pose.yaw_rad)
        dx_global = (cos_yaw * motion_translation[0]) - (sin_yaw * motion_translation[1])
        dy_global = (sin_yaw * motion_translation[0]) + (cos_yaw * motion_translation[1])

        self.pose.x_mm += float(dx_global)
        self.pose.y_mm += float(dy_global)
        self.pose.yaw_rad += float(delta_yaw)
        self.pose.yaw_rad = math.atan2(math.sin(self.pose.yaw_rad), math.cos(self.pose.yaw_rad))

        self._previous_scan = current_scan
        return icp_result


def transform_points_to_global(points_local: np.ndarray, pose: Pose2D) -> np.ndarray:
    if len(points_local) == 0:
        return np.empty((0, 2), dtype=np.float64)

    cos_yaw = math.cos(pose.yaw_rad)
    sin_yaw = math.sin(pose.yaw_rad)
    rotation = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
    points_global = (rotation @ points_local.T).T
    points_global[:, 0] += pose.x_mm
    points_global[:, 1] += pose.y_mm
    return points_global


def save_map_visualization(
    map_points_global: np.ndarray,
    trajectory_global: np.ndarray,
    origin_pose: Pose2D,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"ydlidar_slam_map_{timestamp}.png"

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    points_rel = map_points_global.copy()
    traj_rel = trajectory_global.copy()
    points_rel[:, 0] -= origin_pose.x_mm
    points_rel[:, 1] -= origin_pose.y_mm
    traj_rel[:, 0] -= origin_pose.x_mm
    traj_rel[:, 1] -= origin_pose.y_mm

    figure, axis = plt.subplots(figsize=(9, 9))
    axis.scatter(points_rel[:, 0], points_rel[:, 1], s=1.2, c="#2f4858", alpha=0.35, label="Scan map")
    axis.plot(traj_rel[:, 0], traj_rel[:, 1], color="#d1495b", linewidth=2.0, label="Trajectory")
    axis.scatter([0.0], [0.0], s=70, c="#00798c", marker="*", label="Origin (after 3s warmup)", zorder=4)
    axis.scatter([traj_rel[-1, 0]], [traj_rel[-1, 1]], s=50, c="#edae49", marker="o", label="Final pose", zorder=5)
    axis.axhline(0.0, color="#555555", linewidth=0.8)
    axis.axvline(0.0, color="#555555", linewidth=0.8)
    axis.set_title("YDLIDAR ICP SLAM map (relative to warmup origin)")
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.grid(True, alpha=0.25)
    axis.axis("equal")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def compute_bounds_mm(points_rel: np.ndarray, traj_rel: np.ndarray, padding_mm: float = 220.0) -> tuple[float, float, float, float]:
    x_values = np.concatenate((points_rel[:, 0], traj_rel[:, 0])) if len(points_rel) else traj_rel[:, 0]
    y_values = np.concatenate((points_rel[:, 1], traj_rel[:, 1])) if len(points_rel) else traj_rel[:, 1]

    min_x = float(np.min(x_values) - padding_mm)
    max_x = float(np.max(x_values) + padding_mm)
    min_y = float(np.min(y_values) - padding_mm)
    max_y = float(np.max(y_values) + padding_mm)

    if math.isclose(min_x, max_x):
        min_x -= 250.0
        max_x += 250.0
    if math.isclose(min_y, max_y):
        min_y -= 250.0
        max_y += 250.0

    return min_x, max_x, min_y, max_y


def point_to_pixel(x_mm: float, y_mm: float, bounds: tuple[float, float, float, float], image_size: int, margin: int) -> tuple[int, int]:
    min_x, max_x, min_y, max_y = bounds
    usable = image_size - 2 * margin
    x_ratio = (x_mm - min_x) / max(max_x - min_x, 1e-6)
    y_ratio = (y_mm - min_y) / max(max_y - min_y, 1e-6)
    px = margin + int(x_ratio * usable)
    py = image_size - margin - int(y_ratio * usable)
    return px, py


def render_trajectory_frame(
    map_points_rel: np.ndarray,
    traj_rel: np.ndarray,
    bounds: tuple[float, float, float, float],
    elapsed_s: float,
    image_size: int = 820,
    margin: int = 44,
) -> Image.Image:
    image = Image.new("RGB", (image_size, image_size), "#f7f7f3")
    draw = ImageDraw.Draw(image)

    grid_color = (217, 217, 210)
    for step in range(5):
        p = margin + int(((image_size - (2 * margin)) / 4) * step)
        draw.line((p, margin, p, image_size - margin), fill=grid_color, width=1)
        draw.line((margin, p, image_size - margin, p), fill=grid_color, width=1)

    origin_x, origin_y = point_to_pixel(0.0, 0.0, bounds, image_size, margin)
    draw.line((origin_x, margin, origin_x, image_size - margin), fill=(90, 90, 90), width=1)
    draw.line((margin, origin_y, image_size - margin, origin_y), fill=(90, 90, 90), width=1)

    for point in map_points_rel:
        px, py = point_to_pixel(float(point[0]), float(point[1]), bounds, image_size, margin)
        draw.point((px, py), fill=(53, 74, 93))

    if len(traj_rel) >= 2:
        traj_pixels = [point_to_pixel(float(t[0]), float(t[1]), bounds, image_size, margin) for t in traj_rel]
        draw.line(traj_pixels, fill=(199, 62, 93), width=3)

    if len(traj_rel) >= 1:
        start_px, start_py = point_to_pixel(float(traj_rel[0, 0]), float(traj_rel[0, 1]), bounds, image_size, margin)
        cur_px, cur_py = point_to_pixel(float(traj_rel[-1, 0]), float(traj_rel[-1, 1]), bounds, image_size, margin)
        draw.ellipse((start_px - 5, start_py - 5, start_px + 5, start_py + 5), fill=(0, 130, 148))
        draw.ellipse((cur_px - 6, cur_py - 6, cur_px + 6, cur_py + 6), fill=(237, 174, 73))

    draw.text((margin, 14), f"YDLIDAR SLAM trajectory  t={elapsed_s:.1f}s", fill=(24, 24, 24))
    return image


def save_trajectory_gif(
    map_points_global: np.ndarray,
    trajectory_global: np.ndarray,
    trajectory_times_s: np.ndarray,
    origin_pose: Pose2D,
    output_dir: Path,
    frame_step: int,
    frame_ms: int,
) -> Path | None:
    if len(trajectory_global) < 2:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"ydlidar_slam_traj_{timestamp}.gif"

    points_rel = map_points_global.copy()
    traj_rel = trajectory_global.copy()
    points_rel[:, 0] -= origin_pose.x_mm
    points_rel[:, 1] -= origin_pose.y_mm
    traj_rel[:, 0] -= origin_pose.x_mm
    traj_rel[:, 1] -= origin_pose.y_mm

    if len(points_rel) > 30000:
        points_rel = downsample_points(points_rel, 30000)

    bounds = compute_bounds_mm(points_rel, traj_rel)
    step = max(1, frame_step)
    duration_ms = max(20, frame_ms)
    frames: list[Image.Image] = []

    for idx in range(1, len(traj_rel), step):
        partial_traj = traj_rel[: idx + 1]
        elapsed_s = float(trajectory_times_s[idx])
        frame = render_trajectory_frame(points_rel, partial_traj, bounds, elapsed_s=elapsed_s)
        frames.append(frame)

    if not frames:
        return None

    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return output_path


def main() -> int:
    args = build_parser().parse_args()
    total_seconds = args.warmup_seconds + args.track_seconds

    slam = IncrementalSlam(
        max_points=max(60, args.icp_max_points),
        icp_iters=max(4, args.icp_iters),
        match_threshold_mm=max(50.0, args.icp_match_th_mm),
    )

    buffer = bytearray()
    previous_start_angle: float | None = None
    current_revolution_points: list[np.ndarray] = []
    origin_pose: Pose2D | None = None
    map_point_chunks: list[np.ndarray] = []
    trajectory_points: list[tuple[float, float]] = []
    trajectory_times_s: list[float] = []
    max_map_points = max(5000, args.map_max_points)

    print(f"[INFO] Port={args.port} baudrate={args.baudrate}")
    print(f"[INFO] Warmup {args.warmup_seconds:.1f}s -> Track {args.track_seconds:.1f}s (total {total_seconds:.1f}s)")

    with serial.Serial(port=args.port, baudrate=args.baudrate, timeout=0.08) as lidar_serial:
        lidar_serial.reset_input_buffer()
        lidar_serial.reset_output_buffer()
        lidar_serial.write(START_SCAN_COMMAND)
        lidar_serial.flush()
        descriptor = lidar_serial.read(7)
        print(f"[INFO] Descriptor: {descriptor.hex()}")

        started_at = time.monotonic()

        try:
            while True:
                now = time.monotonic()
                elapsed = now - started_at
                if elapsed >= total_seconds:
                    break

                chunk = lidar_serial.read(4096)
                if not chunk:
                    continue

                buffer.extend(chunk)
                extracted_packets, buffer = extract_packets(buffer)
                if not extracted_packets:
                    continue

                for packet in extracted_packets:
                    if not checksum_is_valid(packet):
                        continue

                    start_angle, points = packet_points(packet, args.min_range_mm, args.max_range_mm)
                    if len(points):
                        current_revolution_points.append(points)

                    if previous_start_angle is not None and is_rotation_wrap(previous_start_angle, start_angle):
                        if current_revolution_points:
                            revolution_scan = np.vstack(current_revolution_points)
                            icp = slam.update(revolution_scan)
                            elapsed = time.monotonic() - started_at

                            if elapsed >= args.warmup_seconds:
                                if origin_pose is None:
                                    origin_pose = Pose2D(slam.pose.x_mm, slam.pose.y_mm, slam.pose.yaw_rad)
                                    trajectory_points.append((slam.pose.x_mm, slam.pose.y_mm))
                                    trajectory_times_s.append(0.0)
                                    print("[TRACK] Origin fixed after warmup. Tracking movement now.")
                                else:
                                    trajectory_points.append((slam.pose.x_mm, slam.pose.y_mm))
                                    trajectory_times_s.append(max(0.0, elapsed - args.warmup_seconds))
                                    map_points_global = transform_points_to_global(revolution_scan, slam.pose)
                                    map_point_chunks.append(map_points_global)
                                    point_total = sum(len(chunk) for chunk in map_point_chunks)
                                    if point_total > max_map_points:
                                        merged = np.vstack(map_point_chunks)
                                        map_point_chunks = [downsample_points(merged, max_map_points)]

                                    rel_x = slam.pose.x_mm - origin_pose.x_mm
                                    rel_y = slam.pose.y_mm - origin_pose.y_mm
                                    rel_dist = math.hypot(rel_x, rel_y)
                                    rel_yaw_deg = math.degrees(slam.pose.yaw_rad - origin_pose.yaw_rad)
                                    rel_yaw_deg = (rel_yaw_deg + 180.0) % 360.0 - 180.0
                                    if icp is None:
                                        print(
                                            f"[TRACK] t={elapsed - args.warmup_seconds:5.2f}s  dx={rel_x:8.1f}mm  dy={rel_y:8.1f}mm  d={rel_dist:8.1f}mm  yaw={rel_yaw_deg:7.2f}deg  (ICP low confidence)"
                                        )
                                    else:
                                        print(
                                            f"[TRACK] t={elapsed - args.warmup_seconds:5.2f}s  dx={rel_x:8.1f}mm  dy={rel_y:8.1f}mm  d={rel_dist:8.1f}mm  yaw={rel_yaw_deg:7.2f}deg  rmse={icp.rmse:6.1f}mm  matches={icp.matched_count:3d}"
                                        )

                        current_revolution_points = []

                    previous_start_angle = start_angle
        finally:
            lidar_serial.write(STOP_SCAN_COMMAND)
            lidar_serial.flush()

    if origin_pose is None:
        print("[WARN] Warmup completed but no stable scans were produced for tracking.")
        return 1

    final_rel_x = slam.pose.x_mm - origin_pose.x_mm
    final_rel_y = slam.pose.y_mm - origin_pose.y_mm
    final_rel_dist = math.hypot(final_rel_x, final_rel_y)
    final_rel_yaw_deg = math.degrees(slam.pose.yaw_rad - origin_pose.yaw_rad)
    final_rel_yaw_deg = (final_rel_yaw_deg + 180.0) % 360.0 - 180.0

    print("[SUMMARY] Tracking finished.")
    print(f"[SUMMARY] Relative movement: dx={final_rel_x:.1f} mm, dy={final_rel_y:.1f} mm, distance={final_rel_dist:.1f} mm")
    print(f"[SUMMARY] Relative heading change: {final_rel_yaw_deg:.2f} deg")

    if map_point_chunks and trajectory_points:
        map_points_array = np.vstack(map_point_chunks)
        trajectory_array = np.asarray(trajectory_points, dtype=np.float64)
        map_path = save_map_visualization(
            map_points_global=map_points_array,
            trajectory_global=trajectory_array,
            origin_pose=origin_pose,
            output_dir=Path(args.map_output_dir).resolve(),
        )
        print(f"[SUMMARY] SLAM map saved: {map_path}")

        gif_path = save_trajectory_gif(
            map_points_global=map_points_array,
            trajectory_global=trajectory_array,
            trajectory_times_s=np.asarray(trajectory_times_s, dtype=np.float64),
            origin_pose=origin_pose,
            output_dir=Path(args.map_output_dir).resolve(),
            frame_step=args.gif_frame_step,
            frame_ms=args.gif_frame_ms,
        )
        if gif_path is not None:
            print(f"[SUMMARY] SLAM trajectory GIF saved: {gif_path}")
        else:
            print("[SUMMARY] SLAM trajectory GIF skipped (too few trajectory points).")
    else:
        print("[SUMMARY] SLAM map skipped (not enough post-warmup scan data).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
