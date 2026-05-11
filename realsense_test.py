import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyrealsense2 as rs

show_window = os.environ.get('RS_SHOW_WINDOW') == '1'
draw_legend = os.environ.get('RS_DRAW_LEGEND') == '1'
adaptive_scale = os.environ.get('RS_ADAPTIVE_SCALE', '1') != '0'
output_dir = Path(os.environ.get('RS_OUTPUT_DIR', '/home/jl/realsense/captures'))
headless_max_frames = int(os.environ.get('RS_HEADLESS_FRAMES', '90'))
headless_save_every = int(os.environ.get('RS_HEADLESS_SAVE_EVERY', '30'))
save_pointcloud = os.environ.get('RS_SAVE_POINTCLOUD', '1') != '0'
depth_min_mm = int(os.environ.get('RS_DEPTH_MIN_MM', '120'))
depth_max_mm = int(os.environ.get('RS_DEPTH_MAX_MM', '1400'))
depth_gamma = float(os.environ.get('RS_DEPTH_GAMMA', '0.60'))
adaptive_low_percentile = float(os.environ.get('RS_ADAPTIVE_LOW_PCT', '10'))
adaptive_high_percentile = float(os.environ.get('RS_ADAPTIVE_HIGH_PCT', '90'))
adaptive_min_span_mm = int(os.environ.get('RS_ADAPTIVE_MIN_SPAN_MM', '300'))

if show_window:
    import cv2
else:
    cv2 = None


def colorize_normalized(normalized: np.ndarray) -> np.ndarray:
    emphasized = np.power(normalized, depth_gamma)
    red = np.clip(1.5 - np.abs(4.0 * emphasized - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * emphasized - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * emphasized - 1.0), 0.0, 1.0)
    return (np.dstack((red, green, blue)) * 255).astype(np.uint8)


def compute_depth_range(depth_image: np.ndarray) -> tuple[int, int]:
    if not adaptive_scale:
        return depth_min_mm, depth_max_mm

    valid = depth_image[depth_image > 0]
    if valid.size < 100:
        return depth_min_mm, depth_max_mm

    low = int(np.percentile(valid, adaptive_low_percentile))
    high = int(np.percentile(valid, adaptive_high_percentile))
    low = max(low, depth_min_mm)
    high = min(high, depth_max_mm)

    if high - low < adaptive_min_span_mm:
        center = (low + high) // 2
        half_span = adaptive_min_span_mm // 2
        low = max(depth_min_mm, center - half_span)
        high = min(depth_max_mm, center + half_span)

    if high <= low:
        return depth_min_mm, depth_max_mm

    return low, high


def make_depth_preview(depth_image: np.ndarray, range_min_mm: int, range_max_mm: int) -> np.ndarray:
    clipped = np.clip(depth_image, range_min_mm, range_max_mm).astype(np.float32)
    normalized = (clipped - range_min_mm) / max(range_max_mm - range_min_mm, 1)
    preview = colorize_normalized(normalized)
    preview[depth_image == 0] = 0
    return preview


def add_legend(image: Image.Image, range_min_mm: int, range_max_mm: int) -> Image.Image:
    legend_width = 110
    font = ImageFont.load_default()
    output = Image.new('RGB', (image.width + legend_width, image.height), 'black')
    output.paste(image, (0, 0))

    draw = ImageDraw.Draw(output)
    legend_top = 20
    legend_bottom = image.height - 20
    legend_height = max(legend_bottom - legend_top, 1)
    legend_x0 = image.width + 18
    legend_x1 = legend_x0 + 22

    values = np.linspace(1.0, 0.0, legend_height, dtype=np.float32).reshape(-1, 1)
    legend_colors = colorize_normalized(values)[:, 0, :]

    for offset, color in enumerate(legend_colors):
        y = legend_top + offset
        draw.line((legend_x0, y, legend_x1, y), fill=tuple(int(channel) for channel in color), width=1)

    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    for tick in ticks:
        y = int(legend_bottom - tick * legend_height)
        depth_value = int(range_min_mm + tick * (range_max_mm - range_min_mm))
        draw.line((legend_x1 + 2, y, legend_x1 + 8, y), fill='white', width=1)
        draw.text((legend_x1 + 12, y - 6), f'{depth_value}mm', fill='white', font=font)

    draw.text((image.width + 12, 4), 'Depth', fill='white', font=font)
    return output


def build_output_image(color_image: np.ndarray, depth_image: np.ndarray) -> tuple[Image.Image, tuple[int, int]]:
    range_min_mm, range_max_mm = compute_depth_range(depth_image)
    depth_preview = make_depth_preview(depth_image, range_min_mm, range_max_mm)
    combined = Image.fromarray(np.hstack((color_image, depth_preview)))
    if draw_legend:
        combined = add_legend(combined, range_min_mm, range_max_mm)
    return combined, (range_min_mm, range_max_mm)


def save_headless_preview(color_image: np.ndarray, depth_image: np.ndarray, frame_count: int) -> tuple[Image.Image, tuple[int, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image, depth_range = build_output_image(color_image, depth_image)
    image.save(output_dir / f'realsense_{frame_count:04d}.png')
    return image, depth_range


def save_pointcloud_ply(
    pointcloud: rs.pointcloud,
    depth_frame: rs.frame,
    color_frame: rs.frame,
    frame_count: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    points = pointcloud.calculate(depth_frame)
    pointcloud.map_to(color_frame)
    ply_path = output_dir / f'realsense_{frame_count:04d}.ply'
    points.export_to_ply(str(ply_path), color_frame)
    return ply_path


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
pointcloud = rs.pointcloud()
frame_count = 0
saved_frames = []

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        frame_count += 1

        if show_window:
            display_image, _depth_range = build_output_image(color_image, depth_image)
            cv2.imshow('RealSense', cv2.cvtColor(np.asarray(display_image), cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        elif frame_count % headless_save_every == 0:
            image, depth_range = save_headless_preview(color_image, depth_image, frame_count)
            saved_frames.append(image)
            ply_path = None
            if save_pointcloud:
                ply_path = save_pointcloud_ply(pointcloud, depth_frame, color_frame, frame_count)
            center_distance = depth_frame.get_distance(
                depth_frame.get_width() // 2,
                depth_frame.get_height() // 2,
            )
            print(
                f"Saved preview: frame={frame_count}, center_distance={center_distance:.3f}m, "
                f"range={depth_range[0]}-{depth_range[1]}mm, "
                f"file={output_dir / f'realsense_{frame_count:04d}.png'}"
            )
            if ply_path is not None:
                print(f"Saved point cloud: {ply_path}")

            if frame_count >= headless_max_frames:
                break
finally:
    pipeline.stop()
    if show_window:
        cv2.destroyAllWindows()
    elif saved_frames:
        saved_frames[0].save(
            output_dir / 'realsense_capture.gif',
            save_all=True,
            append_images=saved_frames[1:],
            duration=200,
            loop=0,
        )
        print(f"Saved animation: {output_dir / 'realsense_capture.gif'}")
