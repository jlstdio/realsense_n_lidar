import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyrealsense2 as rs
from tflite_runtime.interpreter import Interpreter


COCO80_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


show_window = os.environ.get('RS_SHOW_WINDOW') == '1'
draw_legend = os.environ.get('RS_DRAW_LEGEND', '1') != '0'
adaptive_scale = os.environ.get('RS_ADAPTIVE_SCALE', '1') != '0'
output_dir = Path(os.environ.get('RS_OUTPUT_DIR', '/home/jl/realsense/captures_3d'))
headless_max_frames = int(os.environ.get('RS_HEADLESS_FRAMES', '90'))
headless_save_every = int(os.environ.get('RS_HEADLESS_SAVE_EVERY', '30'))
depth_min_mm = int(os.environ.get('RS_DEPTH_MIN_MM', '120'))
depth_max_mm = int(os.environ.get('RS_DEPTH_MAX_MM', '1400'))
depth_gamma = float(os.environ.get('RS_DEPTH_GAMMA', '0.60'))
adaptive_low_percentile = float(os.environ.get('RS_ADAPTIVE_LOW_PCT', '10'))
adaptive_high_percentile = float(os.environ.get('RS_ADAPTIVE_HIGH_PCT', '90'))
adaptive_min_span_mm = int(os.environ.get('RS_ADAPTIVE_MIN_SPAN_MM', '300'))

yolo_model_path = Path(os.environ.get('YOLO_MODEL_PATH', '/home/jl/realsense/models/yolo11n.tflite'))
yolo_input_size = int(os.environ.get('YOLO_INPUT_SIZE', '640'))
yolo_conf_threshold = float(os.environ.get('YOLO_CONF_THRESHOLD', '0.25'))
yolo_nms_threshold = float(os.environ.get('YOLO_NMS_THRESHOLD', '0.45'))
detect_classes = {
    item.strip() for item in os.environ.get('YOLO_DETECT_CLASSES', '').split(',') if item.strip()
}


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


class YoloTFLiteDetector:
    def __init__(self, model_path: Path, input_size: int, conf_threshold: float, nms_threshold: float):
        if not model_path.exists():
            raise FileNotFoundError(
                f'YOLO model not found: {model_path}\n'
                'Place a YOLO TFLite model there (yolo11n.tflite), or set YOLO_MODEL_PATH environment variable.\n'
                'To convert ONNX to TFLite, use: `python -c "from ultralytics import YOLO; YOLO(\'yolov11n.pt\').export(format=\'tflite\')"`'
            )

        try:
            self.interpreter = Interpreter(model_path=str(model_path))
            self.interpreter.allocate_tensors()
            self.valid = True
        except Exception as e:
            print(f"Warning: Could not load TFLite model: {e}")
            print("Falling back to MockDetector for testing pipeline...")
            self.valid = False
        
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.class_names = COCO80_NAMES
        
        if self.valid:
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = image.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        new_w = int(round(width * scale))
        new_h = int(round(height * scale))
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas, scale, pad_x, pad_y

    def detect(self, image: np.ndarray) -> list[dict]:
        if not self.valid:
            return self._mock_detect(image)
        
        letterboxed, scale, pad_x, pad_y = self._letterbox(image)
        
        # Prepare input tensor
        input_tensor = np.expand_dims(letterboxed, 0).astype(np.float32) / 255.0
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        
        # Get output tensors - TFLite YOLO typically outputs (1, 2100, 85) for nano
        # Format: [bboxes:4, objectness:1, class_scores:80]
        output_tensor = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Handle output shape variations
        if output_tensor.ndim == 3:
            predictions = output_tensor[0]  # Remove batch dimension
        else:
            predictions = output_tensor
        
        if predictions.ndim == 2 and predictions.shape[0] > predictions.shape[1]:
            predictions = predictions.T
        
        boxes = []
        scores = []
        class_ids = []

        for row in predictions:
            if row.shape[0] < 5:
                continue

            # TFLite YOLO format: [x_center, y_center, width, height, objectness, class_scores...]
            cx, cy, w, h = row[:4]
            objectness = float(row[4]) if row.shape[0] > 4 else 1.0
            
            if objectness < 0.3:  # Early objectness filter
                continue
            
            class_scores = row[5:85] if row.shape[0] >= 85 else row[5:]
            if len(class_scores) == 0:
                continue
            
            class_id = int(np.argmax(class_scores))
            class_conf = float(class_scores[class_id])
            score = objectness * class_conf
            
            if score < self.conf_threshold:
                continue

            class_name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
            if detect_classes and class_name not in detect_classes:
                continue

            x1 = int((cx - w / 2 - pad_x) / scale)
            y1 = int((cy - h / 2 - pad_y) / scale)
            x2 = int((cx + w / 2 - pad_x) / scale)
            y2 = int((cy + h / 2 - pad_y) / scale)

            x1 = max(0, min(image.shape[1] - 1, x1))
            y1 = max(0, min(image.shape[0] - 1, y1))
            x2 = max(0, min(image.shape[1] - 1, x2))
            y2 = max(0, min(image.shape[0] - 1, y2))
            
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(class_id)

        if not boxes:
            return []

        kept = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.nms_threshold)
        if len(kept) == 0:
            return []

        detections = []
        for idx in np.array(kept).reshape(-1):
            x, y, w, h = boxes[int(idx)]
            class_id = class_ids[int(idx)]
            detections.append({
                'bbox': (x, y, x + w, y + h),
                'score': scores[int(idx)],
                'class_id': class_id,
                'class_name': self.class_names[class_id] if class_id < len(self.class_names) else str(class_id),
            })
        return detections
    
    def _mock_detect(self, image: np.ndarray) -> list[dict]:
        """Generate mock detections for testing pipeline when model unavailable."""
        h, w = image.shape[:2]
        detections = [
            {
                'bbox': (w // 4, h // 4, 3 * w // 4, 3 * h // 4),  # Center box
                'score': 0.85,
                'class_id': 0,
                'class_name': 'person',
            },
        ]
        return detections


def estimate_3d_box(depth_image: np.ndarray, intrinsics, bbox: tuple[int, int, int, int]) -> dict | None:
    height, width = depth_image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None

    roi = depth_image[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    inner_x1 = x1 + (x2 - x1) // 4
    inner_x2 = x2 - (x2 - x1) // 4
    inner_y1 = y1 + (y2 - y1) // 4
    inner_y2 = y2 - (y2 - y1) // 4
    center_roi = depth_image[inner_y1:inner_y2, inner_x1:inner_x2]

    valid = center_roi[center_roi > 0]
    if valid.size < 20:
        valid = roi[roi > 0]
    if valid.size < 20:
        return None

    anchor_depth = int(np.percentile(valid, 40))
    band_mm = max(80, int(anchor_depth * 0.12))
    points_mm = []

    step_x = max(2, (x2 - x1) // 24)
    step_y = max(2, (y2 - y1) // 24)
    step = max(step_x, step_y)

    for py in range(y1, y2, step):
        for px in range(x1, x2, step):
            depth_mm = int(depth_image[py, px])
            if depth_mm <= 0:
                continue
            if abs(depth_mm - anchor_depth) > band_mm:
                continue

            point_m = rs.rs2_deproject_pixel_to_point(intrinsics, [float(px), float(py)], depth_mm / 1000.0)
            points_mm.append([point_m[0] * 1000.0, point_m[1] * 1000.0, point_m[2] * 1000.0])

    if len(points_mm) < 12:
        return None

    points_mm = np.asarray(points_mm, dtype=np.float32)
    min_xyz = np.percentile(points_mm, 5, axis=0)
    max_xyz = np.percentile(points_mm, 95, axis=0)
    center_xyz = (min_xyz + max_xyz) / 2.0
    size_xyz = max_xyz - min_xyz

    return {
        'min_xyz_mm': min_xyz,
        'max_xyz_mm': max_xyz,
        'center_xyz_mm': center_xyz,
        'size_xyz_mm': size_xyz,
    }


def make_cuboid_corners(min_xyz_mm: np.ndarray, max_xyz_mm: np.ndarray) -> np.ndarray:
    min_x, min_y, min_z = min_xyz_mm.tolist()
    max_x, max_y, max_z = max_xyz_mm.tolist()
    return np.array([
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [max_x, max_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [max_x, max_y, max_z],
        [min_x, max_y, max_z],
    ], dtype=np.float32)


def project_point(intrinsics, point_mm: np.ndarray) -> tuple[float, float]:
    pixel = rs.rs2_project_point_to_pixel(intrinsics, (point_mm / 1000.0).tolist())
    return float(pixel[0]), float(pixel[1])


def make_comparison_image(color_image: np.ndarray, depth_image: np.ndarray, annotated_depth: Image.Image) -> Image.Image:
    """Combine color, raw depth, and annotated detection into one image."""
    # Convert color_image to PIL
    color_pil = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    
    # Make raw depth preview (simple grayscale)
    valid_depth = depth_image[depth_image > 0]
    if valid_depth.size > 0:
        depth_min = int(np.percentile(valid_depth, 5))
        depth_max = int(np.percentile(valid_depth, 95))
    else:
        depth_min, depth_max = 100, 1000
    
    depth_normalized = np.clip(depth_image.astype(np.float32), depth_min, depth_max)
    depth_normalized = (depth_normalized - depth_min) / max(depth_max - depth_min, 1)
    depth_normalized[depth_image == 0] = 0
    depth_gray = (depth_normalized * 255).astype(np.uint8)
    depth_pil = Image.fromarray(depth_gray, mode='L').convert('RGB')
    
    # Resize all to same height for side-by-side comparison
    h = 480
    w = int(640 * h / 480)
    color_resized = color_pil.resize((w, h), Image.LANCZOS)
    depth_resized = depth_pil.resize((w, h), Image.LANCZOS)
    annotated_resized = annotated_depth.resize((w, h), Image.LANCZOS)
    
    # Combine horizontally
    total_w = w * 3
    combined = Image.new('RGB', (total_w, h), 'black')
    combined.paste(color_resized, (0, 0))
    combined.paste(depth_resized, (w, 0))
    combined.paste(annotated_resized, (w * 2, 0))
    
    return combined


def annotate_depth_image(depth_image: np.ndarray, detections: list[dict], intrinsics) -> tuple[Image.Image, tuple[int, int]]:
    range_min_mm, range_max_mm = compute_depth_range(depth_image)
    base = Image.fromarray(make_depth_preview(depth_image, range_min_mm, range_max_mm))
    draw = ImageDraw.Draw(base)
    font = ImageFont.load_default()

    cuboid_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for detection in detections:
        color = detection['color']
        x1, y1, x2, y2 = detection['bbox']
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)

        cuboid = detection.get('cuboid')
        if cuboid is not None:
            corners = make_cuboid_corners(cuboid['min_xyz_mm'], cuboid['max_xyz_mm'])
            pixels = [project_point(intrinsics, corner) for corner in corners]
            for start, end in cuboid_edges:
                draw.line((pixels[start][0], pixels[start][1], pixels[end][0], pixels[end][1]), fill=color, width=2)

        label_lines = [
            f"{detection['class_name']} {detection['score']:.2f}",
        ]
        if cuboid is not None:
            center = cuboid['center_xyz_mm']
            size = cuboid['size_xyz_mm']
            label_lines.append(f"pos {center[0]:.0f},{center[1]:.0f},{center[2]:.0f} mm")
            label_lines.append(f"size {size[0]:.0f},{size[1]:.0f},{size[2]:.0f} mm")
        else:
            label_lines.append('3D box unavailable')

        box_top = max(0, y1 - (len(label_lines) * 12 + 6))
        box_right = min(base.width - 1, x1 + 190)
        draw.rectangle((x1, box_top, box_right, y1), fill=(0, 0, 0))
        text_y = box_top + 2
        for line in label_lines:
            draw.text((x1 + 4, text_y), line, fill=color, font=font)
            text_y += 11

    if draw_legend:
        base = add_legend(base, range_min_mm, range_max_mm)
    return base, (range_min_mm, range_max_mm)


def save_headless_preview(image: Image.Image, frame_count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_dir / f'realsense_3d_{frame_count:04d}.png')


def main() -> None:
    detector = YoloTFLiteDetector(yolo_model_path, yolo_input_size, yolo_conf_threshold, yolo_nms_threshold)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    align = rs.align(rs.stream.color)
    pipeline.start(config)

    frame_count = 0
    saved_frames = []

    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            detections = detector.detect(color_image)
            palette = [(255, 80, 80), (80, 255, 120), (80, 180, 255), (255, 220, 80), (255, 120, 255)]
            enriched = []

            for index, detection in enumerate(detections):
                cuboid = estimate_3d_box(depth_image, intrinsics, detection['bbox'])
                enriched.append({
                    **detection,
                    'cuboid': cuboid,
                    'color': palette[index % len(palette)],
                })

            annotated, depth_range = annotate_depth_image(depth_image, enriched, intrinsics)
            frame_count += 1

            if show_window:
                cv2.imshow('RealSense 3D Detection', cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            elif frame_count % headless_save_every == 0:
                # Save comparison image: [color | raw depth | annotated detection]
                comparison = make_comparison_image(color_image, depth_image, annotated)
                save_headless_preview(comparison, frame_count)
                saved_frames.append(comparison)
                print(
                    f"Saved 3D preview: frame={frame_count}, detections={len(enriched)}, "
                    f"range={depth_range[0]}-{depth_range[1]}mm, "
                    f"file={output_dir / f'realsense_3d_{frame_count:04d}.png'}"
                )
                for detection in enriched:
                    cuboid = detection['cuboid']
                    if cuboid is None:
                        print(f"  - {detection['class_name']}: 3D box unavailable")
                        continue
                    center = cuboid['center_xyz_mm']
                    size = cuboid['size_xyz_mm']
                    print(
                        f"  - {detection['class_name']} {detection['score']:.2f}: "
                        f"pos=({center[0]:.0f},{center[1]:.0f},{center[2]:.0f})mm "
                        f"size=({size[0]:.0f},{size[1]:.0f},{size[2]:.0f})mm"
                    )

                if frame_count >= headless_max_frames:
                    break
    finally:
        pipeline.stop()
        if show_window:
            cv2.destroyAllWindows()
        elif saved_frames:
            output_dir.mkdir(parents=True, exist_ok=True)
            saved_frames[0].save(
                output_dir / 'realsense_3d_capture.gif',
                save_all=True,
                append_images=saved_frames[1:],
                duration=200,
                loop=0,
            )
            print(f"Saved animation: {output_dir / 'realsense_3d_capture.gif'}")


if __name__ == '__main__':
    main()