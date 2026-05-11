import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pyrealsense2 as rs
from PIL import Image


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


class YoloONNXDetector:
    def __init__(self, model_path: Path, input_size: int, conf_threshold: float, nms_threshold: float):
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.class_names = COCO80_NAMES
        self.input_name = self.session.get_inputs()[0].name

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
        letterboxed, scale, pad_x, pad_y = self._letterbox(image)
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        input_tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        outputs = self.session.run(None, {self.input_name: input_tensor})
        predictions = outputs[0]

        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.ndim == 2 and predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []

        for row in predictions:
            if row.shape[0] < 6:
                continue

            cx, cy, w, h = row[:4]
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < self.conf_threshold:
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
            detections.append(
                {
                    'bbox': (x, y, x + w, y + h),
                    'score': scores[int(idx)],
                    'class_id': class_id,
                    'class_name': self.class_names[class_id] if class_id < len(self.class_names) else str(class_id),
                }
            )
        return detections


def build_detector(model_path: Path, input_size: int, conf_threshold: float, nms_threshold: float):
    if model_path.suffix.lower() != '.onnx':
        raise ValueError('Unsupported model format. Use .onnx model only.')
    return YoloONNXDetector(model_path, input_size, conf_threshold, nms_threshold)


def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    output = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = f"{det['class_name']} {det['score']:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 220, 0), -1)
        cv2.putText(output, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return output


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    default_model = base_dir / 'models' / 'yolo11n.onnx'

    parser = argparse.ArgumentParser(
        description="Run YOLO object detection using RealSense color frames and save GIF/MP4."
    )
    parser.add_argument("--model", type=Path, default=default_model, help="YOLO ONNX model path")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "captures", help="Directory to save output")
    parser.add_argument("--output-name", type=str, default="", help="Output filename (default: auto timestamp)")
    parser.add_argument("--duration", type=float, default=10.0, help="Detection duration in seconds")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video FPS")
    parser.add_argument("--width", type=int, default=640, help="Color stream width")
    parser.add_argument("--height", type=int, default=480, help="Color stream height")
    parser.add_argument("--camera-fps", type=int, default=30, help="RealSense camera stream FPS")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--format", type=str, default="gif", choices=["gif", "mp4"], help="Output format")
    return parser.parse_args()


def ensure_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def build_output_path(output_dir: Path, output_name: str, output_format: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = '.gif' if output_format == 'gif' else '.mp4'
    if output_name:
        filename = output_name
        if not filename.lower().endswith(ext):
            filename += ext
    else:
        filename = f"realsense_yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
    return output_dir / filename


def main() -> None:
    args = parse_args()

    ensure_positive(args.duration, "duration")
    ensure_positive(args.fps, "fps")
    ensure_positive(args.camera_fps, "camera-fps")

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    total_frames = int(round(args.duration * args.fps))
    if total_frames < 1:
        raise ValueError("duration * fps must produce at least 1 frame")

    output_path = build_output_path(args.output_dir, args.output_name, args.format)

    print(f"Loading model: {args.model}")
    detector = build_detector(
        model_path=args.model,
        input_size=args.imgsz,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
    )

    pipeline = rs.pipeline()

    print(
        f"Starting RealSense color stream ({args.width}x{args.height}@{args.camera_fps}) and running detection "
        f"for {args.duration:.2f}s at {args.fps:.2f}fps ({total_frames} frames)."
    )

    writer = None
    gif_frames: list[Image.Image] = []
    frame_interval = 1.0 / args.fps
    next_frame_time = time.perf_counter()
    processed = 0

    started = False
    active_camera_fps = args.camera_fps

    try:
        # Some RealSense profiles do not support arbitrary FPS values (e.g., 640x480@10).
        fallback_fps = [args.camera_fps, 30, 15, 6]
        tried_fps = []
        for fps in fallback_fps:
            if fps in tried_fps:
                continue
            tried_fps.append(fps)
            config = rs.config()
            config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, fps)
            try:
                pipeline.start(config)
                started = True
                active_camera_fps = fps
                break
            except RuntimeError:
                continue

        if not started:
            raise RuntimeError(
                f"Could not start RealSense color stream for {args.width}x{args.height} with FPS candidates {tried_fps}"
            )

        if active_camera_fps != args.camera_fps:
            print(f"Requested camera-fps={args.camera_fps} not supported, using {active_camera_fps} instead.")

        while processed < total_frames:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            now = time.perf_counter()
            if now < next_frame_time:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            detections = detector.detect(color_image)
            annotated = draw_detections(color_image, detections)

            if args.format == 'mp4':
                if writer is None:
                    height, width = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")
                writer.write(annotated)
            else:
                gif_frames.append(Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)))

            processed += 1
            next_frame_time += frame_interval

            if processed % max(1, int(args.fps)) == 0 or processed == total_frames:
                print(f"Progress: {processed}/{total_frames} frames")

    finally:
        if started:
            pipeline.stop()
        if writer is not None:
            writer.release()

    if args.format == 'gif':
        if not gif_frames:
            raise RuntimeError('No frames captured for GIF output.')
        duration_ms = max(1, int(round(1000.0 / args.fps)))
        gif_frames[0].save(
            output_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
            optimize=False,
        )

    print(f"Saved detection video: {output_path}")


if __name__ == "__main__":
    main()
