#!/bin/bash
# Try to download YOLO11n TFLite model from Ultralytics
# If this fails, instructions for manual conversion are provided

set -e

MODELS_DIR="/home/jl/realsense/models"
OUTPUT="$MODELS_DIR/yolo11n.tflite"

mkdir -p "$MODELS_DIR"

echo "Attempting to download YOLO11n TFLite model..."

# Try several possible URLs
URLs=(
    "https://github.com/ultralytics/ultralytics/releases/download/v8.2.104/yolo11n.tflite"
    "https://github.com/ultralytics/assets/raw/main/yolov11n.tflite"
    "https://ultralytics.com/yolov11n.tflite"
)

for URL in "${URLs[@]}"; do
    echo "Trying: $URL"
    if curl -L -f -o "$OUTPUT" "$URL" 2>/dev/null; then
        SIZE=$(stat -c%s "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT" 2>/dev/null)
        if [ "$SIZE" -gt 1000000 ]; then  # Should be > 1MB
            echo "✓ Successfully downloaded YOLO11n.tflite ($((SIZE / 1000000))MB)"
            exit 0
        else
            rm -f "$OUTPUT"
        fi
    fi
done

echo "❌ Could not download pre-built model from public repositories"
echo ""
echo "=== OPTION 1: Convert ONNX to TFLite (on dev machine) ==="
echo "On a machine with Python/PyTorch/TensorFlow:"
echo ""
echo "  1. Install: pip install ultralytics tensorflow"
echo "  2. Convert:"
echo "     python -c \"from ultralytics import YOLO; YOLO('yolov11n.pt').export(format='tflite')\""
echo "  3. Copy yolo11n.tflite to: $OUTPUT"
echo ""
echo "=== OPTION 2: Use existing ONNX model ==="
echo "The ONNX model is available at: $MODELS_DIR/yolo11n.onnx"
echo "But it requires a custom inference backend."
echo ""
echo "Current status: Detector falls back to mock mode for testing"
exit 1
