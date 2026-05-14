#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv311"
JETSON_INDEX_URL="${JETSON_PIP_EXTRA_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}"

echo "[1/5] Activating virtual environment: ${VENV_DIR}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Virtual environment not found at ${VENV_DIR}." >&2
  echo "Create it first, for example: python3.11 -m venv .venv311" >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"

echo "[2/5] Updating packaging tools"
python -m pip install --upgrade pip setuptools wheel

echo "[3/5] Installing core Python packages"
python -m pip install \
  --extra-index-url "${JETSON_INDEX_URL}" \
  "numpy==1.26.4" \
  "pillow>=10,<13" \
  "imageio>=2.37" \
  "plotly>=6,<7" \
  "matplotlib>=3.5,<3.11" \
  "pyserial==3.5" \
  "smbus2>=0.4,<1" \
  "tflite-runtime==2.14.0" \
  "ipykernel>=6,<8"

echo "[4/5] Installing optional packages used by specific scripts"
python -m pip install \
  --extra-index-url "${JETSON_INDEX_URL}" \
  "mujoco>=3.1,<4" \
  "glfw>=2.8,<3" \
  "onnxruntime>=1.18,<2" \
  "opencv-python-headless>=4.10,<4.14"

echo "[5/5] Post-install notes"
cat <<EOF
- pyrealsense2 is intentionally not installed with pip here.
  On Jetson, install librealsense/pyrealsense2 with the RealSense Jetson build or your existing system package flow.
  This repo includes ./install_realsense_jetson_venv311.sh to build pyrealsense2 into .venv311 from source.
- If you need GUI windows instead of headless OpenCV, replace opencv-python-headless with opencv-python from a Jetson-compatible source.
- If ONNX on Jetson should use GPU, replace onnxruntime with a Jetson-compatible onnxruntime-gpu wheel from your preferred index.

Suggested checks:
  source .venv311/bin/activate
  python -c "import numpy, cv2, serial, smbus2, tflite_runtime.interpreter; print('ok')"
  python -c "import pyrealsense2 as rs; print(rs.__version__)"  # after RealSense install
EOF