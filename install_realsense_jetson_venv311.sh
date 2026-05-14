#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv311"
VENV_PYTHON="${VENV_DIR}/bin/python"
LIBREALSENSE_DIR="${ROOT_DIR}/third_party/librealsense"
BUILD_DIR="${LIBREALSENSE_DIR}/build-venv311-rsusb"
LIBREALSENSE_REF="${LIBREALSENSE_REF:-}"

echo "[1/7] Checking Python virtual environment"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Missing virtual environment Python: ${VENV_PYTHON}" >&2
  echo "Create .venv311 first, then rerun this script." >&2
  exit 1
fi

VENV_SITE_PACKAGES="$("${VENV_PYTHON}" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["platlib"])
PY
)"

echo "[2/7] Installing system build dependencies"
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  git \
  libssl-dev \
  libusb-1.0-0-dev \
  libudev-dev \
  pkg-config \
  python3.11-dev

mkdir -p "$(dirname "${LIBREALSENSE_DIR}")"

echo "[3/7] Fetching librealsense source"
if [[ ! -d "${LIBREALSENSE_DIR}/.git" ]]; then
  if [[ -n "${LIBREALSENSE_REF}" ]]; then
    git clone --depth 1 --branch "${LIBREALSENSE_REF}" https://github.com/IntelRealSense/librealsense.git "${LIBREALSENSE_DIR}"
  else
    git clone --depth 1 https://github.com/IntelRealSense/librealsense.git "${LIBREALSENSE_DIR}"
  fi
else
  git -C "${LIBREALSENSE_DIR}" fetch --tags --prune
  if [[ -n "${LIBREALSENSE_REF}" ]]; then
    git -C "${LIBREALSENSE_DIR}" checkout "${LIBREALSENSE_REF}"
  fi
  git -C "${LIBREALSENSE_DIR}" pull --ff-only
fi

echo "[4/7] Installing udev rules"
(
  cd "${LIBREALSENSE_DIR}"
  sudo ./scripts/setup_udev_rules.sh
)

echo "[5/7] Configuring librealsense for pyrealsense2 in .venv311"
mkdir -p "${BUILD_DIR}"
cmake -S "${LIBREALSENSE_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_WITH_DDS=OFF \
  -DCHECK_FOR_UPDATES=OFF \
  -DFORCE_RSUSB_BACKEND=ON \
  -DPYTHON_EXECUTABLE="${VENV_PYTHON}" \
  -DPython_EXECUTABLE="${VENV_PYTHON}" \
  -DPYTHON_INSTALL_DIR="${VENV_SITE_PACKAGES}/pyrealsense2"

echo "[6/7] Building and installing librealsense + pyrealsense2"
cmake --build "${BUILD_DIR}" --parallel "$(nproc)"
sudo cmake --install "${BUILD_DIR}"
sudo ldconfig

echo "[7/7] Verifying import from .venv311"
"${VENV_PYTHON}" - <<'PY'
import pyrealsense2 as rs
print("pyrealsense2 import ok")
print("version:", getattr(rs, "__version__", "unknown"))
PY

cat <<EOF

RealSense Python binding is now installed for:
  ${VENV_PYTHON}

If camera access still fails on Jetson, the next step is kernel/backend setup from Intel's Jetson guide.
This script builds with RSUSB backend to avoid kernel patching as the default path.
EOF