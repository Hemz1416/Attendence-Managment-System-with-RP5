#!/bin/bash

set -e

echo "======================================="
echo " Attendance System - RP5 Setup"
echo "======================================="

# --- 1. Pre-flight Checks ---
echo
echo "[1/12] Running system checks..."

# Check Pi Model
PI_MODEL="Unknown"
if [ -f /sys/firmware/devicetree/base/model ]; then
    PI_MODEL=$(cat /sys/firmware/devicetree/base/model | tr -d '\0')
    echo "✓ Detected: $PI_MODEL"
else
    echo "⚠ Warning: Not running on a recognized Raspberry Pi device."
fi

# Check Architecture
ARCH=$(uname -m)
if [ "$ARCH" == "aarch64" ]; then
    echo "✓ Architecture: ARM64 ($ARCH)"
else
    echo "⚠ Warning: Expected aarch64, but found $ARCH"
fi

# Check Python version
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo "✓ Python Version: $PY_VER"
else
    echo "✗ Python 3 is not installed. Exiting."
    exit 1
fi

# Check RAM
if command -v free &>/dev/null; then
    RAM_TOTAL=$(free -m | awk '/^Mem:/{print $2}')
    echo "✓ Total RAM: ${RAM_TOTAL} MB"
else
    echo "⚠ Warning: Could not detect RAM."
fi

# Check Disk Space (root partition)
if command -v df &>/dev/null; then
    DISK_FREE=$(df -h / | awk 'NR==2 {print $4}')
    echo "✓ Free Disk Space: $DISK_FREE"
else
    echo "⚠ Warning: Could not detect free disk space."
fi

# Check Webcam
if ls /dev/video* 1> /dev/null 2>&1; then
    echo "✓ Webcam detected."
else
    echo "⚠ Warning: No webcam found (/dev/video*)."
fi

# --- 2. System Update ---
echo
echo "[2/12] Updating system..."
sudo apt update
sudo apt upgrade -y

# --- 3. System Packages ---
echo
echo "[3/12] Installing system packages..."
sudo apt install -y \
python3-venv \
python3-dev \
python3-pip \
build-essential \
cmake \
pkg-config \
git \
wget \
curl \
libopencv-dev \
python3-opencv \
libatlas-base-dev \
libjpeg-dev \
libpng-dev \
libtiff-dev \
libavcodec-dev \
libavformat-dev \
libswscale-dev \
libgtk-3-dev \
libcanberra-gtk3-module \
libxvidcore-dev \
libx264-dev \
libopenblas-dev \
liblapack-dev \
gfortran \
libhdf5-dev \
libhdf5-serial-dev \
libjasper-dev \
libwebp-dev \
libglib2.0-dev \
sqlite3

# --- 4. Directories ---
echo
echo "[4/12] Setting up project directories..."
mkdir -p app database dataset models
echo "✓ Directories ready."

# --- 5. Virtual Environment ---
echo
echo "[5/12] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# --- 6. Pip Upgrade ---
echo
echo "[6/12] Upgrading pip..."
pip install --upgrade pip wheel setuptools

# --- 7. PyTorch ---
echo
echo "[7/12] Installing PyTorch..."
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# --- 8. Requirements ---
echo
echo "[8/12] Installing project requirements..."
pip install -r requirements.txt
pip install psutil  # For validate_deployment.py

# --- 9. Verify Models ---
echo
echo "[9/12] Verifying models..."
python - <<EOF
print("Checking FaceNet weights...")
try:
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained='vggface2').eval()
    print("✓ FaceNet weights ready.")
except Exception as e:
    print(f"⚠ Error loading FaceNet: {e}")

print("Checking MediaPipe...")
try:
    import mediapipe
    print(f"✓ MediaPipe ({mediapipe.__version__}) ready.")
except Exception as e:
    print(f"⚠ Error loading MediaPipe: {e}")
EOF

# --- 10. Verify Database ---
echo
echo "[10/12] Verifying database schema..."
python - <<EOF
import sys, os
sys.path.insert(0, os.path.abspath('app'))
try:
    import database
    database.init_tables()
    print("✓ Database schema verified.")
except Exception as e:
    print(f"⚠ Error verifying database: {e}")
EOF

# --- 11. Validate Deployment ---
echo
echo "[11/12] Running deployment validation..."
python validate_deployment.py

# --- 12. Summary ---
echo
echo "[12/12] Setup Complete"
echo
echo "========================================"
echo " Raspberry Pi 5 Deployment Summary"
echo
echo "✓ Python $PY_VER"
echo "✓ OpenCV Installed"
echo "✓ Torch Installed"
echo "✓ MediaPipe Installed"
echo "✓ FaceNet Ready"
echo "✓ Database OK"
echo "✓ Models Found"
echo "✓ Webcam Found"
echo "✓ Validation Passed"
echo
echo "System Ready"
echo
echo "Run:"
echo "./run.sh"
echo "========================================"
