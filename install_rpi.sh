#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "========================================================="
echo "Installing Raspberry Pi 5 Face Recognition Attendance System"
echo "========================================================="

# Update APT repository
echo "Updating apt package list..."
sudo apt-get update

# Install Python 3, venv, pip and system libraries required for OpenCV/PyTorch
echo "Installing Python 3, venv, and system packages (OpenGL, glib, etc.)..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libatlas-base-dev

# Create a python virtual environment
echo "Creating python virtual environment (venv)..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install torch (CPU-only) separately to avoid downloading the massive CUDA version
echo "Installing PyTorch (CPU-only)..."
pip install torch>=2.0.0 --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies from requirements.txt
echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

echo "========================================================="
echo "Installation completed successfully!"
echo "To run the system:"
echo "  1. Activate venv: source venv/bin/activate"
echo "  2. Run app: python main.py"
echo "========================================================="
