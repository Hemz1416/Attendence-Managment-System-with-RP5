#!/bin/bash
# run_kiosk.sh - Launcher for Raspberry Pi 5
# This script is intended to be run by systemd

cd "$(dirname "$0")"

# Activate the virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Export Display variables for Qt
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb
# export WAYLAND_DISPLAY=wayland-1 # Uncomment if using wayland explicitly

# Run the PySide6 Application
python main.py
