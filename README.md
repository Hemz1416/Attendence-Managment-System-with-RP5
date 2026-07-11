# Raspberry Pi 5 Attendance System Deployment Package

This folder contains the production-ready attendance and registration system optimized for deployment on Raspberry Pi 5 running DietPi OS. All development, testing, benchmarking, and experimental models/scripts have been removed to minimize memory and storage footprints.

## Architecture & Deployment Stack

- **Face Detection**: MediaPipe (TFLite)
- **Face Recognition**: FaceNet (PyTorch model pre-trained on VGGFace2)
- **Database**: SQLite (attendance logging and embedding gallery storage)
- **Target Platform**: Raspberry Pi 5 (8GB / 4GB) running DietPi or Raspberry Pi OS (64-bit)

---

## Directory Structure

```text
RP5/
├── app/
│   ├── register_person.py    # Workflow 1: New user registration script
│   ├── attendance_login.py   # Workflow 2: Daily attendance login script
│   ├── face_detector.py      # MediaPipe face detection wrapper
│   ├── face_recognizer.py    # FaceNet embedding generation and database search
│   ├── database.py           # SQLite database interactions and event logger
│   ├── config.py             # Central configuration (thresholds, webcam indices, etc.)
│   └── utils.py              # Cosine similarity, matching, and MockCapture helpers
│
├── database/
│   └── attendance.db         # Enrolled profiles and recognition events logs
│
├── dataset/                  # Folder to store reference dataset (frontal & cropped images)
│
├── models/
│   └── face_detection_full_range.tflite  # MediaPipe detector weights
│
├── validate_deployment.py    # RP5 performance latency benchmark
├── reset_deployment.py       # Script to safely clear DB and dataset
├── main.py                   # Central launcher script (console menu)
├── requirements.txt          # Minimal required Python packages
├── setup_rpi.sh              # Setup script to prepare the environment
└── README.md                 # This documentation
```

---

## Installation & Setup on Raspberry Pi

### 1. Prerequisites
- **OS**: DietPi OS (64-bit) or Raspberry Pi OS (64-bit) is recommended for Raspberry Pi 5.
- **Hardware**: USB webcam or Raspberry Pi Camera Module.

### 2. Auto-Installation
Run the installation script to update the package list, install system libraries (needed by OpenCV), create a Python virtual environment, and install dependencies:

```bash
chmod +x setup_rpi.sh
./setup_rpi.sh
```

### 3. Manual Installation (Alternative)
If you prefer to install packages manually:
```bash
# Update and install system dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv libgl1-mesa-glx libglib2.0-0 libgomp1 libatlas-base-dev

# Initialize and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install python requirements (Torch CPU-only)
pip install --upgrade pip
pip install torch>=2.0.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## First Run Requirements
- **Internet Connection**: The first time the system runs `FaceRecognizer` (e.g. during registration or attendance login), it must download the `vggface2` weights (~110 MB) for the FaceNet model. The system must have internet access for this initial run. After the weights are downloaded to `~/.cache/torch/checkpoints/`, the system can run fully offline.
- **Offline Deployment**: If the Raspberry Pi cannot connect to the internet, run the system on a connected machine first, then manually copy the `.cache/torch/checkpoints` folder to the target device.

## Deployment Reset
A utility script `reset_deployment.py` is included to easily clear all dataset images and reset the SQLite database. Use this to remove test/development enrollments before deploying to production.
```bash
python reset_deployment.py --confirm
```

## Storage Requirements
- **PyTorch (CPU-only)**: ~200-250 MB
- **FaceNet Model Weights**: ~110 MB
- **MediaPipe Model Weights**: < 5 MB
- **Dataset Storage**: Minimal if `KEEP_REGISTRATION_IMAGES = False` is used in `config.py` (keeps 1 image and DB embeddings per person).

---

## How to Run

1. **Activate the Virtual Environment**:
   ```bash
   source venv/bin/activate
   ```
2. **Start the Launcher Menu**:
   Ensure you run the command from the root of the `RP5` directory:
   ```bash
   python main.py
   ```
3. **Using the GUI**:
   When started, the application presents a graphical dashboard with two primary modes:
   - **Register New Person**: Opens the registration flow where you enter a name and follow on-screen instructions to capture face angles automatically.
   - **Attendance**: Opens the live kiosk window for attendance tracking, displaying live recognition results and a log of today's attendance.
   - **Admin Tasks**: Includes buttons to export attendance logs to CSV/JSON, clear logs, and reset the database.

4. **Exiting**:
   Use the "Exit" buttons provided in the UI to return to the main dashboard or close the application.

---

## Configuration & Tuning

All parameters are configured in `app/config.py`:
- **Similarity Threshold**: Adjusted via `DEFAULT_THRESHOLD`. The default is `0.60` for FaceNet (cosine similarity).
- **Webcam Index**: If your webcam is not detected, change `WEBCAM_INDEX` (default: `0`).
- **Resolution**: Modify `WEBCAM_WIDTH` and `WEBCAM_HEIGHT` to match your camera specifications (default: `640x480` for optimum frame rates on CPU).

## Troubleshooting

- **"MediaPipe Face Detector not found"**: Ensure the `models/` directory contains `face_detection_full_range.tflite`.
- **"FaceNet vggface2 weights not found"**: See "First Run Requirements". The Raspberry Pi needs internet access to download the weights on first run.
- **Low FPS or High Latency**:
  - Run `python validate_deployment.py` to benchmark the system on the device.
  - Ensure you installed the **CPU-only** PyTorch wheel, as specified in `setup_rpi.sh` or `requirements.txt`.
- **Webcam Errors**: Ensure the USB webcam is securely connected. Check `WEBCAM_INDEX` in `app/config.py` (try changing from `0` to `1` or `2`).

# Attendence-Managment-System-with-RP5