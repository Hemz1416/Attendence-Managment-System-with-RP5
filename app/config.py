import os
from pathlib import Path

# Paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

DB_PATH = str(PROJECT_ROOT / "database" / "attendance.db")
DATASET_DIR = PROJECT_ROOT / "dataset"
CROPPED_FACES_DIR = DATASET_DIR / "cropped_face"
MODELS_DIR = PROJECT_ROOT / "models"
MEDIAPIPE_MODEL_PATH = str(MODELS_DIR / "face_detection_full_range.tflite")

# Face Detection Settings
MIN_DETECTION_CONFIDENCE = 0.5

# Face Recognition Settings
RECOGNIZER_MODEL_NAME = "FaceNet"
SIMILARITY_METRIC = "cosine"
DEFAULT_THRESHOLD = 0.60  # Default FaceNet similarity threshold
FACENET_INPUT_SIZE = (160, 160)

# Webcam Settings
WEBCAM_INDEX = 0
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480

# Registration Settings
REGISTRATION_PLAN = [
    {"pose": "Straight", "expression": "Neutral", "lighting": "Normal", "instruction": "Look straight at the camera, keep face neutral."},
    {"pose": "Straight", "expression": "Smile", "lighting": "Normal", "instruction": "Look straight at the camera and smile slightly."},
    {"pose": "Turn Left", "expression": "Neutral", "lighting": "Normal", "instruction": "Turn your head slightly to the left (30 deg)."},
    {"pose": "Turn Right", "expression": "Neutral", "lighting": "Normal", "instruction": "Turn your head slightly to the right (30 deg)."},
    {"pose": "Look Up", "expression": "Neutral", "lighting": "Normal", "instruction": "Look slightly upwards."},
    {"pose": "Look Down", "expression": "Neutral", "lighting": "Normal", "instruction": "Look slightly downwards."},
    {"pose": "Straight", "expression": "Neutral", "lighting": "Dim Light", "instruction": "Move to a dimmer lighting condition or shadow."}
]

# Storage Optimization
KEEP_REGISTRATION_IMAGES = False  # If False, deletes all but one best image after registration
