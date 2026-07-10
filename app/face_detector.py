import cv2
import logging
import numpy as np
from pathlib import Path
from app import config
import mediapipe as mp
import threading
import sys
import platform


# Lazy load MediaPipe task options
_detector = None
_mp_apis_loaded = False
_init_lock = threading.Lock()

# Globals to hold the resolved APIs
mp_BaseOptions = None
mp_FaceDetector = None
mp_FaceDetectorOptions = None
mp_RunningMode = None
mp_Image = None
mp_ImageFormat = None

def _load_mp_apis():
    """Dynamically load MediaPipe APIs with fallbacks for cross-platform compatibility."""
    global _mp_apis_loaded
    global mp_BaseOptions, mp_FaceDetector, mp_FaceDetectorOptions
    global mp_RunningMode, mp_Image, mp_ImageFormat

    if _mp_apis_loaded:
        return

    # 1. BaseOptions
    try:
        # Public API (0.10.35+)
        mp_BaseOptions = mp.tasks.BaseOptions
    except (AttributeError, ImportError):
        # Fallback (0.10.18 or older structures)
        from mediapipe.tasks.python.core import base_options
        mp_BaseOptions = base_options.BaseOptions

    # 2. FaceDetector and Options
    try:
        # Public API
        mp_FaceDetector = mp.tasks.vision.FaceDetector
        mp_FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    except (AttributeError, ImportError):
        # Fallback
        from mediapipe.tasks.python.vision import face_detector
        mp_FaceDetector = face_detector.FaceDetector
        mp_FaceDetectorOptions = face_detector.FaceDetectorOptions

    # 3. RunningMode
    try:
        # Public API
        mp_RunningMode = mp.tasks.vision.RunningMode
    except (AttributeError, ImportError):
        # Fallback resolving
        try:
            from mediapipe.tasks.python.vision import face_detector
            if hasattr(face_detector, 'RunningMode'):
                mp_RunningMode = face_detector.RunningMode
            elif hasattr(face_detector, '_RunningMode'):
                mp_RunningMode = face_detector._RunningMode
        except (AttributeError, ImportError):
            pass
            
    if mp_RunningMode is None:
        raise RuntimeError("Could not resolve MediaPipe RunningMode")

    # 4. Image and ImageFormat
    try:
        # Public API (0.10.35+)
        mp_Image = mp.Image
        mp_ImageFormat = mp.ImageFormat
    except (AttributeError, ImportError):
        # Fallback to direct mediapipe package imports
        from mediapipe import Image
        from mediapipe import ImageFormat
        
        mp_Image = Image
        mp_ImageFormat = ImageFormat

    _mp_apis_loaded = True

def get_detector():
    global _detector
    if _detector is not None:
        return _detector

    with _init_lock:
        # Double-check inside lock
        if _detector is not None:
            return _detector

        # Ensure APIs are loaded first
        _load_mp_apis()

        # Identify candidate model names (order with configured first)
        configured_name = Path(config.MEDIAPIPE_MODEL_PATH).name
        candidates = [configured_name]
        for name in [
            "face_detection_full_range.tflite", 
            "face_detection_short_range.tflite", 
            "face_detection_mobile.tflite",
            "blaze_face_short_range.tflite",
            "blaze_face_full_range.tflite"
        ]:
            if name not in candidates:
                candidates.append(name)

        # Locate folders to search
        search_paths = [
            Path(config.MEDIAPIPE_MODEL_PATH).parent,
            Path("."),
            Path("models"),
            Path("../models"),
            Path(__file__).resolve().parent.parent / "models",
            Path(__file__).resolve().parent.parent.parent / "models"
        ]

        found_files = []
        # Add absolute path of configured model if it exists
        if Path(config.MEDIAPIPE_MODEL_PATH).exists():
            found_files.append(Path(config.MEDIAPIPE_MODEL_PATH))

        # Search through paths
        for base in search_paths:
            if base.exists() and base.is_dir():
                for name in candidates:
                    p = base / name
                    if p.exists() and p not in found_files:
                        found_files.append(p)

        # As a last resort, check inside python package directory
        try:
            for p in Path(mp.__file__).parent.rglob("*.tflite"):
                if "face_detection" in p.name or "blaze_face" in p.name:
                    if p not in found_files:
                        found_files.append(p)
        except Exception:
            pass

        if not found_files:
            mp_version = getattr(mp, '__version__', 'unknown')
            logging.info(f"Error: No MediaPipe face detection model files found on this system.", file=sys.stderr)
            logging.info(f"MediaPipe version: {mp_version}", file=sys.stderr)
            logging.info(f"Python version: {sys.version}", file=sys.stderr)
            logging.info(f"Detected architecture: {platform.machine()}", file=sys.stderr)
            raise RuntimeError("No MediaPipe face detection model files found on this system.")

        # Prepare dry run frame to test compatibility (using 640x480 webcam resolution for realistic tensor allocation)
        dry_run_img = mp_Image(
            image_format=mp_ImageFormat.SRGB,
            data=np.zeros((480, 640, 3), dtype=np.uint8)
        )

        last_err = None
        for candidate in found_files:
            try:
                base_options = mp_BaseOptions(model_asset_path=str(candidate))
                options = mp_FaceDetectorOptions(
                    base_options=base_options,
                    running_mode=mp_RunningMode.IMAGE,
                    min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                )
                detector = mp_FaceDetector.create_from_options(options)
                
                # Dry-run detection to check if it throws metadata/float32 issues
                try:
                    detector.detect(dry_run_img)
                except RuntimeError as e:
                    if "Feedback manager" not in str(e):
                        raise
                
                _detector = detector
                # Retrieve version dynamically, defaulting to 'unknown' if not present
                mp_version = getattr(mp, '__version__', 'unknown')
                logging.info(f"MediaPipe Face Detector (API v{mp_version}) initialized successfully using model: {candidate}")
                
                
                return _detector
            except Exception as e:
                logging.warning(f" MediaPipe model candidate verification failed for {candidate}: {e}")
                last_err = e

        # If we get here, no model worked
        mp_version = getattr(mp, '__version__', 'unknown')
        logging.info(f"Failed to initialize MediaPipe Face Detector.", file=sys.stderr)
        logging.info(f"MediaPipe version: {mp_version}", file=sys.stderr)
        logging.info(f"Python version: {sys.version}", file=sys.stderr)
        logging.info(f"Detected architecture: {platform.machine()}", file=sys.stderr)
        logging.info(f"Models attempted: {[str(p) for p in found_files]}", file=sys.stderr)
        raise RuntimeError("Could not find any functional MediaPipe face detection model.") from last_err

def detect_faces(frame):
    """Detect faces using MediaPipe.
    
    Args:
        frame: BGR image (numpy array)
        
    Returns:
        list of bounding boxes [(x, y, w, h), ...]
    """
    # get_detector() ensures _load_mp_apis() has been called
    detector = get_detector()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image_instance = mp_Image(
        image_format=mp_ImageFormat.SRGB,
        data=rgb
    )
    res = detector.detect(mp_image_instance)
    if not res or not res.detections:
        return []

    faces = []
    for det in res.detections:
        box = det.bounding_box
        x, y, w, h = box.origin_x, box.origin_y, box.width, box.height
        
        # Clamp bounding boxes within frame boundaries
        x = max(0, int(x))
        y = max(0, int(y))
        w = min(frame.shape[1] - x, int(w))
        h = min(frame.shape[0] - y, int(h))
        
        if w > 10 and h > 10:
            faces.append((x, y, w, h))
            
    return faces

