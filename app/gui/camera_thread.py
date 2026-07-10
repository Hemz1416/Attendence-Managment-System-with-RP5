import sys
import logging
import time
import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
import numpy as np
from pathlib import Path


from app import config
from app import utils
import face_detector

class CameraThread(QThread):
    # Signals
    frame_ready = Signal(QImage, np.ndarray, list) # rgb_image, raw_bgr_frame, faces
    error_signal = Signal(str)

    def __init__(self, use_mock_fallback=True):
        super().__init__()
        self.running = False
        self.cap = None
        self.use_mock_fallback = use_mock_fallback

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(config.WEBCAM_INDEX)
        using_mock = False
        
        if not self.cap.isOpened():
            logging.warning(f" Could not open webcam index {config.WEBCAM_INDEX}.")
            using_mock = True
        else:
            ret, _ = self.cap.read()
            if not ret:
                logging.warning(" Camera opened but failed to read a frame.")
                self.cap.release()
                using_mock = True

        if using_mock and self.use_mock_fallback:
            logging.info("Falling back to Mock Capture mode using local dataset images...")
            self.cap = utils.MockCapture(Path(config.PROJECT_ROOT) / "dataset", config.WEBCAM_WIDTH, config.WEBCAM_HEIGHT)
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)

        if not self.cap.isOpened():
            self.error_signal.emit("Failed to open any camera.")
            return

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Detect faces
            try:
                faces = face_detector.detect_faces(frame)
            except Exception as e:
                faces = []
                
            # Convert to QImage
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            
            self.frame_ready.emit(q_img, frame, faces)
            
            # ~30 fps cap
            time.sleep(1/30.0)
            
        self.cap.release()

    def stop(self):
        self.running = False
