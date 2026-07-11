import sys
import time
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar  # type: ignore
from PySide6.QtCore import Qt, QThread, Signal  # type: ignore
from PySide6.QtGui import QFont  # type: ignore

# We need to make sure the app directory is available
from pathlib import Path

class ModelLoaderThread(QThread):
    finished_loading = Signal(object) # emits the initialized FaceRecognizer
    progress = Signal(str, int)

    def run(self):
        self.progress.emit("Initializing Database...", 20)
        from app import database
        time.sleep(0.2) # UI buffer
        
        self.progress.emit("Loading MediaPipe Face Detector...", 40)
        from app import face_detector
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        face_detector.detect_faces(dummy_img)
        
        self.progress.emit("Loading PyTorch FaceNet Model (This takes a moment)...", 60)
        from app.face_recognizer import FaceRecognizer
        recognizer = FaceRecognizer()
        
        self.progress.emit("Warming up inference engine...", 90)
        face_crop = np.zeros((160, 160, 3), dtype=np.uint8)
        try:
            recognizer.generate_embedding(face_crop)
        except Exception:
            pass # ignore if it fails on zeros
        
        self.progress.emit("Starting Application...", 100)
        time.sleep(0.5)
        
        self.finished_loading.emit(recognizer)

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        # Frameless and transparent background
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(500, 300)
        
        # Catppuccin Mocha theme inspired
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2e;
                border-radius: 15px;
                border: 2px solid #89b4fa;
            }
            QLabel {
                color: #cdd6f4;
                border: none;
            }
            QProgressBar {
                border: 2px solid #313244;
                border-radius: 5px;
                text-align: center;
                background-color: #181825;
                color: transparent;
            }
            QProgressBar::chunk {
                background-color: #89b4fa;
                border-radius: 3px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 40, 30, 40)
        
        title = QLabel("AI Attendance System")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        subtitle = QLabel("Initializing Embedded Environment")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #a6adc8;")
        
        self.status_label = QLabel("Loading components...")
        self.status_label.setFont(QFont("Segoe UI", 9))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #a6adc8;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setValue(0)
        
        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch(1)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
        
        self.loader_thread = ModelLoaderThread()
        self.loader_thread.progress.connect(self.update_progress)
        
    def start_loading(self):
        self.loader_thread.start()

    def update_progress(self, message, value):
        self.status_label.setText(message)
        self.progress_bar.setValue(value)
