import sys
import logging
import time
import cv2
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QLineEdit, QStackedWidget,
                               QMessageBox, QProgressBar)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QPixmap, QPainter, QPen, QColor


from app import config
from app import database
from app import utils
from app import cleanup_enrollment_images
from app.gui.camera_thread import CameraThread

class RegistrationWindow(QWidget):
    def __init__(self, recognizer, parent=None):
        super().__init__()
        self.recognizer = recognizer
        self.parent_window = parent
        self.setWindowTitle("Register New Person")
        self.resize(800, 480)
        
        self.setStyleSheet("""
            QWidget { background-color: #1e1e2e; color: #cdd6f4; }
            QLineEdit {
                background-color: #313244; border: 2px solid #45475a;
                border-radius: 5px; padding: 10px; font-size: 16px;
            }
            QPushButton {
                background-color: #89b4fa; color: #11111b;
                border-radius: 5px; font-size: 16px; font-weight: bold; padding: 10px;
            }
            QPushButton:hover { background-color: #b4befe; }
            QPushButton:disabled { background-color: #45475a; color: #a6adc8; }
            QPushButton#btnBack { background-color: #f38ba8; }
            QPushButton#btnBack:hover { background-color: #f5c2e7; }
            QProgressBar {
                border: 2px solid #45475a;
                border-radius: 5px;
                text-align: center;
                color: #cdd6f4;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #a6e3a1;
                border-radius: 3px;
            }
        """)

        self.stacked = QStackedWidget()
        
        self.page1 = QWidget()
        self.setup_page1()
        
        self.page2 = QWidget()
        self.setup_page2()
        
        self.stacked.addWidget(self.page1)
        self.stacked.addWidget(self.page2)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.stacked)
        self.setLayout(main_layout)
        
        self.camera_thread = None
        self.person_name = ""
        self.person_id = None
        self.person_dir = None
        self.cropped_dir = None
        
        self.samples_per_pose = 12
        self.total_poses = len(config.REGISTRATION_PLAN)
        self.target_count = self.total_poses * self.samples_per_pose
        self.saved_count = 0
        
        self.state = "waiting" # waiting, countdown, capturing, complete
        self.countdown_val = 3
        self.burst_count = 0
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer_tick)

    def setup_page1(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title = QLabel("Register New Person")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter full name...")
        self.name_input.setFixedWidth(400)
        
        btn_next = QPushButton("Start Enrollment")
        btn_next.setFixedWidth(400)
        btn_next.clicked.connect(self.go_to_page2)
        
        btn_back = QPushButton("Cancel")
        btn_back.setObjectName("btnBack")
        btn_back.setFixedWidth(400)
        btn_back.clicked.connect(self.close_window)
        
        layout.addWidget(title)
        layout.addSpacing(30)
        layout.addWidget(self.name_input)
        layout.addSpacing(20)
        layout.addWidget(btn_next)
        layout.addWidget(btn_back)
        self.page1.setLayout(layout)

    def setup_page2(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        header_layout = QHBoxLayout()
        self.lbl_title = QLabel("Automated Enrollment")
        self.lbl_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(200)
        
        header_layout.addWidget(self.lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(self.progress_bar)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000; border-radius: 10px;")
        
        self.lbl_instruction = QLabel("Initializing...")
        self.lbl_instruction.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.lbl_instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_instruction.setStyleSheet("color: #f9e2af; margin-top: 10px;")
        
        self.btn_capture = QPushButton("Start Sequence")
        self.btn_capture.clicked.connect(self.start_sequence)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setObjectName("btnBack")
        btn_cancel.clicked.connect(self.close_window)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(self.btn_capture)
        
        layout.addLayout(header_layout)
        layout.addWidget(self.video_label, 1)
        layout.addWidget(self.lbl_instruction)
        layout.addLayout(btn_layout)
        self.page2.setLayout(layout)

    def go_to_page2(self):
        self.person_name = self.name_input.text().strip()
        if not self.person_name:
            QMessageBox.warning(self, "Error", "Name cannot be empty.")
            return
            
        self.person_id = database.get_or_create_person(self.person_name)
        self.person_dir = Path(config.DATASET_DIR) / self.person_name
        self.cropped_dir = Path(config.CROPPED_FACES_DIR) / self.person_name
        self.person_dir.mkdir(parents=True, exist_ok=True)
        self.cropped_dir.mkdir(parents=True, exist_ok=True)
        
        self.saved_count = 0
        self.update_progress_ui()
        self.lbl_instruction.setText("Press 'Start Sequence' when ready.")
        self.stacked.setCurrentIndex(1)
        
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.on_frame_ready)
        self.camera_thread.finished.connect(self.camera_thread.deleteLater)
        self.camera_thread.start()

    def start_sequence(self):
        self.btn_capture.setEnabled(False)
        self.start_next_pose_countdown()

    def start_next_pose_countdown(self):
        pose_index = self.saved_count // self.samples_per_pose
        if pose_index >= self.total_poses:
            self.state = "complete"
            self.complete_registration()
            return
            
        step = config.REGISTRATION_PLAN[pose_index]
        self.state = "countdown"
        self.countdown_val = 3
        
        # Display instruction immediately
        self.lbl_instruction.setText(f"{step['instruction']} (Starting in {self.countdown_val}s...)")
        self.lbl_instruction.setStyleSheet("color: #f9e2af; margin-top: 10px;")
        
        self.timer.start(1000)

    def on_timer_tick(self):
        if self.state == "countdown":
            self.countdown_val -= 1
            if self.countdown_val > 0:
                pose_index = self.saved_count // self.samples_per_pose
                step = config.REGISTRATION_PLAN[pose_index]
                self.lbl_instruction.setText(f"{step['instruction']} (Starting in {self.countdown_val}s...)")
            else:
                self.timer.stop()
                self.state = "capturing"
                self.burst_count = 0
                pose_index = self.saved_count // self.samples_per_pose
                step = config.REGISTRATION_PLAN[pose_index]
                self.lbl_instruction.setText(f"Capturing: {step['instruction']}")
                self.lbl_instruction.setStyleSheet("color: #a6e3a1; margin-top: 10px;")

    def update_progress_ui(self):
        percent = int((self.saved_count / self.target_count) * 100) if self.target_count > 0 else 0
        self.progress_bar.setValue(percent)

    def on_frame_ready(self, q_img, raw_frame, faces):
        pixmap = QPixmap.fromImage(q_img)
        w = self.video_label.width()
        h = self.video_label.height()
        
        if w > 0 and h > 0:
            pixmap = pixmap.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio)
            scale_x = pixmap.width() / raw_frame.shape[1]
            scale_y = pixmap.height() / raw_frame.shape[0]
            
            painter = QPainter(pixmap)
            
            if self.state == "capturing":
                pen = QPen(QColor(166, 227, 161)) # Green
            else:
                pen = QPen(QColor(137, 180, 250)) # Blue
                
            pen.setWidth(2)
            painter.setPen(pen)
            
            for (x, y, w_f, h_f) in faces:
                painter.drawRect(int(x * scale_x), int(y * scale_y), int(w_f * scale_x), int(h_f * scale_y))
            
            # Big visual overlay for countdown and capturing
            if self.state == "countdown":
                painter.setPen(QPen(QColor(249, 226, 175)))
                painter.setFont(QFont("Segoe UI", 48, QFont.Weight.Bold))
                painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, str(self.countdown_val))
            elif self.state == "capturing":
                painter.setPen(QPen(QColor(166, 227, 161)))
                painter.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
                painter.drawText(20, 40, f"Capturing... {self.burst_count}/{self.samples_per_pose}")
                
            painter.end()
            self.video_label.setPixmap(pixmap)
        
        # Process saving only if in capturing state
        if self.state == "capturing" and faces:
            largest_face = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            x, y, w_f, h_f = largest_face
            face_crop = raw_frame[y:y+h_f, x:x+w_f].copy()
            
            if face_crop.size > 0:
                blur_val = utils.calculate_blur(face_crop)
                if blur_val >= 30.0:
                    self.save_crop(raw_frame, face_crop)
                    self.burst_count += 1
                    self.saved_count += 1
                    self.update_progress_ui()
                    
                    if self.burst_count >= self.samples_per_pose:
                        self.state = "waiting"
                        self.start_next_pose_countdown()

    def save_crop(self, raw_frame, face_crop):
        filename = f"face_{self.saved_count+1:03d}.jpg"
        filepath = self.person_dir / filename
        cropped_filepath = self.cropped_dir / filename
        
        cv2.imwrite(str(filepath), raw_frame)
        cv2.imwrite(str(cropped_filepath), face_crop)
        
        pose_index = self.saved_count // self.samples_per_pose
        step = config.REGISTRATION_PLAN[pose_index]
        
        rel_path = str(filepath.relative_to(config.PROJECT_ROOT))
        rel_crop_path = str(cropped_filepath.relative_to(config.PROJECT_ROOT))
        
        image_id = database.save_record(
            person_id=self.person_id,
            image_path=rel_path,
            cropped_path=rel_crop_path,
            pose=step["pose"],
            expression=step["expression"],
            lighting_condition=step["lighting"],
            width=raw_frame.shape[1],
            height=raw_frame.shape[0],
            cropped_width=face_crop.shape[1],
            cropped_height=face_crop.shape[0],
            blur_score=utils.calculate_blur(face_crop),
            brightness_score=utils.calculate_brightness(face_crop),
            detector_name="MediaPipe",
            session_id="registration"
        )
        
        try:
            embedding = self.recognizer.generate_embedding(face_crop)
            database.save_embedding_record(
                image_id=image_id,
                person_id=self.person_id,
                model_name=config.RECOGNIZER_MODEL_NAME,
                embedding_vector=embedding,
                embedding_dim=len(embedding),
                detector_name="MediaPipe",
                session_id="registration"
            )
        except Exception as e:
            logging.warning(f" Failed to generate embedding: {e}")

    def complete_registration(self):
        if self.camera_thread:
            self.camera_thread.stop()
        
        self.lbl_instruction.setText("Registration Complete! Optimizing storage...")
        self.lbl_instruction.setStyleSheet("color: #a6e3a1;")
        
        if not config.KEEP_REGISTRATION_IMAGES:
            cleanup_enrollment_images.cleanup_images_for_person(self.person_id, self.person_name, dry_run=False)
            
        # Reload gallery with new person
        try:
            self.recognizer.reload_gallery()
        except Exception as e:
            logging.warning(f" Failed to reload gallery: {e}")
            
        # Add profile picture
        try:
            default_img = self.cropped_dir / "face_001.jpg"
            if default_img.exists():
                database.set_profile_image_path(self.person_id, str(default_img.relative_to(config.PROJECT_ROOT)))
        except Exception:
            pass

        QMessageBox.information(self, "Success", f"Registration complete for {self.person_name}!")
        self.close_window()

    def close_window(self):
        self.timer.stop()
        if self.camera_thread:
            self.camera_thread.stop()
            if self.parent_window:
                if not hasattr(self.parent_window, 'dying_threads'):
                    self.parent_window.dying_threads = []
                self.parent_window.dying_threads.append(self.camera_thread)
        if self.parent_window:
            self.parent_window.show()
        self.close()
