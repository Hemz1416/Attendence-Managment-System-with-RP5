import sys
import time
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,  # type: ignore
                               QPushButton, QLabel, QMessageBox)
from PySide6.QtCore import Qt, QTimer  # type: ignore
from PySide6.QtGui import QFont, QPixmap, QPainter, QPen, QColor  # type: ignore


from app import database
from app.attendance_login import AttendanceManager
from app.gui.camera_thread import CameraThread

class AttendanceWindow(QWidget):
    def __init__(self, recognizer, parent=None):
        super().__init__()
        self.parent_window = parent
        self.setWindowTitle("Attendance Kiosk")
        self.resize(1024, 600)
        
        self.setStyleSheet("""
            QWidget { background-color: #1e1e2e; color: #cdd6f4; }
            QTableWidget {
                background-color: #181825;
                alternate-background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #313244;
                gridline-color: #313244;
            }
            QHeaderView::section {
                background-color: #313244;
                color: #cdd6f4;
                font-weight: bold;
                border: 1px solid #1e1e2e;
                padding: 4px;
            }
            QPushButton {
                background-color: #f38ba8; color: #11111b;
                border-radius: 5px; font-size: 14px; font-weight: bold; padding: 10px;
            }
            QPushButton:hover { background-color: #f5c2e7; }
        """)

        # Manager for AI logic
        self.manager = AttendanceManager(recognizer=recognizer)
        self.manager.start_worker()

        self.setup_ui()
        
        # State
        self.prev_time = time.time()
        self.prev_recognized_faces = []
        self.next_track_id = 0
        self.frame_counter = 0
        
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.on_frame_ready)
        self.camera_thread.error_signal.connect(self.on_camera_error)
        self.camera_thread.finished.connect(self.camera_thread.deleteLater)
        self.camera_thread.start()

    def on_camera_error(self, message):
        QMessageBox.critical(self, "Camera Error", f"Unable to start the video capture: {message}")
        self.close_window()

    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)
        
        title = QLabel("Attendance Mode")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #cdd6f4;")
        
        subtitle = QLabel("Please look at the camera to check in")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #a6adc8;")
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; border: 2px solid #45475a; border-radius: 10px;")
        self.video_label.setMinimumSize(640, 480)
        
        bottom_layout = QHBoxLayout()
        self.lbl_fps = QLabel("FPS: 0.0")
        self.lbl_fps.setStyleSheet("color: #a6adc8; font-size: 14px;")
        
        btn_back = QPushButton("Exit")
        btn_back.clicked.connect(self.close_window)
        btn_back.setFixedWidth(200)
        btn_back.setStyleSheet("""
            QPushButton {
                background-color: #f38ba8; color: #11111b;
                border-radius: 8px; font-size: 16px; font-weight: bold; padding: 12px;
            }
            QPushButton:hover { background-color: #f5c2e7; }
        """)
        
        bottom_layout.addWidget(self.lbl_fps)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(btn_back)
        
        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)
        main_layout.addWidget(self.video_label, 1, Qt.AlignmentFlag.AlignCenter)
        main_layout.addLayout(bottom_layout)
        
        self.setLayout(main_layout)

    def on_frame_ready(self, q_img, raw_frame, faces):
        curr_time = time.time()
        time_diff = curr_time - self.prev_time
        fps = 1.0 / time_diff if time_diff > 0 else 0.0
        self.prev_time = curr_time
        self.lbl_fps.setText(f"FPS: {fps:.1f}")
        
        self.frame_counter += 1
        
        if self.frame_counter % 100 == 0:
            with self.manager.results_lock:
                stale_keys = [k for k, v in self.manager.recognized_results.items() if curr_time - v["last_updated"] > 30.0]
                for k in stale_keys:
                    del self.manager.recognized_results[k]
                stale_events = [k for k, v in self.manager.last_event_time.items() if curr_time - v > 120.0]
                for k in stale_events:
                    del self.manager.last_event_time[k]

        current_faces_data = []
        
        pixmap = QPixmap.fromImage(q_img)
        w = self.video_label.width()
        h = self.video_label.height()
        
        if w > 0 and h > 0:
            pixmap = pixmap.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio)
            scale_x = pixmap.width() / raw_frame.shape[1]
            scale_y = pixmap.height() / raw_frame.shape[0]
            
            painter = QPainter(pixmap)
            painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))

            for (x, y, w_f, h_f) in faces:
                cx, cy = x + w_f // 2, y + h_f // 2
                
                best_dist = 80
                best_prev_idx = -1
                
                for idx, prev in enumerate(self.prev_recognized_faces):
                    dist = np.sqrt((cx - prev["cx"])**2 + (cy - prev["cy"])**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_prev_idx = idx
                        
                if best_prev_idx != -1:
                    prev_face = self.prev_recognized_faces[best_prev_idx]
                    track_id = prev_face["track_id"]
                    frames_since_rec = prev_face["frames_since_rec"] + 1
                else:
                    track_id = self.next_track_id
                    self.next_track_id += 1
                    frames_since_rec = 999
                    
                with self.manager.results_lock:
                    cache = self.manager.recognized_results.get(track_id)
                    
                if cache:
                    matched_name = cache["name"]
                    matched_score = cache["score"]
                else:
                    matched_name = "Unknown"
                    matched_score = 0.0
                    
                if (not cache or frames_since_rec >= 15) and (track_id not in self.manager.pending_tracks):
                    face_crop = raw_frame[y:y+h_f, x:x+w_f].copy()
                    if face_crop.size > 0:
                        if not self.manager.recognition_queue.full():
                            self.manager.pending_tracks.add(track_id)
                            self.manager.recognition_queue.put((track_id, face_crop))
                            frames_since_rec = 0
                            
                current_faces_data.append({
                    "cx": cx, "cy": cy,
                    "track_id": track_id,
                    "frames_since_rec": frames_since_rec
                })
                
                # Draw
                draw_x = int(x * scale_x)
                draw_y = int(y * scale_y)
                draw_w = int(w_f * scale_x)
                draw_h = int(h_f * scale_y)
                
                if matched_name != "Unknown":
                    color = QColor(166, 227, 161) # Green
                    text = f"{matched_name} ({matched_score:.2f})"
                else:
                    color = QColor(243, 139, 168) # Red
                    text = "Unknown"
                    
                pen = QPen(color)
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawRect(draw_x, draw_y, draw_w, draw_h)
                painter.drawText(draw_x, draw_y - 10, text)

            painter.end()
            self.video_label.setPixmap(pixmap)
            
        self.prev_recognized_faces = current_faces_data



    def close_window(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.manager.stop_worker()
        if self.parent_window:
            self.parent_window.show()
        self.close()
