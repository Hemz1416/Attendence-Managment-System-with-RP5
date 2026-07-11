import sys
import subprocess
from pathlib import Path
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox, QFileDialog, QInputDialog, QLineEdit  # type: ignore
from PySide6.QtCore import Qt  # type: ignore
from PySide6.QtGui import QFont  # type: ignore
from app.gui.registration_window import RegistrationWindow
from app.gui.attendance_window import AttendanceWindow
from app import database

class MainWindow(QMainWindow):
    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer
        self.setWindowTitle("RP5 AI Attendance System")
        # Fixed size for typical Raspberry Pi 7-inch touchscreen
        self.resize(800, 480) 
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QLabel {
                color: #cdd6f4;
            }
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 2px solid #45475a;
                border-radius: 10px;
                font-size: 18px;
                font-weight: bold;
                padding: 20px;
            }
            QPushButton:hover {
                background-color: #45475a;
                border: 2px solid #89b4fa;
            }
            QPushButton:pressed {
                background-color: #585b70;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(50, 50, 50, 50)
        
        title = QLabel("AI Attendance System")
        title.setFont(QFont("Segoe UI", 32, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        subtitle = QLabel("Select an operation mode")
        subtitle.setFont(QFont("Segoe UI", 14))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #a6adc8; margin-bottom: 30px;")
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(30)
        
        self.btn_register = QPushButton("Register New Person")
        self.btn_attendance = QPushButton("Attendance")
        
        button_layout.addWidget(self.btn_register)
        button_layout.addWidget(self.btn_attendance)
        
        self.btn_register.clicked.connect(self.open_registration)
        self.btn_attendance.clicked.connect(self.open_attendance)
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(button_layout)
        
        admin_title = QLabel("Admin Tasks")
        admin_title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        admin_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        admin_title.setStyleSheet("color: #a6adc8; margin-top: 30px; margin-bottom: 10px;")
        
        admin_layout = QHBoxLayout()
        admin_layout.setSpacing(20)
        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_json = QPushButton("Export JSON")
        self.btn_clear_logs = QPushButton("Clear Logs")
        
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_json.clicked.connect(self.export_json)
        self.btn_clear_logs.clicked.connect(self.clear_logs)
        
        admin_layout.addWidget(self.btn_export_csv)
        admin_layout.addWidget(self.btn_export_json)
        admin_layout.addWidget(self.btn_clear_logs)
        
        layout.addWidget(admin_title)
        layout.addLayout(admin_layout)
        
        layout.addStretch(1)
        
        status_layout = QHBoxLayout()
        self.status_db = QLabel("Database: OK")
        self.status_db.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        self.status_model = QLabel("Models: LOADED")
        self.status_model.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        
        status_layout.addWidget(self.status_db)
        status_layout.addStretch(1)
        status_layout.addWidget(self.status_model)
        
        layout.addLayout(status_layout)
        central_widget.setLayout(layout)

    def open_registration(self):
        self.reg_window = RegistrationWindow(self.recognizer, parent=self)
        self.reg_window.show()
        self.hide()

    def open_attendance(self):
        self.att_window = AttendanceWindow(self.recognizer, parent=self)
        self.att_window.show()
        self.hide()

    def check_admin_password(self):
        password, ok = QInputDialog.getText(self, "Admin Auth", "Enter Password:", QLineEdit.EchoMode.Password)
        if ok and password == "siriAB":
            return True
        elif ok:
            QMessageBox.warning(self, "Auth Failed", "Incorrect password.")
        return False

    def export_csv(self):
        if not self.check_admin_password():
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV Files (*.csv)")
        if filename:
            try:
                database.export_logs_csv(filename)
                QMessageBox.information(self, "Success", f"Logs exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export CSV: {e}")

    def export_json(self):
        if not self.check_admin_password():
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Export JSON", "", "JSON Files (*.json)")
        if filename:
            try:
                database.export_logs_json(filename)
                QMessageBox.information(self, "Success", f"Logs exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export JSON: {e}")

    def clear_logs(self):
        if not self.check_admin_password():
            return
        reply = QMessageBox.question(self, "Clear Logs", "Are you sure you want to clear ALL attendance logs?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                database.clear_all_logs()
                QMessageBox.information(self, "Success", "Logs cleared successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to clear logs: {e}")
