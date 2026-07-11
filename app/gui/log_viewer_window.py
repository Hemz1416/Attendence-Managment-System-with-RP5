import sys
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox)  # type: ignore
from PySide6.QtCore import Qt  # type: ignore
from PySide6.QtGui import QFont  # type: ignore
from app import database

class LogViewerWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent_window = parent
        self.setWindowTitle("Attendance Log Viewer")
        self.resize(800, 480)
        
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
                background-color: #313244;
                color: #cdd6f4;
                border: 2px solid #45475a;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #45475a;
                border: 2px solid #89b4fa;
            }
        """)
        
        self.setup_ui()
        self.refresh_logs()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        
        title = QLabel("Attendance Logs")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("margin-bottom: 10px;")
        
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Date", "Time", "Name", "Score"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        btn_back = QPushButton("Back")
        btn_back.clicked.connect(self.close_window)
        btn_back.setFixedWidth(150)
        
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(btn_back)
        
        layout.addWidget(title)
        layout.addWidget(self.table, 1)
        layout.addLayout(bottom_layout)
        self.setLayout(layout)
        
    def refresh_logs(self):
        try:
            logs = database.get_all_logs(limit=1000)
            self.table.setRowCount(0)
            for row, log in enumerate(logs):
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(log.get("date", "-")))
                self.table.setItem(row, 1, QTableWidgetItem(log.get("time", "-")))
                self.table.setItem(row, 2, QTableWidgetItem(log.get("person_name", "Unknown")))
                score = log.get("recognition_score")
                score_str = f"{score:.2f}" if score is not None else "-"
                self.table.setItem(row, 3, QTableWidgetItem(score_str))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load logs: {e}")
            
    def close_window(self):
        if self.parent_window:
            self.parent_window.show()
        self.close()
