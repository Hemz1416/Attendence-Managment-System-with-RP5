import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Setup logging
log_file = Path(__file__).resolve().parent / "attendance.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3),
        logging.StreamHandler(sys.stdout)
    ]
)

# Add the app directory to sys.path so modules can find sibling imports
app_dir = Path(__file__).resolve().parent / "app"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

import signal
from PySide6.QtWidgets import QApplication  # type: ignore
from PySide6.QtCore import QTimer  # type: ignore
from app.gui.splash_screen import SplashScreen
from app.gui.main_window import MainWindow

def main():
    from app import database
    database.init_tables()
    
    app = QApplication(sys.argv)
    
    # Graceful shutdown handler for service lifecycle signals (SIGINT, SIGTERM)
    def handle_signal(sig, frame):
        logging.info(f"Received signal {sig}. Shutting down application gracefully...")
        app.quit()
        
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Heartbeat timer to periodically release the Python interpreter lock
    # enabling prompt signal delivery while the Qt event loop is running.
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)
    app.timer = timer  # Keep reference alive
    
    # Initialize the animated splash screen
    splash = SplashScreen()
    splash.show()
    
    # Container to keep the main window reference alive
    main_win = []
    
    def on_models_loaded(recognizer):
        # Models are loaded, close splash and open main dashboard
        splash.close()
        window = MainWindow(recognizer)
        main_win.append(window)
        window.show()
        
    # Connect the loader thread's finish signal to our callback
    splash.loader_thread.finished_loading.connect(on_models_loaded)
    
    # Start background loading
    splash.start_loading()
    
    # Start Qt Event Loop
    sys.exit(app.exec())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nExiting launcher.")
