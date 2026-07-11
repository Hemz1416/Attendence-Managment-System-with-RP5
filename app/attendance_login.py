import logging
import time
import threading
from queue import Queue, Empty

from app import config
from app import database
from app import utils
from app.face_recognizer import FaceRecognizer

_EVENT_COOLDOWN_SECONDS = 60

class AttendanceManager:
    def __init__(self, recognizer=None):
        if recognizer is not None:
            self.recognizer = recognizer
        else:
            try:
                self.recognizer = FaceRecognizer()
            except Exception as e:
                logging.info(f"ERROR: Failed to initialize face recognizer: {e}")
                raise

        if not self.recognizer.gallery:
            logging.warning("\n No enrolled FaceNet embeddings found in database.")
            logging.info("The system will function, but all faces will be detected as 'Unknown'.")
        
        self.recognition_queue = Queue(maxsize=3)
        self.recognized_results = {}  # track_id -> {"name": str, "score": float, "person_id": int/None, "last_updated": float}
        self.pending_tracks = set()   # track_ids currently in queue
        self.results_lock = threading.Lock()
        self.last_event_time = {}     # person_key -> timestamp (cooldown tracker)
        
        self.running = False
        self.worker_thread = None

    def _recognition_worker(self):
        while self.running:
            try:
                item = self.recognition_queue.get(timeout=0.5)
            except Empty:
                continue
                
            if item is None:
                break
                
            track_id, face_crop = item
            try:
                name, score, person_id = self.recognizer.recognize_face(face_crop)

                with self.results_lock:
                    self.recognized_results[track_id] = {
                        "name": name,
                        "score": score,
                        "person_id": person_id,
                        "last_updated": time.time()
                    }
                
                # Cooldown: only log event if enough time has passed for this person
                now = time.time()
                event_key = person_id if person_id is not None else f"unknown_{track_id}"
                should_log = (now - self.last_event_time.get(event_key, 0)) >= _EVENT_COOLDOWN_SECONDS

                if should_log:
                    self.last_event_time[event_key] = now
                    database.log_recognition_event(
                        person_id=person_id,
                        person_name=name,
                        similarity_score=score,
                        threshold_used=config.DEFAULT_THRESHOLD,
                        detector_name="MediaPipe",
                        recognizer_name="FaceNet"
                    )
                    
                    if person_id is not None:
                        logging.info(f"[Attendance] Recognized and logged check-in: '{name}' (Similarity: {score:.3f})")
                        database.log_attendance(
                            person_id=person_id,
                            person_name=name,
                            recognition_score=score,
                            model_name="FaceNet",
                            detector_name="MediaPipe",
                            device="RP5"
                        )
                    else:
                        logging.info(f"[Attendance] Unknown face detected (Similarity: {score:.3f})")
            except Exception as e:
                logging.warning(f" Error in recognition worker: {e}")
            finally:
                self.pending_tracks.discard(track_id)
                self.recognition_queue.task_done()

    def start_worker(self):
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._recognition_worker, daemon=True)
            self.worker_thread.start()

    def stop_worker(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
            if self.worker_thread.is_alive():
                logging.warning(" Worker thread did not terminate cleanly.")
