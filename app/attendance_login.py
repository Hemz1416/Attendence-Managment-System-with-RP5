import sys
import logging
import time
from pathlib import Path
import cv2
import numpy as np
import threading
from queue import Queue, Empty

# Ensure app folder is on sys.path

from app import config
from app import database
import face_detector
from app import utils
from face_recognizer import FaceRecognizer

_EVENT_COOLDOWN_SECONDS = 60

class AttendanceManager:
    def __init__(self):
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
                        database.log_attendance(
                            person_id=person_id,
                            person_name=name,
                            recognition_score=score,
                            model_name="FaceNet",
                            detector_name="MediaPipe",
                            device="RP5"
                        )
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

    def run_cli_loop(self):
        self.start_worker()
        
        cap = cv2.VideoCapture(config.WEBCAM_INDEX)
        using_mock = False
        
        if not cap.isOpened():
            logging.warning(f" Could not open webcam index {config.WEBCAM_INDEX}.")
            using_mock = True
        else:
            ret, _ = cap.read()
            if not ret:
                logging.warning(" Camera opened but failed to read a frame.")
                cap.release()
                using_mock = True

        if using_mock:
            logging.info("Falling back to Mock Capture mode using local dataset images...")
            cap = utils.MockCapture(Path(config.PROJECT_ROOT) / "dataset", config.WEBCAM_WIDTH, config.WEBCAM_HEIGHT)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)

        logging.info("\nStarting camera stream. Press 'q' in the window to quit.")
        
        prev_time = time.time()
        prev_recognized_faces = []
        next_track_id = 0
        frame_counter = 0
        retry_count = 0
        max_retries = 3

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    if not using_mock and retry_count < max_retries:
                        retry_count += 1
                        logging.info(f"[Info] Attempting to reconnect camera (Attempt {retry_count}/{max_retries})...")
                        cap.release()
                        time.sleep(2.0)
                        cap = cv2.VideoCapture(config.WEBCAM_INDEX)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
                        continue
                    else:
                        logging.error(" Camera connection lost permanently.")
                        break
                        
                retry_count = 0
                frame_counter += 1
                curr_time = time.time()
                
                # Periodically prune stale tracks and cooldowns (every 100 frames)
                if frame_counter % 100 == 0:
                    with self.results_lock:
                        stale_keys = [k for k, v in self.recognized_results.items() if curr_time - v["last_updated"] > 30.0]
                        for k in stale_keys:
                            del self.recognized_results[k]
                    # Prune cooldowns older than 2 minutes
                    stale_events = [k for k, v in self.last_event_time.items() if curr_time - v > 120.0]
                    for k in stale_events:
                        del self.last_event_time[k]

                fps = 1.0 / (curr_time - prev_time)
                prev_time = curr_time

                try:
                    faces = face_detector.detect_faces(frame)
                except Exception as e:
                    logging.info(f"Detection error: {e}")
                    faces = []

                current_faces_data = []

                for (x, y, w, h) in faces:
                    cx, cy = x + w // 2, y + h // 2
                    
                    best_dist = 80
                    best_prev_idx = -1
                    
                    for idx, prev in enumerate(prev_recognized_faces):
                        dist = np.sqrt((cx - prev["cx"])**2 + (cy - prev["cy"])**2)
                        if dist < best_dist:
                            best_dist = dist
                            best_prev_idx = idx
                            
                    if best_prev_idx != -1:
                        prev_face = prev_recognized_faces[best_prev_idx]
                        track_id = prev_face["track_id"]
                        frames_since_recognition = prev_face["frames_since_recognition"] + 1
                    else:
                        track_id = next_track_id
                        next_track_id += 1
                        frames_since_recognition = 999 

                    with self.results_lock:
                        cache = self.recognized_results.get(track_id)
                    
                    if cache:
                        matched_name = cache["name"]
                        matched_score = cache["score"]
                    else:
                        matched_name = "Unknown"
                        matched_score = 0.0

                    if (not cache or frames_since_recognition >= 15) and (track_id not in self.pending_tracks):
                        face_crop = frame[y:y+h, x:x+w].copy()
                        if face_crop.size > 0:
                            if not self.recognition_queue.full():
                                self.pending_tracks.add(track_id)
                                self.recognition_queue.put((track_id, face_crop))
                                frames_since_recognition = 0

                    current_faces_data.append({
                        "x": x, "y": y, "w": w, "h": h,
                        "cx": cx, "cy": cy,
                        "track_id": track_id,
                        "name": matched_name,
                        "score": matched_score,
                        "frames_since_recognition": frames_since_recognition
                    })

                    if matched_name != "Unknown":
                        color = (0, 255, 0)
                        status_text = "MATCH"
                        display_name = matched_name
                    else:
                        color = (0, 0, 255)
                        status_text = "NO MATCH"
                        display_name = "Unknown Person"

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{display_name}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, f"Similarity: {matched_score:.2f}", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, f"Status: {status_text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                prev_recognized_faces = current_faces_data
                overlay_text = f"FPS: {fps:.1f} | Detector: MediaPipe | Recognizer: FaceNet"
                cv2.putText(frame, overlay_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow("Raspberry Pi 5 Attendance System", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.stop_worker()
            cap.release()
            cv2.destroyAllWindows()

def main():
    logging.info("\n" + "=" * 60)
    logging.info("                2. ATTENDANCE LOGIN WORKFLOW")
    logging.info("=" * 60)
    try:
        manager = AttendanceManager()
        manager.run_cli_loop()
    except Exception as e:
        logging.info(f"Attendance login failed: {e}")

if __name__ == "__main__":
    main()
