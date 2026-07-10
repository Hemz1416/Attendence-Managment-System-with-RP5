import sys
import logging
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure app folder is on sys.path

from app import config
from app import database
import face_detector
from app import utils
from app import cleanup_enrollment_images
from face_recognizer import FaceRecognizer

class RegistrationManager:
    def __init__(self, name):
        self.name = name.strip()
        if not self.name:
            raise ValueError("Name cannot be empty")
            
        self.person_id = database.get_or_create_person(self.name)
        
        self.person_dir = Path(config.DATASET_DIR) / self.name
        self.cropped_dir = Path(config.CROPPED_FACES_DIR) / self.name
        self.person_dir.mkdir(parents=True, exist_ok=True)
        self.cropped_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.recognizer = FaceRecognizer()
        except Exception as e:
            logging.info(f"ERROR: Failed to load FaceNet model: {e}")
            raise
            
        self.samples_per_pose = 12
        self.total_poses = len(config.REGISTRATION_PLAN)
        self.target_count = self.total_poses * self.samples_per_pose
        self.saved_count = 0

    def run_cli_loop(self):
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

        logging.info(f"\nRegistering: {self.name}")
        logging.info("Follow the on-screen instructions for poses and expressions.")
        logging.info("Press SPACE or 'c' to capture a frame. Press 'q' to cancel.")
        logging.info("-" * 60)
        
        try:
            while self.saved_count < self.target_count:
                ret, frame = cap.read()
                if not ret:
                    logging.info("Failed to read frame.")
                    break
                
                pose_index = self.saved_count // self.samples_per_pose
                current_step = config.REGISTRATION_PLAN[pose_index]
                instruction = current_step["instruction"]
                pose = current_step["pose"]
                expr = current_step["expression"]
                light = current_step["lighting"]

                try:
                    faces = face_detector.detect_faces(frame)
                except Exception as e:
                    logging.info(f"Detection error: {e}")
                    faces = []
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
                cv2.rectangle(overlay, (0, frame.shape[0] - 80), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            
                alpha = 0.6 
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                cv2.putText(
                    frame,
                    f"Registering: {self.name}  |  Pose {pose_index + 1} of {self.total_poses} ({self.saved_count}/{self.target_count} crops)",
                    (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    f"Pose: {pose}  |  Expression: {expr}  |  Lighting: {light}",
                    (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
            
                cv2.putText(
                    frame,
                    f"Instruction: {instruction}",
                    (15, frame.shape[0] - 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                cv2.putText(
                    frame,
                    "Press SPACE or 'C' to start burst capture  |  Press 'Q' to quit",
                    (15, frame.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1
                )
            
                cv2.imshow("User Registration UI", frame)
                key = cv2.waitKey(1) & 0xFF
            
                if key in (ord('c'), ord('C'), 32): 
                    if not faces:
                        logging.info("No face detected! Please position your face clearly in the frame.")
                        continue
                    
                    logging.info(f"\nCapturing burst of {self.samples_per_pose} images for pose: {pose}... Hold still!")
                    captured_burst = 0
                
                    while captured_burst < self.samples_per_pose:
                        ret_burst, frame_burst = cap.read()
                        if not ret_burst:
                            break
                        
                        try:
                            faces_burst = face_detector.detect_faces(frame_burst)
                        except Exception:
                            faces_burst = []
                        
                        if not faces_burst:
                            time.sleep(0.05)
                            continue
                        
                        largest_face = sorted(faces_burst, key=lambda f: f[2] * f[3], reverse=True)[0]
                        x, y, w, h = largest_face
                    
                        face_crop = frame_burst[y:y+h, x:x+w].copy()
                        if face_crop.size == 0:
                            continue
                        
                        blur_val = utils.calculate_blur(face_crop)
                        if blur_val < 30.0:
                            continue
                        
                        captured_burst += 1
                        self.saved_count += 1
                    
                        filename = f"face_{self.saved_count:03d}.jpg"
                        filepath = self.person_dir / filename
                        cropped_filepath = self.cropped_dir / filename
                    
                        cv2.imwrite(str(filepath), frame_burst)
                        cv2.imwrite(str(cropped_filepath), face_crop)
                    
                        frame_h, frame_w = frame_burst.shape[:2]
                        crop_h, crop_w = face_crop.shape[:2]
                    
                        try:
                            rel_path = str(filepath.relative_to(config.PROJECT_ROOT))
                            rel_crop_path = str(cropped_filepath.relative_to(config.PROJECT_ROOT))
                        except ValueError:
                            rel_path = str(filepath)
                            rel_crop_path = str(cropped_filepath)

                        image_id = database.save_record(
                            person_id=self.person_id,
                            image_path=rel_path,
                            cropped_path=rel_crop_path,
                            pose=pose,
                            expression=expr,
                            lighting_condition=light,
                            width=frame_w,
                            height=frame_h,
                            cropped_width=crop_w,
                            cropped_height=crop_h,
                            blur_score=blur_val,
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
                            logging.warning(f" Failed to generate embedding for crop {self.saved_count}: {e}")
                        
                        cv2.rectangle(frame_burst, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(
                            frame_burst,
                            f"Capturing: {captured_burst}/{self.samples_per_pose}",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
                        cv2.imshow("User Registration UI", frame_burst)
                        cv2.waitKey(20)
                    
                    logging.info(f"Pose registration complete! Captured {captured_burst} crops.")
                    
                elif key in (ord('q'), ord('Q')):
                    logging.info("Registration cancelled.")
                    break
                
        finally:
            cap.release()
        cv2.destroyAllWindows()
        
        if self.saved_count >= self.target_count:
            logging.info("\nRegistration Complete!")
            
            if not config.KEEP_REGISTRATION_IMAGES:
                logging.info("\nRunning storage optimization...")
                cleanup_enrollment_images.cleanup_images_for_person(self.person_id, self.name, dry_run=False)

def main():
    logging.info("\n" + "=" * 60)
    logging.info("                1. USER REGISTRATION WORKFLOW")
    logging.info("=" * 60)
    
    name = input("Enter person name to enroll: ").strip()
    if not name:
        logging.info("Invalid name. Registration cancelled.")
        return
        
    try:
        manager = RegistrationManager(name)
        manager.run_cli_loop()
    except Exception as e:
        logging.info(f"Registration failed: {e}")

if __name__ == "__main__":
    main()
