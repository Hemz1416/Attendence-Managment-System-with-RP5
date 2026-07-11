#!/usr/bin/env python
"""Deployment validation script for Raspberry Pi 5.

This script runs a quick, headless performance benchmark of the detection
and recognition pipeline to ensure it meets production latency requirements
on the deployment hardware.
"""

import time
import logging
import sys
import os
from pathlib import Path
import cv2
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Ensure app/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

from app import config  # type: ignore
from app import database  # type: ignore
from app import face_detector  # type: ignore
from app import utils  # type: ignore
from app.face_recognizer import FaceRecognizer  # type: ignore

import platform
try:
    import torch  # type: ignore
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import mediapipe as mp
    HAS_MP = True
except ImportError:
    HAS_MP = False

def main():
    database.init_tables()
    # Pre-load gallery for banner stats
    gallery = database.load_gallery_embeddings(config.RECOGNIZER_MODEL_NAME)
    num_embeddings = len(gallery)
    num_persons = len(set(item['person_id'] for item in gallery))
    emb_dim = gallery[0]['embedding_dim'] if num_embeddings > 0 else 0

    logging.info("=" * 60)
    logging.info("      RP5 DEPLOYMENT PERFORMANCE VALIDATION")
    logging.info("=" * 60)
    logging.info(f"Board          : {platform.node() or 'Raspberry Pi'}")
    logging.info(f"Architecture   : {platform.machine()}")
    logging.info(f"Python         : {platform.python_version()}")
    logging.info(f"MediaPipe      : {getattr(mp, '__version__', 'unknown') if HAS_MP else 'unknown'}")
    logging.info(f"Torch          : {torch.__version__ if HAS_TORCH else 'unknown'}")
    logging.info(f"FaceNet        : {config.RECOGNIZER_MODEL_NAME}")
    logging.info(f"Detector       : MediaPipe")
    logging.info(f"Database Size  : {num_embeddings} embeddings")
    logging.info("=" * 60)
    
    # 1. Pre-flight Checks
    logging.info("\n1. Running Pre-flight Checks...")
    models_dir = Path(__file__).resolve().parent / "models"
    tflite_path = models_dir / "face_detection_full_range.tflite"
    if not tflite_path.exists():
        logging.info(f"   [FAIL] Missing MediaPipe model: {tflite_path}")
    else:
        logging.info(f"   [PASS] MediaPipe model found.")

    torch_cache = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "checkpoints"
    vgg_found = False
    if torch_cache.exists():
        for fname in os.listdir(torch_cache):
            if "vggface2" in fname.lower() or "20180402" in fname:
                vgg_found = True
                break
    if not vgg_found:
        logging.info("   [WARN] FaceNet vggface2 weights not found in cache. Will require internet download on first run.")
    else:
        logging.info("   [PASS] FaceNet weights found in cache.")

    # 2. Init Database & Gallery
    logging.info("\n2. Initializing Database...")
    t0 = time.perf_counter()
    # Gallery is already loaded for the banner, but we calculate time to simulate normal load
    _ = database.load_gallery_embeddings(config.RECOGNIZER_MODEL_NAME)
    db_time = (time.perf_counter() - t0) * 1000
    logging.info(f"   Loaded embeddings : {num_embeddings}")
    logging.info(f"   Registered persons: {num_persons}")
    logging.info(f"   Embedding dimension: {emb_dim}")
    logging.info(f"   (Load time: {db_time:.1f} ms)")
    if not gallery:
        logging.info("   [WARN] Gallery is empty! Recognition will always return 'Unknown'.")
        logging.info("          Run 'python main.py' and choose option 1 to register a person first.")

    # 3. Init Face Detector
    logging.info("\n3. Initializing Detector (MediaPipe)...")
    t0 = time.perf_counter()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = face_detector.detect_faces(dummy_frame) # Warmup
    det_init_time = (time.perf_counter() - t0) * 1000
    logging.info(f"   Detector ready in {det_init_time:.1f} ms")

    # 4. Init Face Recognizer
    logging.info("\n4. Initializing Recognizer (FaceNet CPU)...")
    t0 = time.perf_counter()
    recognizer = FaceRecognizer()
    dummy_crop = np.zeros((160, 160, 3), dtype=np.uint8)
    _ = recognizer.generate_embedding(dummy_crop) # Warmup
    rec_init_time = (time.perf_counter() - t0) * 1000
    logging.info(f"   Recognizer ready in {rec_init_time:.1f} ms")
    
    # 5. Pipeline Benchmark
    logging.info("\n5. Running Pipeline Benchmark (50 iterations)...")
    num_iterations = 50
    det_latencies = []
    rec_latencies = []
    match_latencies = []

    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        psutil.cpu_percent() # discard first measure
    else:
        mem_before = 0
    
    # Try to load a real face image for realistic timings, fallback to random noise
    dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
    test_img = None
    if dataset_dir.exists():
        for ext in ["*.jpg", "*.png"]:
            images = list(dataset_dir.rglob(ext))
            if images:
                test_img = cv2.imread(str(images[0]))
                if test_img is not None:
                    test_img = cv2.resize(test_img, (640, 480))
                    break
                    
    if test_img is None:
        logging.warning("    No real images found, using synthetic noise frame.")
        test_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    for i in range(num_iterations):
        # Detection
        t_det_start = time.perf_counter()
        faces = face_detector.detect_faces(test_img)
        t_det_end = time.perf_counter()
        det_latencies.append((t_det_end - t_det_start) * 1000)
        
        # Determine face crop
        if faces:
            x, y, w, h = faces[0]
            face_crop = test_img[y:y+h, x:x+w]
        else:
            face_crop = dummy_crop
            
        # Recognition
        t_rec_start = time.perf_counter()
        emb = recognizer.generate_embedding(face_crop)
        t_rec_end = time.perf_counter()
        rec_latencies.append((t_rec_end - t_rec_start) * 1000)
        
        # Matching (use gallery_matrix if populated by the updated recognizer)
        t_match_start = time.perf_counter()
        utils.find_best_match(emb, gallery, gallery_matrix=getattr(recognizer, 'gallery_matrix', None))
        t_match_end = time.perf_counter()
        match_latencies.append((t_match_end - t_match_start) * 1000)

    if HAS_PSUTIL:
        mem_after = process.memory_info().rss / (1024 * 1024)
        cpu_usage = psutil.cpu_percent()
    else:
        mem_after = 0
        cpu_usage = 0

    # 6. Summary
    avg_det = np.mean(det_latencies)
    avg_rec = np.mean(rec_latencies)
    avg_match = np.mean(match_latencies)
    avg_total = avg_det + avg_rec + avg_match
    estimated_fps = 1000.0 / avg_total if avg_total > 0 else 0
    
    logging.info("\n" + "=" * 60)
    logging.info("      VALIDATION RESULTS")
    logging.info("=" * 60)
    logging.info(f"Average Detection Latency : {avg_det:>6.1f} ms")
    logging.info(f"Average Recognition Latency: {avg_rec:>6.1f} ms")
    logging.info(f"Average Matching Latency  : {avg_match:>6.1f} ms")
    logging.info("-" * 60)
    logging.info(f"Total Pipeline Latency    : {avg_total:>6.1f} ms")
    logging.info(f"Estimated Max FPS         : {estimated_fps:>6.1f} FPS")
    if HAS_PSUTIL:
        logging.info(f"RAM Usage                 : {mem_after:>6.1f} MB")
        logging.info(f"CPU Usage                 : {cpu_usage:>6.1f} %")
    else:
        logging.info("RAM/CPU Usage             : N/A (psutil not installed)")
    logging.info("=" * 60)
    
    # 7. RP5 Pass/Fail Evaluation
    logging.info("\n--- Raspberry Pi 5 Threshold Evaluation ---")
    
    # Thresholds for RP5
    thresholds = {
        "Detection latency": (avg_det < 15.0, f"<{15} ms"),
        "Recognition latency": (avg_rec < 250.0, f"<{250} ms"),
        "Matching latency": (avg_match < 10.0, f"<{10} ms"),
        "Pipeline latency": (avg_total < 250.0, f"<{250} ms"),
        "Recognition throughput": (estimated_fps > 4.0, f">{4} FPS")
    }
    
    if HAS_PSUTIL:
        thresholds["RAM Usage"] = (mem_after < 750.0, f"<{750} MB")
    
    all_passed = True
    for name, (passed, condition) in thresholds.items():
        status = "[PASS]" if passed else "[FAIL]"
        logging.info(f"{status} {name} ({condition})")
        if not passed:
            all_passed = False
            
    logging.info(f"[INFO] CPU usage ({cpu_usage:.1f}%) is report-only and does not affect pass/fail.")
    
    if all_passed:
        logging.info("\n[SUCCESS] Performance meets all realistic RP5 deployment requirements!")
    else:
        logging.info("\n[FAILURE] Performance is below RP5 minimum requirements on one or more metrics.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )
    main()
