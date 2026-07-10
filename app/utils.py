import time
import logging
import cv2
import numpy as np
from pathlib import Path
from app import config

class MockCapture:
    """Simulates a cv2.VideoCapture using dataset images or a bouncing box."""
    def __init__(self, dataset_dir, width=640, height=480):
        self.width = width
        self.height = height
        
        # Search for raw webcam frames in dataset directory or parent dataset directory
        self.image_paths = []
        search_paths = [
            Path(dataset_dir),
            Path(dataset_dir).parent / "dataset",
            Path(__file__).resolve().parent.parent.parent / "dataset"
        ]
        for sp in search_paths:
            if sp.exists() and sp.is_dir():
                for child in sp.iterdir():
                    if child.is_dir() and "cropped" not in child.name.lower():
                        self.image_paths.extend(child.glob("*.jpg"))
                        self.image_paths.extend(child.glob("*.png"))
                        self.image_paths.extend(child.glob("*.jpeg"))
                if self.image_paths:
                    break
                
        self.image_paths = sorted(list(set(self.image_paths)))
        
        if not self.image_paths:
            logging.warning(" No raw images found in dataset directory. Creating a blank placeholder frame for testing.")
            self.blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(self.blank_frame, "Mock Face (No images)", (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.rectangle(self.blank_frame, (220, 140), (420, 340), (0, 255, 0), 2)
        else:
            logging.info(f"[Info] Mock capture initialized with {len(self.image_paths)} raw images from dataset.")
            
        self.idx = 0

    def read(self):
        if not self.image_paths:
            time.sleep(0.033)  # Simulate ~30fps
            return True, self.blank_frame.copy()
            
        path = self.image_paths[self.idx % len(self.image_paths)]
        self.idx += 1
        
        frame = cv2.imread(str(path))
        if frame is None:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, "Image Read Fail", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            frame = cv2.resize(frame, (self.width, self.height))
            
        time.sleep(0.033)  # Simulate ~30fps
        return True, frame

    def isOpened(self):
        return True

    def release(self):
        pass


def cosine_similarity(vec_a, vec_b):
    """Cosine similarity in [-1, 1]. Higher -> more similar."""
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def euclidean_distance(vec_a, vec_b):
    """L2 distance. Lower -> more similar."""
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    return float(np.linalg.norm(a - b))

def similarity_score(vec_a, vec_b, metric=None):
    """Compute similarity score where higher = more similar."""
    metric = metric or config.SIMILARITY_METRIC
    if metric == "cosine":
        return cosine_similarity(vec_a, vec_b)
    elif metric == "euclidean":
        dist = euclidean_distance(vec_a, vec_b)
        return 1.0 / (1.0 + dist)
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")

def find_best_match(query_embedding, gallery_embeddings, gallery_matrix=None, metric=None):
    """Find the closest embedding in gallery_embeddings.
    
    Returns:
        (best_entry, best_score)
    """
    if not gallery_embeddings:
        return None, -1.0

    metric = metric or config.SIMILARITY_METRIC

    # Fast path: Vectorized cosine similarity using pre-stacked matrix
    if metric == "cosine" and gallery_matrix is not None:
        query = np.asarray(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm
            
        scores = np.dot(gallery_matrix, query)
        best_idx = int(np.argmax(scores))
        return gallery_embeddings[best_idx], float(scores[best_idx])

    # Fallback: iterative evaluation
    best_entry = None
    best_score = -float("inf")

    for entry in gallery_embeddings:
        score = similarity_score(query_embedding, entry["embedding_vector"], metric)
        if score > best_score:
            best_score = score
            best_entry = entry

    return best_entry, best_score

def calculate_blur(image):
    """Calculate a blur score using the variance of the Laplacian."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def calculate_brightness(image):
    """Calculate a brightness score using the mean pixel intensity."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(np.mean(gray))
