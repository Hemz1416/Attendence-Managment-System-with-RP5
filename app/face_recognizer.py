import cv2
import logging
import numpy as np
import torch  # type: ignore
from app import config
from app import database
from app import utils

class FaceRecognizer:
    def __init__(self):
        logging.info("Loading FaceNet Recognition Model...")
        from facenet_pytorch import InceptionResnetV1  # type: ignore
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            torch.set_num_threads(2)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        logging.info(f"FaceNet loaded successfully on device: {self.device}")
        
        # Load database gallery embeddings
        self.gallery = database.load_gallery_embeddings()
        self._update_gallery_matrix()
        logging.info(f"Loaded {len(self.gallery)} enrolled embeddings from database.")

    def reload_gallery(self):
        """Reload embeddings from the database."""
        self.gallery = database.load_gallery_embeddings()
        self._update_gallery_matrix()
        logging.info(f"Reloaded {len(self.gallery)} enrolled embeddings.")

    def _update_gallery_matrix(self):
        """Pre-compute the gallery matrix for fast vectorized cosine matching."""
        if self.gallery:
            self.gallery_matrix = np.vstack([e["embedding_vector"] for e in self.gallery])
            # Ensure L2 normalization of gallery matrix
            norms = np.linalg.norm(self.gallery_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.gallery_matrix = self.gallery_matrix / norms
        else:
            self.gallery_matrix = None

    def generate_embedding(self, face_bgr):
        """Generate a 512-D L2-normalised FaceNet embedding from a BGR face crop."""
        img = cv2.resize(face_bgr, config.FACENET_INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        
        # Standardization (fixed_image_standardization)
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        
        tensor = torch.FloatTensor(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(tensor).cpu().numpy()[0]
            
        # Re-normalise defensively
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding.astype(np.float32)

    def recognize_face(self, face_bgr):
        """Compare a face crop against the database gallery.
        
        Returns:
            person_name (str): The name of the best match or "Unknown"
            similarity_score (float): The similarity score
            person_id (int or None): The database person ID
        """
        if not self.gallery:
            return "Unknown", 0.0, None

        embedding = self.generate_embedding(face_bgr)
        best_match, score = utils.find_best_match(embedding, self.gallery, gallery_matrix=self.gallery_matrix)
        
        if best_match and score >= config.DEFAULT_THRESHOLD:
            return best_match["name"], score, best_match["person_id"]
        else:
            return "Unknown", score, None
