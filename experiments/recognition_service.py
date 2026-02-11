import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Any, Sequence
from bp_face_recognition.database.database import FaceDatabase
from bp_face_recognition.models.model import FaceTracker


class RecognitionService:
    """
    Headless service for face recognition.
    Encapsulates detection, embedding extraction, and database matching.
    Can be used by CLI tools, APIs, or GUIs.
    """

    def __init__(
        self,
        tracker: Optional[FaceTracker] = None,
        database: Optional[FaceDatabase] = None,
        threshold: float = 0.6,
    ):
        self.tracker = tracker or FaceTracker()
        self.db = database or FaceDatabase(db_type="csv")
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

    def process_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Processes a single frame and returns structured recognition results.
        Does NOT perform any drawing or UI operations.

        Returns:
            List[Dict]: List of results, each containing 'box', 'id', 'label', 'confidence'.
        """
        results = []
        boxes_confidences = self.tracker.detect_faces(frame)
        known_embeddings = self.db.get_all_embeddings()

        for box, confidence in boxes_confidences:
            if confidence < 0.8:  # Detector confidence threshold
                continue

            x, y, w, h = box
            x, y = max(0, x), max(0, y)
            face_crop = frame[y : y + h, x : x + w]

            if face_crop.size == 0:
                continue

            embedding = self.tracker.get_embedding(face_crop)

            # Match against database
            match_id, label, dist = self._match_embedding(embedding, known_embeddings)

            # Handle new faces (optional behavior, can be configured)
            if label == "stranger":
                match_id = self.db.add_face(embedding)
                label = str(match_id)

            results.append(
                {
                    "box": (x, y, w, h),
                    "id": match_id,
                    "label": label,
                    "confidence": float(confidence),
                    "distance": float(dist),
                }
            )

            # Log detection for persistence
            if match_id is not None:
                self.db.log_detection(match_id, label)

        return results

    def _match_embedding(
        self, embedding: np.ndarray, known_embeddings: Sequence[Tuple[Any, Any]]
    ) -> Tuple[Optional[int], str, float]:
        """Compares embedding against known gallery and returns the best match."""
        if not known_embeddings:
            return None, "stranger", float("inf")

        distances = [np.linalg.norm(embedding - emb) for _, emb in known_embeddings]
        min_dist_raw = min(distances)
        min_dist = float(min_dist_raw)
        match_idx = distances.index(min_dist_raw)
        potential_id, _ = known_embeddings[match_idx]

        if min_dist < self.threshold:
            return int(potential_id), str(potential_id), min_dist

        return None, "stranger", min_dist
