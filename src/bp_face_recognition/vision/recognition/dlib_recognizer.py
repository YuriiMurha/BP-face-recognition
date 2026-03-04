"""
Dlib-based recognizer using the face_recognition library.
Provides highly accurate 128D face embeddings.
"""

import logging
import numpy as np
import face_recognition
from typing import List, Optional, Tuple, Dict, Any

from bp_face_recognition.vision.recognition.base import (
    BaseRecognizer,
    EmbeddingMetadata,
)

logger = logging.getLogger(__name__)


class DlibRecognizer(BaseRecognizer):
    """
    Recognizer using dlib's state-of-the-art face recognition model.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (160, 160),
        normalize: bool = True,
    ):
        """
        Initialize Dlib recognizer.
        """
        super().__init__(input_size=input_size, normalize=normalize)
        self._initialized = True
        self.embedding_size = 128
        logger.info("DlibRecognizer (face_recognition) initialized")

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Get face embedding from face image.

        Args:
            face_image: Input face image (expected BGR)

        Returns:
            128D embedding vector
        """
        if face_image is None or face_image.size == 0:
            return np.array([])

        try:
            # Convert BGR to RGB (dlib expects RGB)
            import cv2

            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Since the image is already cropped, we provide the full image as the face location
            # Format: (top, right, bottom, left)
            height, width = rgb_image.shape[:2]
            face_locations = [(0, width, height, 0)]

            encodings = face_recognition.face_encodings(
                rgb_image, known_face_locations=face_locations
            )

            if encodings:
                return encodings[0]
            else:
                # Try without locations as fallback
                encodings = face_recognition.face_encodings(rgb_image)
                if encodings:
                    return encodings[0]
                return np.array([])
        except Exception as e:
            logger.error(f"Dlib embedding failed: {e}")
            return np.array([])

    def get_recognizer_info(self) -> Dict[str, Any]:
        info = super().get_recognizer_info()
        info.update(
            {
                "model_type": "Dlib-ResNet34",
                "embedding_size": 128,
                "library": "face_recognition",
            }
        )
        return info
