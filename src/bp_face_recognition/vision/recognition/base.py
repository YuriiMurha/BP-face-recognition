"""
Base utilities and common functionality for vision recognition methods.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, Tuple

from bp_face_recognition.vision.interfaces import FaceRecognizer

logger = logging.getLogger(__name__)


class BaseRecognizer(FaceRecognizer):
    """
    Base class for all face recognizers with common utilities.
    """

    def __init__(
        self, input_size: Tuple[int, int] = (224, 224), normalize: bool = True
    ):
        """
        Initialize base recognizer with common settings.

        Args:
            input_size: Expected input image size (height, width)
            normalize: Whether to normalize pixel values
        """
        self.input_size = input_size
        self.normalize = normalize
        self._initialized = False

    def _validate_face_image(self, face_image: np.ndarray) -> bool:
        """
        Validate face image format and dimensions.

        Args:
            face_image: Input face image array

        Returns:
            True if valid, False otherwise
        """
        if face_image is None:
            logger.error("Face image is None")
            return False

        if not isinstance(face_image, np.ndarray):
            logger.error(f"Face image must be numpy array, got {type(face_image)}")
            return False

        if len(face_image.shape) != 3:
            logger.error(f"Face image must be 3D array, got shape {face_image.shape}")
            return False

        if face_image.shape[2] != 3:
            logger.error(f"Face image must have 3 channels, got {face_image.shape[2]}")
            return False

        h, w = face_image.shape[:2]
        expected_h, expected_w = self.input_size

        if (h, w) != (expected_h, expected_w):
            logger.warning(
                f"Face image size {(h, w)} differs from expected {self.input_size}"
            )

        return True

    def _preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Common face image preprocessing.

        Args:
            face_image: Input face image array

        Returns:
            Preprocessed face image array
        """
        # Convert to RGB if needed (assuming input is BGR from OpenCV)
        if face_image.shape[2] == 3:
            # Standardize to RGB - this handles both BGR and RGB input
            processed = face_image.copy()
        else:
            # If grayscale, convert to 3-channel
            processed = np.stack([face_image] * 3, axis=-1)

        # Resize if needed
        h, w = processed.shape[:2]
        expected_h, expected_w = self.input_size

        if (h, w) != (expected_h, expected_w):
            # Use simple resize - subclasses can override with better methods
            import cv2

            processed = cv2.resize(processed, (expected_w, expected_h))

        # Normalize if requested
        if self.normalize:
            processed = processed.astype(np.float32) / 255.0

        return processed

    def _validate_embedding(self, embedding: np.ndarray) -> bool:
        """
        Validate embedding vector format.

        Args:
            embedding: Input embedding vector

        Returns:
            True if valid, False otherwise
        """
        if embedding is None:
            logger.error("Embedding is None")
            return False

        if not isinstance(embedding, np.ndarray):
            logger.error(f"Embedding must be numpy array, got {type(embedding)}")
            return False

        if len(embedding.shape) != 1:
            logger.error(f"Embedding must be 1D array, got shape {embedding.shape}")
            return False

        if np.any(np.isnan(embedding)):
            logger.error("Embedding contains NaN values")
            return False

        if np.any(np.isinf(embedding)):
            logger.error("Embedding contains infinite values")
            return False

        return True

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        L2-normalize embedding vector.

        Args:
            embedding: Input embedding vector

        Returns:
            L2-normalized embedding vector
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def _handle_recognition_error(self, error: Exception, context: str) -> np.ndarray:
        """
        Handle recognition errors gracefully.

        Args:
            error: Exception that occurred
            context: Context where error occurred

        Returns:
            Empty embedding array
        """
        logger.error(f"Recognition error in {context}: {error}")
        return np.array([])

    def get_recognizer_info(self) -> Dict[str, Any]:
        """
        Get information about the recognizer.

        Returns:
            Dictionary with recognizer metadata
        """
        return {
            "name": self.__class__.__name__,
            "input_size": self.input_size,
            "normalize": self.normalize,
            "initialized": self._initialized,
            "type": "face_recognizer",
        }

    def compute_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray, method: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            method: Similarity method ("cosine" or "euclidean")

        Returns:
            Similarity score
        """
        if not self._validate_embedding(embedding1) or not self._validate_embedding(
            embedding2
        ):
            return 0.0

        if method == "cosine":
            # Cosine similarity
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return np.dot(embedding1, embedding2) / (norm1 * norm2)

        elif method == "euclidean":
            # Convert euclidean distance to similarity
            distance = np.linalg.norm(embedding1 - embedding2)
            # Scale to [0, 1] where higher is more similar
            return float(1.0 / (1.0 + distance))

        else:
            logger.warning(f"Unknown similarity method: {method}")
            return 0.0


class EmbeddingMetadata:
    """
    Container for embedding metadata.
    """

    def __init__(
        self,
        embedding: np.ndarray,
        confidence: float = 1.0,
        processing_time: Optional[float] = None,
        face_quality: Optional[float] = None,
    ):
        """
        Initialize embedding metadata.

        Args:
            embedding: The embedding vector
            confidence: Confidence in embedding quality
            processing_time: Time taken for embedding extraction (seconds)
            face_quality: Optional face quality score
        """
        self.embedding = embedding
        self.confidence = confidence
        self.processing_time = processing_time
        self.face_quality = face_quality
        self.timestamp = None  # Can be set later

    def get_embedding_info(self) -> Dict[str, Any]:
        """Get summary of embedding information."""
        return {
            "embedding_shape": self.embedding.shape,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "face_quality": self.face_quality,
            "timestamp": self.timestamp,
        }


def create_embedding_metadata(
    embedding: np.ndarray,
    confidence: float = 1.0,
    processing_time: Optional[float] = None,
) -> EmbeddingMetadata:
    """
    Create EmbeddingMetadata from embedding vector.

    Args:
        embedding: The embedding vector
        confidence: Confidence in embedding quality
        processing_time: Optional processing time

    Returns:
        EmbeddingMetadata instance
    """
    return EmbeddingMetadata(
        embedding=embedding, confidence=confidence, processing_time=processing_time
    )
