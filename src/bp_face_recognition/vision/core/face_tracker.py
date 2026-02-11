"""
High-level Face Tracker that integrates detection and recognition.

This class provides a unified interface for face detection,
recognition, and tracking functionality.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import logging

from bp_face_recognition.vision.interfaces import FaceDetector, FaceRecognizer
from bp_face_recognition.vision.factory import RecognizerFactory

logger = logging.getLogger(__name__)


class FaceTracker:
    """
    High-level face tracking class that combines detection and recognition.

    Provides unified interface for face detection, embedding extraction,
    and tracking across video frames or image sequences.
    """

    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        recognizer: Optional[FaceRecognizer] = None,
        detector_type: str = "mediapipe_v1",
        recognizer_type: str = "custom_cnn_v1",
        **kwargs: Any,
    ):
        """
        Initialize FaceTracker with detector and recognizer.

        Args:
            detector: Optional pre-configured detector
            recognizer: Optional pre-configured recognizer
            detector_type: Type of detector to create if detector not provided
            recognizer_type: Type of recognizer to create if recognizer not provided
            **kwargs: Additional configuration for detector/recognizer
        """
        self.detector_type = detector_type
        self.recognizer_type = recognizer_type

        # Initialize detector and recognizer using factory
        if detector is None:
            self.detector = RecognizerFactory.get_detector(detector_type, **kwargs)
        else:
            self.detector = detector

        if recognizer is None:
            self.recognizer = RecognizerFactory.get_recognizer(
                recognizer_type, **kwargs
            )
        else:
            self.recognizer = recognizer

        self.tracking_history: List[Dict[str, Any]] = []
        self.current_faces: List[Dict[str, Any]] = []

        logger.info(
            f"FaceTracker initialized with {detector_type} detector and {recognizer_type} recognizer"
        )

    def detect_faces(
        self, image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces in image with confidence scores.

        Args:
            image: Input image array

        Returns:
            List of tuples ((x, y, w, h), confidence)
        """
        try:
            if hasattr(self.detector, "detect_with_confidence"):
                return self.detector.detect_with_confidence(image)
            else:
                # Legacy support
                boxes = self.detector.detect(image)
                return [(box, 1.0) for box in boxes]
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []

    def get_embeddings(
        self, faces: List[Tuple[int, int, int, int]], image: np.ndarray
    ) -> np.ndarray:
        """
        Extract embeddings for detected faces.

        Args:
            faces: List of face bounding boxes (x, y, w, h)
            image: Original image array

        Returns:
            Array of embeddings for each face
        """
        if not faces:
            return np.array([])

        try:
            # Extract face crops
            face_crops = []
            for x, y, w, h in faces:
                face_crop = image[y : y + h, x : x + w]
                face_crops.append(face_crop)

            if not face_crops:
                return np.array([])

            # Extract embeddings
            embeddings = []
            for crop in face_crops:
                embedding = self.recognizer.get_embedding(crop)
                embeddings.append(embedding)

            return np.array(embeddings)

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return np.array([])

    def track_faces(
        self, image: np.ndarray, update_history: bool = True
    ) -> Dict[str, Any]:
        """
        Detect and recognize faces in image.

        Args:
            image: Input image array
            update_history: Whether to update tracking history

        Returns:
            Dictionary with detection and recognition results
        """
        # Detect faces
        detections = self.detect_faces(image)

        if not detections:
            self.current_faces = []
            return {
                "faces": [],
                "num_faces": 0,
                "detection_confidence": [],
                "embeddings": [],
            }

        # Extract bounding boxes and confidences
        boxes = []
        confidences = []
        for box, confidence in detections:
            boxes.append(box)
            confidences.append(confidence)

        # Get embeddings
        embeddings = self.get_embeddings(boxes, image)

        # Prepare face information
        face_info = []
        for i, ((box, confidence), embedding) in enumerate(zip(detections, embeddings)):
            face_data = {
                "id": i,
                "box": box,
                "confidence": confidence,
                "embedding": embedding,
                "recognized": False,
                "identity": None,
                "recognition_confidence": 0.0,
            }
            face_info.append(face_data)

        self.current_faces = face_info

        # Update tracking history
        if update_history:
            self._update_tracking_history(face_info, image.shape)

        return {
            "faces": face_info,
            "num_faces": len(face_info),
            "detection_confidence": confidences,
            "embeddings": embeddings.tolist() if len(embeddings) > 0 else [],
        }

    def recognize_face(
        self,
        embedding: np.ndarray,
        known_embeddings: Dict[str, np.ndarray],
        threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Recognize a face by comparing embedding with known embeddings.

        Args:
            embedding: Face embedding to recognize
            known_embeddings: Dictionary of known face embeddings
            threshold: Recognition confidence threshold

        Returns:
            Recognition result dictionary
        """
        if not known_embeddings:
            return {"recognized": False, "identity": None, "confidence": 0.0}

        try:
            best_match = None
            best_confidence = 0.0

            for identity, known_embedding in known_embeddings.items():
                if len(known_embedding) != len(embedding):
                    continue

                # Calculate cosine similarity
                similarity = np.dot(embedding, known_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(known_embedding)
                )

                if similarity > best_confidence:
                    best_confidence = similarity
                    best_match = identity

            if best_confidence >= threshold:
                return {
                    "recognized": True,
                    "identity": best_match,
                    "confidence": float(best_confidence),
                }
            else:
                return {
                    "recognized": False,
                    "identity": None,
                    "confidence": float(best_confidence),
                }

        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return {"recognized": False, "identity": None, "confidence": 0.0}

    def _update_tracking_history(
        self, faces: List[Dict[str, Any]], image_shape: Tuple[int, ...]
    ) -> None:
        """
        Update internal tracking history.

        Args:
            faces: List of detected faces
            image_shape: Shape of the processed image
        """
        timestamp = np.datetime64("now")

        for face in faces:
            history_entry = {
                "timestamp": timestamp,
                "image_shape": image_shape,
                "face_id": face["id"],
                "box": face["box"],
                "confidence": face["confidence"],
                "embedding": face["embedding"],
            }
            self.tracking_history.append(history_entry)

        # Limit history size to prevent memory issues
        max_history = 1000
        if len(self.tracking_history) > max_history:
            self.tracking_history = self.tracking_history[-max_history:]

    def get_tracking_summary(self) -> Dict[str, Any]:
        """
        Get summary of tracking statistics.

        Returns:
            Dictionary with tracking statistics
        """
        if not self.tracking_history:
            return {"total_detections": 0, "avg_confidence": 0.0, "unique_faces": 0}

        confidences = [entry["confidence"] for entry in self.tracking_history]
        unique_boxes = set((entry["box"]) for entry in self.tracking_history)

        return {
            "total_detections": len(self.tracking_history),
            "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "unique_faces": len(unique_boxes),
        }

    def clear_history(self) -> None:
        """Clear tracking history."""
        self.tracking_history = []
        logger.info("Tracking history cleared")

    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the detector."""
        if hasattr(self.detector, "get_detector_info"):
            return self.detector.get_detector_info()
        else:
            return {"name": self.detector.__class__.__name__, "type": "face_detector"}

    def get_recognizer_info(self) -> Dict[str, Any]:
        """Get information about the recognizer."""
        if hasattr(self.recognizer, "get_recognizer_info"):
            return self.recognizer.get_recognizer_info()
        else:
            return {
                "name": self.recognizer.__class__.__name__,
                "type": "face_recognizer",
            }

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "detector": self.get_detector_info(),
            "recognizer": self.get_recognizer_info(),
            "detector_type": self.detector_type,
            "recognizer_type": self.recognizer_type,
            "current_faces_count": len(self.current_faces),
            "tracking_history_size": len(self.tracking_history),
            "tracking_summary": self.get_tracking_summary(),
        }
