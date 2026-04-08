"""
Closed-Set Pipeline Service - Direct Classification Without Database

This service provides face recognition through direct classification
using a trained classifier model (e.g., fine-tuned FaceNet with softmax).
No registration or database required — identities are baked into the model.
"""

import cv2
import logging
import time
import numpy as np
from typing import Dict, Any, List

from bp_face_recognition.vision.factory import RecognizerFactory

logger = logging.getLogger(__name__)


class ClosedSetPipelineService:
    """
    Closed-set face recognition pipeline.

    Uses a classifier model to directly predict identity from detected faces.
    No database, no registration — the model outputs one of N known classes.

    Requires a recognizer with a `recognize()` method (e.g., FinetunedRecognizer).
    """

    def __init__(
        self,
        detector_type: str = "mediapipe_v1",
        recognizer_type: str = "facenet_pu",
        confidence_threshold: float = 0.7,
        **kwargs: Any,
    ):
        self.detector_type = detector_type
        self.recognizer_type = recognizer_type
        self.confidence_threshold = confidence_threshold

        # Create detector and recognizer via factory
        self.detector = RecognizerFactory.get_detector(detector_type, **kwargs)
        self.recognizer = RecognizerFactory.get_recognizer(recognizer_type, **kwargs)

        # Validate that recognizer supports direct classification
        if not hasattr(self.recognizer, "recognize"):
            raise ValueError(
                f"Recognizer '{recognizer_type}' does not support closed-set recognition "
                f"(no recognize() method). Use facenet_tl, facenet_pu, or facenet_tloss."
            )

        # Performance tracking
        self.processing_stats = {
            "total_images": 0,
            "total_faces_detected": 0,
            "total_faces_recognized": 0,
            "total_processing_time": 0.0,
            "errors": [],
        }

        logger.info(
            f"ClosedSetPipelineService initialized: {detector_type} + {recognizer_type} "
            f"(threshold={confidence_threshold})"
        )

    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process image through closed-set pipeline.

        Detects faces, then classifies each directly into one of N classes.
        No database lookup — uses model's softmax output.

        Args:
            image: Input image array (BGR format from OpenCV)

        Returns:
            Result dict compatible with PipelineService.process_image() format
        """
        start_time = time.time()

        try:
            # Step 1: Detect faces at reduced resolution
            scale = 0.5
            small_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

            if hasattr(self.detector, "detect_with_confidence"):
                detections = self.detector.detect_with_confidence(small_image)
            else:
                detections = [(box, 1.0) for box in self.detector.detect(small_image)]

            if not detections:
                processing_time = time.time() - start_time
                self._update_stats(0, 0, processing_time)
                return {
                    "success": True,
                    "detection_result": {"num_faces": 0},
                    "recognition_result": {
                        "faces": [],
                        "num_faces": 0,
                        "num_recognized": 0,
                    },
                    "processing_time": processing_time,
                    "mode": "closed-set",
                }

            # Step 2: Classify each detected face
            h_img, w_img = image.shape[:2]
            recognized_faces = []

            for i, ((sx, sy, sw, sh), det_confidence) in enumerate(detections):
                # Scale bounding box back to original resolution
                x = max(0, int(sx / scale))
                y = max(0, int(sy / scale))
                w = int(sw / scale)
                h = int(sh / scale)

                # Clamp to image boundaries
                w = min(w, w_img - x)
                h = min(h, h_img - y)

                if w <= 0 or h <= 0:
                    continue

                # Crop face from original high-res image
                face_crop = image[y : y + h, x : x + w]

                # Direct classification — no database
                identity, confidence = self.recognizer.recognize(face_crop)

                # Apply confidence threshold
                if confidence < self.confidence_threshold:
                    recognized = False
                    display_identity = "Unknown"
                else:
                    recognized = True
                    display_identity = identity

                recognized_faces.append(
                    {
                        "id": i,
                        "box": (x, y, w, h),
                        "confidence": det_confidence,
                        "identity": display_identity,
                        "recognized": recognized,
                        "recognition_confidence": confidence,
                        "raw_prediction": identity,
                    }
                )

            processing_time = time.time() - start_time
            num_recognized = sum(1 for f in recognized_faces if f["recognized"])
            self._update_stats(len(recognized_faces), num_recognized, processing_time)

            return {
                "success": True,
                "detection_result": {"num_faces": len(recognized_faces)},
                "recognition_result": {
                    "faces": recognized_faces,
                    "num_faces": len(recognized_faces),
                    "num_recognized": num_recognized,
                },
                "processing_time": processing_time,
                "mode": "closed-set",
            }

        except Exception as e:
            logger.error(f"Closed-set pipeline failed: {e}")
            self.processing_stats["errors"].append(str(e))
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "mode": "closed-set",
            }

    def get_class_names(self) -> List[str]:
        """Return the list of known identities from the classifier."""
        if hasattr(self.recognizer, "class_names"):
            return self.recognizer.class_names
        return []

    def _update_stats(
        self, faces_detected: int, faces_recognized: int, processing_time: float
    ) -> None:
        self.processing_stats["total_images"] += 1
        self.processing_stats["total_faces_detected"] += faces_detected
        self.processing_stats["total_faces_recognized"] += faces_recognized
        self.processing_stats["total_processing_time"] += processing_time

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        stats = self.processing_stats.copy()

        if stats["total_images"] > 0:
            stats["avg_faces_per_image"] = (
                stats["total_faces_detected"] / stats["total_images"]
            )
            stats["recognition_success_rate"] = stats[
                "total_faces_recognized"
            ] / max(stats["total_faces_detected"], 1)
            stats["avg_processing_time"] = (
                stats["total_processing_time"] / stats["total_images"]
            )
        else:
            stats["avg_faces_per_image"] = 0.0
            stats["recognition_success_rate"] = 0.0
            stats["avg_processing_time"] = 0.0

        stats["detector_type"] = self.detector_type
        stats["recognizer_type"] = self.recognizer_type
        stats["confidence_threshold"] = self.confidence_threshold
        stats["mode"] = "closed-set"
        stats["error_count"] = len(stats["errors"])

        return stats

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "mode": "closed-set",
            "detector_type": self.detector_type,
            "recognizer_type": self.recognizer_type,
            "confidence_threshold": self.confidence_threshold,
            "class_names": self.get_class_names(),
            "num_classes": len(self.get_class_names()),
            "processing_stats": self.get_processing_stats(),
        }
