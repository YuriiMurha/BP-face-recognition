"""OpenCV Haar Cascade Face Detector

This is the migrated version of the Haar Cascade detector
with improved error handling and configuration flexibility.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from pathlib import Path

from bp_face_recognition.vision.detection.base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)


class HaarCascadeDetector(BaseDetector):
    """
    OpenCV Haar Cascade face detector with configurable parameters.

    Simple, fast face detection using Haar Cascades with
    configurable cascade files and detection parameters.
    """

    def __init__(
        self,
        cascade_file: Optional[str] = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Union[Tuple[int, int], List[int]] = (30, 30),
        max_size: Optional[Union[Tuple[int, int], List[int]]] = None,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize Haar Cascade detector.

        Args:
            cascade_file: Path to custom cascade file (optional)
            scale_factor: Parameter specifying how much the image size is reduced at each image scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have
            min_size: Minimum possible object size
            max_size: Maximum possible object size
            confidence_threshold: Minimum confidence threshold for detections
        """
        super().__init__(confidence_threshold)

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        # Convert list to tuple if needed
        self.min_size = tuple(min_size) if isinstance(min_size, list) else min_size
        self.max_size = tuple(max_size) if isinstance(max_size, list) else max_size

        # Load cascade classifier
        self.cascade_file = cascade_file or self._get_default_cascade()
        try:
            self.face_cascade = cv2.CascadeClassifier(self.cascade_file)
            self._initialized = True
            logger.info(f"Haar Cascade loaded from: {self.cascade_file}")
        except Exception as e:
            logger.error(f"Failed to load cascade from {self.cascade_file}: {e}")
            self.face_cascade = None
            self._initialized = False

    def _get_default_cascade(self) -> str:
        """
        Get default Haar Cascade file path.

        Returns:
            Path to default frontal face cascade
        """
        try:
            import cv2

            return cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        except AttributeError:
            # Fallback for older OpenCV versions
            import os

            cv2_path = os.path.dirname(cv2.__file__)
            return os.path.join(
                cv2_path, "data", "haarcascades", "haarcascade_frontalface_default.xml"
            )

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image using Haar Cascade.

        Args:
            image: Input image array (BGR format expected)

        Returns:
            List of bounding boxes (x, y, w, h)
        """
        if not self._validate_image(image):
            return []

        if self.face_cascade is None:
            logger.error("Haar Cascade not initialized")
            return []

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                maxSize=self.max_size or (image.shape[1], image.shape[0]),
            )

            # Convert results to standard format
            boxes = []
            if faces is not None:
                for x, y, w, h in faces:
                    boxes.append((int(x), int(y), int(w), int(h)))

            return self._filter_detections_by_confidence([(box, 1.0) for box in boxes])

        except Exception as e:
            return self._handle_detection_error(e, "Haar Cascade detection")

    def detect_with_confidence(
        self, image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces with confidence estimates using detectMultiScale3.

        Args:
            image: Input image array

        Returns:
            List of tuples ((x, y, w, h), confidence)
        """
        if not self._validate_image(image):
            return []

        if self.face_cascade is None:
            logger.error("Haar Cascade not initialized")
            return []

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Try detectMultiScale3 for confidence scores
            try:
                faces, rejectLevels, levelWeights = self.face_cascade.detectMultiScale3(
                    gray,
                    scaleFactor=self.scale_factor,
                    minNeighbors=self.min_neighbors,
                    minSize=self.min_size,
                    outputRejectLevels=True,
                )

                if len(faces) == 0 or isinstance(faces, tuple):
                    return []

                # Extract confidence scores
                results = []
                for i, (x, y, w, h) in enumerate(faces):
                    confidence = (
                        float(levelWeights[i]) if i < len(levelWeights) else 1.0
                    )
                    box = (int(x), int(y), int(w), int(h))
                    results.append((box, confidence))

                return self._filter_detections_by_confidence(results)

            except Exception:
                # Fallback to basic detection
                boxes = self.detect(image)
                return [(box, 1.0) for box in boxes]

        except Exception as e:
            return self._handle_detection_error(
                e, "Haar Cascade detection with confidence"
            )

    def get_detector_info(self) -> Dict[str, Any]:
        """
        Get information about the detector.

        Returns:
            Dictionary with detector metadata
        """
        info = super().get_detector_info()
        info.update(
            {
                "cascade_file": self.cascade_file,
                "scale_factor": self.scale_factor,
                "min_neighbors": self.min_neighbors,
                "min_size": self.min_size,
                "max_size": self.max_size,
                "initialized": self._initialized,
            }
        )

        return info
