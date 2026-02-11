"""Face Recognition Library Face Detector - Migrated to New Architecture"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

from bp_face_recognition.vision.detection.base import BaseDetector
from bp_face_recognition.vision.interfaces import FaceDetector

# Import face_recognition with better error handling
try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = True
except ImportError as e:
    FACE_RECOGNITION_AVAILABLE = False
    face_recognition = None
    logging.warning(f"face_recognition library not available: {e}")

logger = logging.getLogger(__name__)


class FaceRecognitionLibDetector(BaseDetector):
    """
    Face Recognition Library face detector with configurable parameters.

    Uses the python face_recognition library which provides
    HOG-based face detection similar to dlib but with
    better Python packaging.
    """

    def __init__(
        self,
        model: str = "hog",
        number_of_times_to_upsample: int = 1,
        resize_scale: float = 1.0,
        min_neighbors: int = 3,
        min_face_size: int = 20,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize Face Recognition Library detector.

        Args:
            model: Detection model ('hog' or 'cnn')
            number_of_times_to_upsample: Upsampling factor for image pyramid
            resize_scale: Image scaling factor
            min_neighbors: Minimum neighbors for each candidate rectangle
            min_face_size: Minimum face size in pixels
            confidence_threshold: Detection confidence threshold
        """
        super().__init__(confidence_threshold)

        self.model = model
        self.number_of_times_to_upsample = number_of_times_to_upsample
        self.resize_scale = resize_scale
        self.min_neighbors = min_neighbors
        self.min_face_size = min_face_size

        if FACE_RECOGNITION_AVAILABLE:
            try:
                self.detector = face_recognition.FaceLocator(
                    model=model,
                    number_of_times_to_upsample=number_of_times_to_upsample,
                    resize_scale=resize_scale,
                    min_neighbors=min_neighbors,
                )
                self._initialized = True
                logger.info(
                    "Face Recognition Library detector initialized successfully"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize Face Recognition Library detector: {e}"
                )
                self.detector = None
                self._initialized = False
        else:
            self.detector = None
            self._initialized = False
            logger.error("Face Recognition Library not available")

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using Face Recognition Library.

        Args:
            image: Input image array (grayscale preferred)

        Returns:
            List of bounding boxes (x, y, w, h)
        """
        if not self._validate_image(image):
            return []

        if self.detector is None:
            logger.error("Face Recognition Library detector not initialized")
            return []

        try:
            # Convert to grayscale (face_recognition prefers grayscale)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Run detection
            faces = self.detector.detect(gray)

            # Convert to standard format
            boxes = []
            for face in faces:
                # face_recognition returns left, top, right, bottom
                x = face.left()
                y = face.top()
                w = face.right() - face.left()
                h = face.bottom() - face.top()
                boxes.append((int(x), int(y), int(w), int(h)))

            return self._filter_detections_by_confidence([(box, 1.0) for box in boxes])

        except Exception as e:
            return self._handle_detection_error(e, "Face Recognition Library detection")

    def detect_with_confidence(
        self, image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces with confidence scores using Face Recognition Library.

        Args:
            image: Input image array

        Returns:
            List of tuples ((x, y, w, h), confidence)
        """
        if not self._validate_image(image):
            return []

        if self.detector is None:
            logger.error("Face Recognition Library detector not initialized")
            return []

        try:
            # Convert to grayscale (face_recognition prefers grayscale)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Face Recognition Library doesn't provide confidence scores directly
            # Use default confidence for all detections
            faces = self.detect(gray)

            results = []
            for face in faces:
                x = face.left()
                y = face.top()
                w = face.right() - face.left()
                h = face.bottom() - face.top()
                box = (int(x), int(y), int(w), int(h))
                results.append((box, 1.0))  # Default confidence

            return self._filter_detections_by_confidence(results)

        except Exception as e:
            return self._handle_detection_error(
                e, "Face Recognition Library detection with confidence"
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
                "face_recognition_available": FACE_RECOGNITION_AVAILABLE,
                "model": self.model,
                "number_of_times_to_upsample": self.number_of_times_to_upsample,
                "resize_scale": self.resize_scale,
                "min_neighbors": self.min_neighbors,
                "min_face_size": self.min_face_size,
                "detector_type": "FaceRecognitionLib",
            }
        )

        return info
