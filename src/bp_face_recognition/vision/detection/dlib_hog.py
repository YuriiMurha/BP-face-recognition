"""dlib HOG Face Detector

This is a migrated version of the dlib HOG detector
with improved error handling and configuration flexibility.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

from bp_face_recognition.vision.detection.base import BaseDetector, DetectionResult

# Import dlib with better error handling
try:
    import dlib

    DLIB_AVAILABLE = True
except ImportError as e:
    DLIB_AVAILABLE = False
    dlib = None
    logging.warning(f"dlib not available: {e}")

logger = logging.getLogger(__name__)


class DlibHOGDetector(BaseDetector):
    """
    dlib Histogram of Oriented Gradients face detector.

    Fast, reliable face detection using HOG features
    with configurable upscaling and detection parameters.
    """

    def __init__(
        self,
        upsample_times: int = 1,
        confidence_threshold: float = 0.5,
        adjust_threshold: float = 0.0,
    ):
        """
        Initialize dlib HOG detector.

        Args:
            upsample_times: Number of times to upsample the image
            confidence_threshold: Detection confidence threshold
            adjust_threshold: Detection adjustment threshold
        """
        super().__init__(confidence_threshold)

        self.upsample_times = upsample_times
        self.adjust_threshold = adjust_threshold

        if DLIB_AVAILABLE:
            try:
                self.detector = dlib.get_frontal_face_detector()
                self._initialized = True
                logger.info("dlib HOG detector initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize dlib HOG detector: {e}")
                self.detector = None
                self._initialized = False
        else:
            self.detector = None
            self._initialized = False
            logger.warning("dlib not available - detector will not work")

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using dlib HOG detector.

        Args:
            image: Input image array (BGR format expected)

        Returns:
            List of bounding boxes (x, y, w, h)
        """
        if not self._validate_image(image):
            return []

        if self.detector is None:
            logger.error("dlib HOG detector not initialized")
            return []

        try:
            # Ensure image is in 8-bit format for dlib
            if image.dtype != np.uint8:
                # Convert to 8-bit if needed
                if image.max() <= 1.0:
                    image_8bit = (image * 255).astype(np.uint8)
                else:
                    image_8bit = image.astype(np.uint8)
            else:
                image_8bit = image

            # Convert to grayscale for dlib
            gray = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2GRAY)

            # Run detection with upscaling
            faces = self.detector(gray, self.upsample_times)

            # Convert dlib rectangles to bounding boxes
            boxes = []
            for face in faces:
                # dlib.rectangle provides left, top, right, bottom
                x = face.left()
                y = face.top()
                w = face.right() - face.left()
                h = face.bottom() - face.top()
                boxes.append((int(x), int(y), int(w), int(h)))

            return self._filter_detections_by_confidence([(box, 1.0) for box in boxes])

        except Exception as e:
            return self._handle_detection_error(e, "dlib HOG detection")

    def detect_with_confidence(
        self, image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces with confidence scores using dlib run method.

        Args:
            image: Input image array

        Returns:
            List of tuples ((x, y, w, h), confidence)
        """
        if not self._validate_image(image):
            return []

        if self.detector is None:
            logger.error("dlib HOG detector not initialized")
            return []

        try:
            # Ensure image is in 8-bit format for dlib
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image_8bit = (image * 255).astype(np.uint8)
                else:
                    image_8bit = image.astype(np.uint8)
            else:
                image_8bit = image

            gray = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2GRAY)

            # Use run method to get confidence scores
            faces, scores, _ = self.detector.run(gray, self.upsample_times)

            # Convert to bounding boxes with confidence
            results = []
            for face, score in zip(faces, scores):
                # dlib.rectangle provides left, top, right, bottom
                x = face.left()
                y = face.top()
                w = face.right() - face.left()
                h = face.bottom() - face.top()

                # Adjust confidence with threshold
                adjusted_confidence = score + self.adjust_threshold
                confidence = max(0.0, min(1.0, adjusted_confidence))

                box = (int(x), int(y), int(w), int(h))
                results.append((box, confidence))

            return self._filter_detections_by_confidence(results)

        except Exception as e:
            return self._handle_detection_error(e, "dlib HOG detection with confidence")

    def get_detection_result(
        self, image: np.ndarray, include_confidence: bool = True
    ) -> DetectionResult:
        """
        Get standardized detection result.

        Args:
            image: Input image array
            include_confidence: Whether to include confidence scores

        Returns:
            DetectionResult with standardized format
        """
        if include_confidence:
            detections = self.detect_with_confidence(image)
        else:
            detections = [
                (box, conf) for box, conf in self.detect_with_confidence(image)
            ]

        return create_detection_result_from_legacy(detections)

    def get_detector_info(self) -> Dict[str, Any]:
        """
        Get information about the detector.

        Returns:
            Dictionary with detector metadata
        """
        info = super().get_detector_info()
        info.update(
            {
                "dlib_available": DLIB_AVAILABLE,
                "upsample_times": self.upsample_times,
                "adjust_threshold": self.adjust_threshold,
                "detector_type": "HOG",
            }
        )

        return info
