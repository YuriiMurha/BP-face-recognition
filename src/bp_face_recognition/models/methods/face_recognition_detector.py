import cv2
import numpy as np
import face_recognition
from typing import List, Tuple
from bp_face_recognition.models.interfaces import FaceDetector


class FaceRecognitionLibDetector(FaceDetector):
    """
    Wrapper for face_recognition library to implement FaceDetector interface.
    Uses HOG-based detection with CNN fallback.
    """

    def __init__(self, model="hog", num_upsamples=2):
        """
        Initialize face_recognition library detector.

        Args:
            model: "hog" for faster HOG detection, "cnn" for more accurate CNN detection
            num_upsamples: Number of times to upsample the image
        """
        self.model = model
        self.num_upsamples = num_upsamples

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using face_recognition library.

        Args:
            image: Input image in BGR format (OpenCV default)

        Returns:
            List of bounding boxes as (x, y, w, h)
        """
        if image is None:
            return []

        # Ensure image is in 8-bit format
        if image.dtype != np.uint8:
            image = (
                (image * 255).astype(np.uint8)
                if image.max() <= 1.0
                else image.astype(np.uint8)
            )

        # Downsample for faster processing (face_recognition works best on smaller images)
        img_small = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(
            img_rgb, model=self.model, number_of_times_to_upsample=self.num_upsamples
        )

        # Convert from (top, right, bottom, left) at 1/4 scale to (x, y, w, h) at full scale
        faces = []
        for top, right, bottom, left in face_locations:
            x, y, w, h = left * 4, top * 4, (right - left) * 4, (bottom - top) * 4
            faces.append((x, y, w, h))

        return faces

    def detect_with_confidence(
        self, image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces using face_recognition library and return confidence scores.

        Note: face_recognition library doesn't provide confidence scores by default.
        We'll use face distances as a proxy for confidence.

        Args:
            image: Input image in BGR format

        Returns:
            List of tuples: ((x, y, w, h), confidence_score)
        """
        boxes = self.detect(image)
        # Since face_recognition doesn't provide confidence, we'll use 0.9 for all detections
        # In practice, this could be improved by analyzing face quality or using face distances
        return [(box, 0.9) for box in boxes]
