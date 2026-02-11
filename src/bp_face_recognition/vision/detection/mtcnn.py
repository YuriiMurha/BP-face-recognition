"""MTCNN Face Detector - Migrated to New Architecture"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

from bp_face_recognition.vision.detection.base import BaseDetector

# Try to import optional dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from mtcnn import MTCNN as MTCNN_Lib

    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    MTCNN_Lib = None

logger = logging.getLogger(__name__)


class MTCNNDetector(BaseDetector):
    """
    MTCNN face detector with configurable thresholds.

    Multi-task Cascaded Convolutional Networks for robust
    face detection with adjustable confidence thresholds.
    """

    def __init__(
        self,
        thresholds: list = [0.6, 0.7, 0.8],
        min_face_size: int = 20,
        device: str = "cpu",
        model_file: Optional[str] = None,
    ):
        """
        Initialize MTCNN detector.

        Args:
            thresholds: List of confidence thresholds for 3 stages
            min_face_size: Minimum face size to detect
            device: Device to run on ('cpu' or 'gpu')
            model_file: Optional path to pre-trained model file
        """
        super().__init__()

        self.thresholds = thresholds
        self.min_face_size = min_face_size
        self.device = device
        self.model_file = model_file

        if not MTCNN_AVAILABLE:
            logger.error("MTCNN library not available")
            self.detector = None
            self._initialized = False
            return

        try:
            # Load MTCNN
            self.detector = MTCNN_Lib()

            # Set device
            if hasattr(self.detector, "device") and TORCH_AVAILABLE:
                self.detector.device = torch.device(device)

            # Load model if provided
            if model_file:
                logger.info(f"Loading MTCNN model from: {model_file}")
                # Note: MTCNN has built-in model loading
                # self.detector.load_model(model_file)

            self._initialized = True
            logger.info("MTCNN detector initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MTCNN: {e}")
            self.detector = None
            self._initialized = False

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using MTCNN.

        Args:
            image: Input image array (RGB format preferred)

        Returns:
            List of bounding boxes (x, y, w, h)
        """
        if not self._validate_image(image):
            return []

        if self.detector is None:
            logger.error("MTCNN detector not initialized")
            return []

        try:
            # Convert to RGB if needed
            rgb_image = self._ensure_rgb(image)

            # Run detection
            faces = self.detector.detect_faces(rgb_image, landmarks=False)

            # Convert to standard format
            boxes = []
            for face in faces:
                if face["confidence"] >= self.confidence_threshold:
                    box = (
                        int(face["box"][0]),
                        int(face["box"][1]),
                        int(face["box"][2]),
                        int(face["box"][3]),
                    )
                    boxes.append(box)

            return self._filter_detections_by_confidence([(box, 1.0) for box in boxes])

        except Exception as e:
            return self._handle_detection_error(e, "MTCNN detection")

    def detect_with_confidence(
        self, image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces with confidence scores using MTCNN.

        Args:
            image: Input image array

        Returns:
            List of tuples ((x, y, w, h), confidence)
        """
        if not self._validate_image(image):
            return []

        if self.detector is None:
            logger.error("MTCNN detector not initialized")
            return []

        try:
            # Convert to RGB if needed
            rgb_image = self._ensure_rgb(image)

            # Run detection with confidence
            faces = self.detector.detect_faces(rgb_image, landmarks=False)

            # Convert to standard format with confidence
            results = []
            for face in faces:
                box = (
                    int(face["box"][0]),
                    int(face["box"][1]),
                    int(face["box"][2]),
                    int(face["box"][3]),
                )
                confidence = float(face["confidence"])
                results.append((box, confidence))

            return self._filter_detections_by_confidence(results)

        except Exception as e:
            return self._handle_detection_error(e, "MTCNN detection with confidence")

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Ensure image is in RGB format.

        Args:
            image: Input image array

        Returns:
            RGB image array
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB
            return image

        # Convert BGR to RGB
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def get_detector_info(self) -> Dict[str, Any]:
        """
        Get information about the detector.

        Returns:
            Dictionary with detector metadata
        """
        info = super().get_detector_info()
        info.update(
            {
                "thresholds": self.thresholds,
                "min_face_size": self.min_face_size,
                "device": self.device,
                "model_file": self.model_file,
                "detector_type": "MTCNN",
            }
        )

        return info
