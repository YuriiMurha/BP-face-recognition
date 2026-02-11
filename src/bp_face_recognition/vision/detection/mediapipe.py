"""
MediaPipe Face Detector using new Tasks API

This is the migrated version of the MediaPipe detector with
proper model file handling and improved error management.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
from pathlib import Path
import logging
import time

from bp_face_recognition.vision.detection.base import BaseDetector, DetectionResult

# Import MediaPipe with better error handling
try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    mp = None

logger = logging.getLogger(__name__)


class MediaPipeDetector(BaseDetector):
    """
    MediaPipe face detector with intelligent GPU detection and fallback to OpenCV.

    Uses the new MediaPipe Tasks API with proper model file handling
    and configuration-driven model loading.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        use_gpu: bool = True,
        auto_gpu_detection: bool = True,
        model_file: Optional[str] = None,
    ):
        """
        Initialize face detector with intelligent GPU detection.

        Args:
            min_detection_confidence: Minimum confidence threshold
            use_gpu: Enable GPU delegate for acceleration (if auto_gpu_detection is False)
            auto_gpu_detection: Automatically detect and configure GPU support
            model_file: Optional path to MediaPipe model file
        """
        super().__init__(min_detection_confidence)

        self.min_detection_confidence = min_detection_confidence
        self.use_gpu_requested = use_gpu
        self.auto_gpu_detection = auto_gpu_detection
        self.model_file = model_file
        self.use_gpu = False
        self.gpu_delegate_test_result = None
        self.initialization_method = None

        if not MEDIAPIPE_AVAILABLE:
            self._initialize_opencv_fallback("MediaPipe not available")
            return

        # Model file handling - use default or custom model
        if model_file and Path(model_file).exists():
            self.model_file = model_file
            logger.info(f"Using custom model: {self.model_file}")
        else:
            # Look for model in standard location
            project_root = Path(__file__).parent.parent.parent.parent.parent
            default_model_path = (
                project_root
                / "src/bp_face_recognition/vision/detection/models/blaze_face_short_range.tflite"
            )

            if default_model_path.exists():
                self.model_file = str(default_model_path)
                logger.info(f"Using default model: {self.model_file}")
            else:
                self.model_file = None
                logger.warning(f"Default model not found: {default_model_path}")
                logger.info("Using built-in MediaPipe model")

        # GPU detection
        if auto_gpu_detection:
            self.use_gpu = self._detect_and_validate_gpu_support()
        else:
            self.use_gpu = use_gpu

        # Try MediaPipe initialization with model file
        self._initialize_mediapipe_with_model()

    def _detect_and_validate_gpu_support(self) -> bool:
        """
        Detect and validate GPU support for MediaPipe.

        Returns:
            bool: True if GPU is available and working
        """
        try:
            # Test basic MediaPipe GPU availability
            if not MEDIAPIPE_AVAILABLE:
                return False

            # Create base options with GPU delegate and built-in model
            base_options = mp.tasks.BaseOptions(
                delegate=mp.tasks.BaseOptions.Delegate.GPU
            )

            # Quick test by creating and closing detector
            test_options = mp.tasks.vision.FaceDetectorOptions(
                base_options=base_options, min_detection_confidence=0.5
            )

            # Test creation
            test_detector = mp.tasks.vision.FaceDetector.create_from_options(
                test_options
            )
            test_detector.close()

            logger.info("GPU delegate validation successful")
            return True

        except Exception as e:
            logger.warning(f"GPU validation failed: {e}")
            return False

    def _initialize_mediapipe_with_model(self) -> None:
        """
        Initialize MediaPipe detector with model file handling.
        """
        try:
            # Determine delegate
            delegate = (
                mp.tasks.BaseOptions.Delegate.GPU
                if self.use_gpu
                else mp.tasks.BaseOptions.Delegate.CPU
            )

            # Create base options
            base_options = mp.tasks.BaseOptions(delegate=delegate)

            # If we have a model file, configure it properly
            if self.model_file and Path(self.model_file).exists():
                logger.info(f"Loading model from file: {self.model_file}")
                # Read model file as bytes for external model loading
                with open(self.model_file, "rb") as f:
                    model_buffer = f.read()
                base_options = mp.tasks.BaseOptions(
                    delegate=delegate, model_asset_buffer=model_buffer
                )
            else:
                raise FileNotFoundError(
                    f"MediaPipe model file not found: {self.model_file}"
                )

            # Create detector options
            options = mp.tasks.vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=self.min_detection_confidence,
            )

            # Create detector
            self.detector = mp.tasks.vision.FaceDetector.create_from_options(options)
            self.use_opencv_fallback = False
            self.initialization_method = f"MediaPipe-{'GPU' if self.use_gpu else 'CPU'}"
            self._initialized = True

            logger.info(
                f"MediaPipe detector initialized with {self.initialization_method}"
            )

        except Exception as e:
            logger.error(f"MediaPipe initialization failed: {e}, attempting fallback")
            self._initialize_opencv_fallback(f"MediaPipe failed: {e}")

    def _initialize_opencv_fallback(self, reason: str) -> None:
        """
        Initialize OpenCV Haar Cascade fallback.

        Args:
            reason: Why fallback was triggered
        """
        try:
            from bp_face_recognition.vision.detection.haar_cascade import (
                HaarCascadeDetector,
            )

            self.detector = HaarCascadeDetector()
            self.use_opencv_fallback = True
            self.initialization_method = f"OpenCV-HaarCascade ({reason})"
            self._initialized = True

            logger.warning(f"Initialized OpenCV fallback: {reason}")

        except Exception as e:
            logger.error(f"Even OpenCV fallback failed: {e}")
            self.detector = None
            self.use_opencv_fallback = True
            self.initialization_method = f"Failed: {reason}"
            self._initialized = False

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image.

        Args:
            image: Input image array (BGR format expected)

        Returns:
            List of bounding boxes (x, y, w, h)
        """
        if not self._validate_image(image):
            return []

        if self.detector is None:
            logger.error("No detector available")
            return []

        start_time = time.time()

        try:
            if self.use_opencv_fallback:
                # Use OpenCV fallback
                boxes = self.detector.detect(image)
                return self._filter_detections_by_confidence(
                    [(box, 1.0) for box in boxes]
                )

            # Use MediaPipe detector
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # Detect faces
            detection_result = self.detector.detect(mp_image)

            # Convert MediaPipe results to bounding boxes
            boxes = []
            confidences = []

            if detection_result.detections:
                for detection in detection_result.detections:
                    # MediaPipe provides bounding box in normalized coordinates
                    bbox = detection.bounding_box
                    h, w, y, x = bbox.height, bbox.width, bbox.y, bbox.x

                    # Convert to (x, y, w, h) format
                    box = (int(x), int(y), int(w), int(h))
                    boxes.append(box)
                    confidences.append(
                        float(detection.categories[0].score)
                        if detection.categories
                        else 1.0
                    )

            processing_time = time.time() - start_time
            self._log_detection_stats(
                [(box, conf) for box, conf in zip(boxes, confidences)], image.shape
            )

            # Return filtered boxes based on confidence threshold
            return self._filter_detections_by_confidence(
                [(box, conf) for box, conf in zip(boxes, confidences)]
            )

        except Exception as e:
            return self._handle_detection_error(e, "detection")

    def detect_with_confidence(
        self, image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces with confidence scores.

        Args:
            image: Input image array

        Returns:
            List of tuples ((x, y, w, h), confidence)
        """
        if not self._validate_image(image):
            return []

        if self.detector is None:
            logger.error("No detector available")
            return []

        start_time = time.time()

        try:
            if self.use_opencv_fallback:
                # Use OpenCV fallback
                boxes = self.detector.detect(image)
                confidences = [1.0] * len(boxes)
                return list(zip(boxes, confidences))

            # Use MediaPipe detector
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            detection_result = self.detector.detect(mp_image)

            boxes = []
            confidences = []

            if detection_result.detections:
                for detection in detection_result.detections:
                    bbox = detection.bounding_box
                    h, w, y, x = bbox.height, bbox.width, bbox.y, bbox.x
                    box = (int(x), int(y), int(w), int(h))
                    boxes.append(box)

                    if detection.categories:
                        confidences.append(float(detection.categories[0].score))
                    else:
                        confidences.append(1.0)

            processing_time = time.time() - start_time
            result = list(zip(boxes, confidences))
            self._log_detection_stats(result, image.shape)

            return result

        except Exception as e:
            return self._handle_detection_error(e, "detection_with_confidence")

    def get_detector_info(self) -> dict:
        """
        Get information about the detector.

        Returns:
            Dictionary with detector metadata
        """
        info = super().get_detector_info()
        info.update(
            {
                "use_gpu": self.use_gpu,
                "model_file": self.model_file,
                "use_opencv_fallback": self.use_opencv_fallback,
                "mediapipe_available": MEDIAPIPE_AVAILABLE,
                "gpu_requested": self.use_gpu_requested,
                "auto_gpu_detection": self.auto_gpu_detection,
            }
        )

        return info
