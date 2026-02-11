import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
from pathlib import Path
import logging

from bp_face_recognition.utils.gpu import (
    check_mediapipe_gpu_compatibility,
    get_gpu_info,
    test_mediapipe_gpu_delegate,
    print_gpu_diagnostics,
)

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class MediaPipeDetector:
    """
    MediaPipe face detector with intelligent GPU detection and fallback to OpenCV.
    """

    def __init__(
        self, min_detection_confidence=0.5, use_gpu=True, auto_gpu_detection=True
    ):
        """
        Initialize face detector with intelligent GPU detection.

        Args:
            min_detection_confidence: Minimum confidence threshold
            use_gpu: Enable GPU delegate for acceleration (if auto_gpu_detection is False)
            auto_gpu_detection: Automatically detect and configure GPU support
        """
        self.min_detection_confidence = min_detection_confidence
        self.use_gpu_requested = use_gpu
        self.auto_gpu_detection = auto_gpu_detection
        self.use_gpu = False
        self.gpu_delegate_test_result = None
        self.initialization_method = None

        # Try MediaPipe first
        if not MEDIAPIPE_AVAILABLE:
            self._initialize_opencv_fallback("MediaPipe not available")
            return

        # Let MediaPipe handle model auto-download (most reliable approach)
        use_built_in = True
        model_path = None
        logger.info("Using MediaPipe's built-in face detection model (auto-download)")

        # Intelligent GPU detection
        if auto_gpu_detection:
            self.use_gpu = self._detect_and_validate_gpu_support()
        else:
            self.use_gpu = use_gpu

        try:
            # Create MediaPipe detector with determined delegate
            delegate = (
                mp.tasks.BaseOptions.Delegate.GPU
                if self.use_gpu
                else mp.tasks.BaseOptions.Delegate.CPU
            )

            # Use MediaPipe's built-in model (auto-download)
            base_options = mp.tasks.BaseOptions(
                delegate=delegate,
            )

            options = mp.tasks.vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=min_detection_confidence,
            )

            self.detector = mp.tasks.vision.FaceDetector.create_from_options(options)
            self.use_opencv_fallback = False
            self.initialization_method = f"MediaPipe-{'GPU' if self.use_gpu else 'CPU'}"

            logger.info(
                f"MediaPipe detector initialized successfully with {self.initialization_method}"
            )
            return

        except Exception as e:
            logger.error(f"MediaPipe initialization failed: {e}, attempting fallback")
            if self.use_gpu:
                # Try again with CPU if GPU failed
                try:
                    logger.info("Attempting MediaPipe CPU fallback after GPU failure")

                    # Use MediaPipe's built-in model for fallback too
                    base_options = mp.tasks.BaseOptions(
                        delegate=mp.tasks.BaseOptions.Delegate.CPU,
                    )

                    options = mp.tasks.vision.FaceDetectorOptions(
                        base_options=base_options,
                        min_detection_confidence=min_detection_confidence,
                    )

                    self.detector = mp.tasks.vision.FaceDetector.create_from_options(
                        options
                    )
                    self.use_opencv_fallback = False
                    self.use_gpu = False
                    self.initialization_method = "MediaPipe-CPU (GPU fallback)"
                    logger.info("MediaPipe CPU fallback successful")
                    return
                except Exception as e2:
                    logger.error(f"MediaPipe CPU fallback also failed: {e2}")

        # Final fallback to OpenCV Haar Cascade
        self._initialize_opencv_fallback(
            "MediaPipe not available or all initialization attempts failed"
        )

    def _detect_and_validate_gpu_support(self) -> bool:
        """
        Detect and validate GPU support for MediaPipe.

        Returns:
            bool: True if GPU support is available and working
        """
        if not self.use_gpu_requested:
            logger.info("GPU disabled by user request")
            return False

        # Check MediaPipe GPU compatibility
        compatible, reason = check_mediapipe_gpu_compatibility()
        if not compatible:
            logger.info(f"MediaPipe GPU not compatible: {reason}")
            return False

        # Test actual GPU delegate creation
        success, error = test_mediapipe_gpu_delegate()
        self.gpu_delegate_test_result = (success, error)

        if success:
            logger.info("MediaPipe GPU delegate validation successful")
            return True
        else:
            logger.warning(f"MediaPipe GPU delegate validation failed: {error}")
            return False

    def _initialize_opencv_fallback(self, reason: str):
        """Initialize OpenCV Haar Cascade fallback."""
        self.use_opencv_fallback = True
        self.initialization_method = f"OpenCV-HaarCascade ({reason})"

        logger.info(f"Initializing OpenCV Haar Cascade fallback: {reason}")

        # Load OpenCV's pre-trained face detector
        face_cascade_path = "haarcascade_frontalface_default.xml"

        # Try to find OpenCV data directory
        try:
            cv2_path = os.path.dirname(cv2.__file__)
            possible_paths = [
                os.path.join(cv2_path, "data", "haarcascade_frontalface_default.xml"),
                os.path.join(
                    cv2_path, "haarcascades", "haarcascade_frontalface_default.xml"
                ),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    face_cascade_path = path
                    break
        except Exception:
            pass  # Keep default path

        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError("Failed to load OpenCV face cascade classifier")

        logger.info("OpenCV Haar Cascade fallback initialized successfully")

    def get_gpu_status(self) -> dict:
        """
        Get detailed GPU status information.

        Returns:
            dict: GPU status details
        """
        gpu_info = get_gpu_info()
        status = {
            "initialization_method": self.initialization_method,
            "use_gpu": self.use_gpu,
            "use_gpu_requested": self.use_gpu_requested,
            "auto_gpu_detection": self.auto_gpu_detection,
            "gpu_delegate_test_result": self.gpu_delegate_test_result,
            "gpu_info": gpu_info,
        }
        return status

        # Load OpenCV's pre-trained face detector
        face_cascade_path = "haarcascade_frontalface_default.xml"

        # Try to find OpenCV data directory
        try:
            cv2_path = os.path.dirname(cv2.__file__)
            possible_paths = [
                os.path.join(cv2_path, "data", "haarcascade_frontalface_default.xml"),
                os.path.join(
                    cv2_path, "haarcascades", "haarcascade_frontalface_default.xml"
                ),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    face_cascade_path = path
                    break
        except Exception:
            pass  # Keep default path

        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError("Failed to load OpenCV face cascade classifier")

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using MediaPipe or OpenCV fallback.

        Args:
            image: Input image in BGR format (OpenCV default)

        Returns:
            List of bounding boxes as (x, y, w, h)
        """
        if image is None:
            return []

        if hasattr(self, "use_opencv_fallback") and self.use_opencv_fallback:
            # Use OpenCV Haar Cascade
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
        else:
            # Use MediaPipe Tasks API
            if not MEDIAPIPE_AVAILABLE:
                logger.error(
                    "MediaPipe not available but detector initialized without OpenCV fallback"
                )
                return []

            rgb_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            )

            detection_result = self.detector.detect(rgb_image)

            boxes = []
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                boxes.append(
                    (
                        int(bbox.origin_x),
                        int(bbox.origin_y),
                        int(bbox.width),
                        int(bbox.height),
                    )
                )

            return boxes

    def detect_with_confidence(
        self, image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces and return bounding boxes with confidence scores.

        Args:
            image: Input image in BGR format

        Returns:
            List of tuples: ((x, y, w, h), confidence)
        """
        if image is None:
            return []

        if hasattr(self, "use_opencv_fallback") and self.use_opencv_fallback:
            # Use OpenCV Haar Cascade (no confidence, use default)
            faces = self.detect(image)
            return [
                ((x, y, w, h), float(self.min_detection_confidence))
                for x, y, w, h in faces
            ]
        else:
            # Use MediaPipe Tasks API
            if not MEDIAPIPE_AVAILABLE:
                logger.error(
                    "MediaPipe not available but detector initialized without OpenCV fallback"
                )
                return []

            rgb_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            )

            detection_result = self.detector.detect(rgb_image)

            results = []
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                box = (
                    int(bbox.origin_x),
                    int(bbox.origin_y),
                    int(bbox.width),
                    int(bbox.height),
                )
                confidence = detection.categories[0].score
                results.append((box, float(confidence)))

            return results

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "use_opencv_fallback") and self.use_opencv_fallback:
            # OpenCV cascade doesn't need explicit cleanup
            pass
        elif hasattr(self, "detector"):
            # MediaPipe detector cleanup
            self.detector.close()
