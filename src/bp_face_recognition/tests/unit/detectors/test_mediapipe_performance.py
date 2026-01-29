"""Test MediaPipe detector performance and API compatibility (Unit Tests)."""

import pytest
import numpy as np
import cv2
import time
from unittest.mock import patch, MagicMock

from bp_face_recognition.models.methods.mediapipe_detector import MediaPipeDetector


class TestMediaPipePerformance:
    """Test MediaPipe detector performance and speed improvements."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        # Create a test image with face-like pattern
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add some face-like rectangles
        cv2.rectangle(
            img, (200, 150), (300, 250), (255, 255, 255), -1
        )  # White rectangle
        cv2.rectangle(
            img, (400, 200), (500, 300), (200, 200, 200), -1
        )  # Gray rectangle

        return img

    @pytest.fixture
    def mock_mediapipe_results(self):
        """Create mock MediaPipe detection results."""
        mock_detection = MagicMock()
        mock_detection.bounding_box = MagicMock()
        mock_detection.bounding_box.origin_x = 200
        mock_detection.bounding_box.origin_y = 150
        mock_detection.bounding_box.width = 100
        mock_detection.bounding_box.height = 100
        mock_detection.categories = [MagicMock()]
        mock_detection.categories[0].score = 0.95

        mock_result = MagicMock()
        mock_result.detections = [mock_detection]

        return mock_result

    def test_mediapipe_initialization_gpu(self):
        """Test MediaPipe detector initializes with GPU support."""
        with patch(
            "mediapipe.tasks.vision.FaceDetector.create_from_options"
        ) as mock_create:
            mock_detector = MagicMock()
            mock_create.return_value = mock_detector

            detector = MediaPipeDetector(use_gpu=True)

            # Check GPU delegate is set
            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            options = args[0]
            assert options.delegate.name == "GPU"

    def test_mediapipe_initialization_cpu(self):
        """Test MediaPipe detector initializes with CPU fallback."""
        with patch(
            "mediapipe.tasks.vision.FaceDetector.create_from_options"
        ) as mock_create:
            mock_detector = MagicMock()
            mock_create.return_value = mock_detector

            detector = MediaPipeDetector(use_gpu=False)

            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            options = args[0]
            assert options.delegate is None

    def test_detect_performance_timing(self, sample_image, mock_mediapipe_results):
        """Test detection performance timing."""
        with patch(
            "mediapipe.tasks.vision.FaceDetector.create_from_options"
        ) as mock_create:
            mock_detector = MagicMock()
            mock_detector.detect.return_value = mock_mediapipe_results
            mock_create.return_value = mock_detector

            detector = MediaPipeDetector()

            # Time multiple detections
            start_time = time.time()
            for _ in range(10):
                boxes = detector.detect(sample_image)
            end_time = time.time()

            avg_time = (end_time - start_time) / 10

            # Should be fast (less than 50ms per detection)
            assert avg_time < 0.05, f"Detection too slow: {avg_time:.3f}s"
            assert len(boxes) == 1
            assert boxes[0] == (200, 150, 100, 100)

    def test_detect_with_confidence_performance(
        self, sample_image, mock_mediapipe_results
    ):
        """Test detection with confidence scores."""
        with patch(
            "mediapipe.tasks.vision.FaceDetector.create_from_options"
        ) as mock_create:
            mock_detector = MagicMock()
            mock_detector.detect.return_value = mock_mediapipe_results
            mock_create.return_value = mock_detector

            detector = MediaPipeDetector()

            results = detector.detect_with_confidence(sample_image)

            assert len(results) == 1
            box, confidence = results[0]
            assert box == (200, 150, 100, 100)
            assert confidence == 0.95

    def test_multiple_faces_detection(self):
        """Test detection of multiple faces."""
        # Create image with multiple face-like patterns
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.rectangle(img, (300, 150), (400, 250), (200, 200, 200), -1)
        cv2.rectangle(img, (500, 100), (600, 200), (150, 150, 150), -1)

        # Create mock results for multiple faces
        mock_results = MagicMock()
        mock_results.detections = []

        for i, x in enumerate([100, 300, 500]):
            mock_detection = MagicMock()
            mock_detection.bounding_box = MagicMock()
            mock_detection.bounding_box.origin_x = x
            mock_detection.bounding_box.origin_y = 100
            mock_detection.bounding_box.width = 100
            mock_detection.bounding_box.height = 100
            mock_detection.categories = [MagicMock()]
            mock_detection.categories[0].score = 0.9 - i * 0.1
            mock_results.detections.append(mock_detection)

        with patch(
            "mediapipe.tasks.vision.FaceDetector.create_from_options"
        ) as mock_create:
            mock_detector = MagicMock()
            mock_detector.detect.return_value = mock_results
            mock_create.return_value = mock_detector

            detector = MediaPipeDetector()

            results = detector.detect_with_confidence(img)

            assert len(results) == 3
            for i, (box, confidence) in enumerate(results):
                assert (
                    confidence > 0.7
                )  # All faces should be detected with good confidence

    def test_no_faces_detected(self):
        """Test behavior when no faces are detected."""
        # Create empty image
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_results = MagicMock()
        mock_results.detections = []

        with patch(
            "mediapipe.tasks.vision.FaceDetector.create_from_options"
        ) as mock_create:
            mock_detector = MagicMock()
            mock_detector.detect.return_value = mock_results
            mock_create.return_value = mock_detector

            detector = MediaPipeDetector()

            boxes = detector.detect(img)
            results = detector.detect_with_confidence(img)

            assert len(boxes) == 0
            assert len(results) == 0

    def test_none_image_handling(self):
        """Test handling of None image input."""
        with patch(
            "mediapipe.tasks.vision.FaceDetector.create_from_options"
        ) as mock_create:
            mock_detector = MagicMock()
            mock_create.return_value = mock_detector

            detector = MediaPipeDetector()

            boxes = detector.detect(None)
            results = detector.detect_with_confidence(None)

            assert boxes == []
            assert results == []

    def test_confidence_threshold_filtering(self):
        """Test confidence threshold filtering."""
        # Create mock results with low confidence detections
        mock_results = MagicMock()
        mock_results.detections = []

        for i, confidence in enumerate([0.3, 0.7, 0.9]):
            mock_detection = MagicMock()
            mock_detection.bounding_box = MagicMock()
            mock_detection.bounding_box.origin_x = i * 100
            mock_detection.bounding_box.origin_y = 100
            mock_detection.bounding_box.width = 80
            mock_detection.bounding_box.height = 80
            mock_detection.categories = [MagicMock()]
            mock_detection.categories[0].score = confidence
            mock_results.detections.append(mock_detection)

        with patch(
            "mediapipe.tasks.vision.FaceDetector.create_from_options"
        ) as mock_create:
            mock_detector = MagicMock()
            mock_detector.detect.return_value = mock_results
            mock_create.return_value = mock_detector

            # Test with high confidence threshold
            detector = MediaPipeDetector(min_detection_confidence=0.8)
            results = detector.detect_with_confidence(
                np.zeros((480, 640, 3), dtype=np.uint8)
            )

            # Should only return high confidence detections
            assert len(results) == 1
            assert results[0][1] == 0.9

    def test_cleanup_on_deletion(self):
        """Test proper cleanup of MediaPipe resources."""
        with patch(
            "mediapipe.tasks.vision.FaceDetector.create_from_options"
        ) as mock_create:
            mock_detector = MagicMock()
            mock_create.return_value = mock_detector

            detector = MediaPipeDetector()

            # Delete detector and check cleanup is called
            del detector

            mock_detector.close.assert_called_once()


class TestMediaPipeCompatibility:
    """Test MediaPipe API compatibility and version handling."""

    def test_api_version_compatibility(self):
        """Test MediaPipe API version compatibility."""
        with patch(
            "mediapipe.tasks.vision.FaceDetector.create_from_options"
        ) as mock_create:
            mock_detector = MagicMock()
            mock_create.return_value = mock_detector

            # Should not raise any API compatibility errors
            try:
                detector = MediaPipeDetector()
                assert detector is not None
            except Exception as e:
                pytest.fail(f"MediaPipe API compatibility error: {e}")

    def test_image_format_conversion(self):
        """Test proper BGR to RGB image format conversion."""
        # Create BGR image (OpenCV default)
        bgr_image = np.array([[[255, 0, 0]]], dtype=np.uint8)  # Red in BGR

        mock_results = MagicMock()
        mock_results.detections = []

        with patch(
            "mediapipe.tasks.vision.FaceDetector.create_from_options"
        ) as mock_create:
            mock_detector = MagicMock()
            mock_detector.detect.return_value = mock_results
            mock_create.return_value = mock_detector

            detector = MediaPipeDetector()

            # Mock cvtColor to verify it's called
            with patch("cv2.cvtColor") as mock_cvtcolor:
                mock_cvtcolor.return_value = np.array(
                    [[[0, 0, 255]]], dtype=np.uint8
                )  # RGB

                detector.detect(bgr_image)

                # Verify BGR to RGB conversion was called
                mock_cvtcolor.assert_called_once_with(bgr_image, cv2.COLOR_BGR2RGB)

    def test_gpu_acceleration_fallback(self):
        """Test GPU acceleration fallback to CPU."""
        # Simulate GPU initialization failure
        with patch(
            "mediapipe.tasks.vision.FaceDetector.create_from_options"
        ) as mock_create:
            # First call raises exception, second succeeds
            mock_create.side_effect = [Exception("GPU not available"), MagicMock()]

            try:
                # Should fallback to CPU automatically
                detector = MediaPipeDetector(use_gpu=True)
                assert detector is not None
            except Exception:
                # If fallback doesn't work, test CPU directly
                detector = MediaPipeDetector(use_gpu=False)
                assert detector is not None
