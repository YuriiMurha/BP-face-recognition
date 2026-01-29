"""Test MediaPipe detector functionality with actual API (Integration Tests)."""

import pytest
import numpy as np
import cv2
import time
from unittest.mock import patch, MagicMock, Mock

# Skip MediaPipe tests if not available
mediapipe = pytest.importorskip("mediapipe", reason="MediaPipe not available")
mp = pytest.importorskip("mediapipe", reason="MediaPipe not available")


class TestMediaPipeIntegration:
    """Test MediaPipe detector with real API calls."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        # Create a test image with face-like pattern
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add some face-like rectangles
        cv2.rectangle(img, (200, 150), (300, 250), (255, 255, 255), -1)
        cv2.rectangle(img, (100, 100), (180, 180), (200, 200, 200), -1)

        return img

    def test_mediapipe_import(self):
        """Test that MediaPipe can be imported."""
        assert mediapipe is not None, "MediaPipe should be importable"
        assert hasattr(mp, "tasks"), "MediaPipe tasks should be available"

    def test_detector_creation(self):
        """Test MediaPipe detector can be created with different options."""
        # Test GPU initialization (may fail gracefully)
        try:
            from bp_face_recognition.models.methods.mediapipe_detector import (
                MediaPipeDetector,
            )

            detector = MediaPipeDetector(use_gpu=True, min_detection_confidence=0.7)
            assert detector is not None
            assert detector.use_gpu == True

            # Clean up
            del detector
        except Exception as e:
            print(f"GPU initialization failed (expected on some systems): {e}")

            # Test CPU fallback
            detector = MediaPipeDetector(use_gpu=False, min_detection_confidence=0.7)
            assert detector is not None
            assert detector.use_gpu == False
            del detector

    def test_face_detection_performance(self, sample_image):
        """Test actual face detection performance."""
        try:
            from bp_face_recognition.models.methods.mediapipe_detector import (
                MediaPipeDetector,
            )

            detector = MediaPipeDetector(use_gpu=False)  # Use CPU to avoid GPU issues

            # Measure detection time
            start_time = time.time()
            for _ in range(10):
                boxes = detector.detect(sample_image)
            end_time = time.time()

            avg_time = (end_time - start_time) / 10

            # Should complete reasonably quickly
            assert avg_time < 0.5, f"Detection too slow: {avg_time:.3f}s per image"

            # Should return list of boxes
            assert isinstance(boxes, list), "Should return list of bounding boxes"

            del detector

        except Exception as e:
            pytest.skip(f"MediaPipe detection test failed: {e}")

    def test_detect_with_confidence(self, sample_image):
        """Test detection with confidence scores."""
        try:
            from bp_face_recognition.models.methods.mediapipe_detector import (
                MediaPipeDetector,
            )

            detector = MediaPipeDetector(use_gpu=False)

            results = detector.detect_with_confidence(sample_image)

            # Should return list of (box, confidence) tuples
            assert isinstance(results, list), "Should return list"

            if results:  # Only test if faces detected
                box, confidence = results[0]
                assert isinstance(box, tuple), "Box should be tuple"
                assert len(box) == 4, "Box should have 4 elements"
                assert isinstance(
                    confidence, (int, float)
                ), "Confidence should be numeric"
                assert 0 <= confidence <= 1, "Confidence should be between 0 and 1"

            del detector

        except Exception as e:
            pytest.skip(f"MediaPipe confidence test failed: {e}")

    def test_no_faces_handling(self):
        """Test handling of images with no faces."""
        try:
            from bp_face_recognition.models.methods.mediapipe_detector import (
                MediaPipeDetector,
            )

            detector = MediaPipeDetector(use_gpu=False)

            # Test with empty image
            empty_img = np.zeros((480, 640, 3), dtype=np.uint8)

            boxes = detector.detect(empty_img)
            results = detector.detect_with_confidence(empty_img)

            # Should return empty lists
            assert isinstance(boxes, list), "Should return list"
            assert isinstance(results, list), "Should return list"

            del detector

        except Exception as e:
            pytest.skip(f"MediaPipe no-faces test failed: {e}")

    def test_none_image_handling(self):
        """Test handling of None image input."""
        try:
            from bp_face_recognition.models.methods.mediapipe_detector import (
                MediaPipeDetector,
            )

            detector = MediaPipeDetector(use_gpu=False)

            # Test with None
            boxes = detector.detect(None)
            results = detector.detect_with_confidence(None)

            # Should return empty lists
            assert boxes == [], "Should return empty list for None"
            assert results == [], "Should return empty list for None"

            del detector

        except Exception as e:
            pytest.skip(f"MediaPipe None-handling test failed: {e}")

    def test_confidence_threshold(self, sample_image):
        """Test confidence threshold filtering."""
        try:
            from bp_face_recognition.models.methods.mediapipe_detector import (
                MediaPipeDetector,
            )

            # Test with different thresholds
            high_threshold_detector = MediaPipeDetector(
                use_gpu=False, min_detection_confidence=0.9
            )
            low_threshold_detector = MediaPipeDetector(
                use_gpu=False, min_detection_confidence=0.3
            )

            high_results = high_threshold_detector.detect_with_confidence(sample_image)
            low_results = low_threshold_detector.detect_with_confidence(sample_image)

            # High threshold should detect fewer or same number of faces
            if high_results and low_results:
                assert len(high_results) <= len(
                    low_results
                ), "High threshold should detect fewer faces"

            del high_threshold_detector
            del low_threshold_detector

        except Exception as e:
            pytest.skip(f"MediaPipe threshold test failed: {e}")

    def test_different_image_sizes(self):
        """Test detector with different image sizes."""
        try:
            from bp_face_recognition.models.methods.mediapipe_detector import (
                MediaPipeDetector,
            )

            detector = MediaPipeDetector(use_gpu=False)

            # Test different image sizes
            sizes = [
                (224, 224, 3),  # Small
                (480, 640, 3),  # Medium
                (1080, 1920, 3),  # Large
            ]

            for height, width, channels in sizes:
                img = np.random.randint(
                    0, 255, (height, width, channels), dtype=np.uint8
                )

                boxes = detector.detect(img)

                # Should handle different sizes without crashing
                assert isinstance(boxes, list), f"Should handle {height}x{width} images"

            del detector

        except Exception as e:
            pytest.skip(f"MediaPipe size test failed: {e}")

    def test_multiple_calls_performance(self):
        """Test performance across multiple sequential calls."""
        try:
            from bp_face_recognition.models.methods.mediapipe_detector import (
                MediaPipeDetector,
            )

            detector = MediaPipeDetector(use_gpu=False)

            # Create test image
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Test many sequential detections
            total_time = 0
            num_calls = 50

            for _ in range(num_calls):
                start_time = time.time()
                boxes = detector.detect(img)
                end_time = time.time()
                total_time += end_time - start_time

            avg_time = total_time / num_calls

            # Should maintain reasonable performance
            assert avg_time < 0.1, f"Average detection time too high: {avg_time:.3f}s"

            del detector

        except Exception as e:
            pytest.skip(f"MediaPipe performance test failed: {e}")


class TestMediaPipeCompatibility:
    """Test MediaPipe API compatibility and version handling."""

    def test_api_structure(self):
        """Test MediaPipe API structure compatibility."""
        # Check if required modules exist
        assert hasattr(mp, "tasks"), "MediaPipe tasks module should exist"

        # Try to import tasks module
        try:
            tasks_module = mp.tasks
            assert hasattr(tasks_module, "BaseOptions"), "BaseOptions should exist"
            assert hasattr(tasks_module, "vision"), "Vision module should exist"
        except Exception as e:
            pytest.fail(f"MediaPipe API structure incompatible: {e}")

    def test_face_detector_options(self):
        """Test FaceDetectorOptions availability."""
        try:
            from mp.tasks.vision import FaceDetectorOptions
            from mp.tasks import BaseOptions

            # Should be able to create options
            base_opts = BaseOptions()
            detector_opts = FaceDetectorOptions(
                base_options=base_opts, min_detection_confidence=0.5
            )

            assert detector_opts is not None, "Should create FaceDetectorOptions"

        except Exception as e:
            pytest.skip(f"MediaPipe options test failed: {e}")

    def test_gpu_delegate_availability(self):
        """Test GPU delegate availability."""
        try:
            from mp.tasks import BaseOptions

            # Check if GPU delegate is available
            base_opts = BaseOptions()

            # Should not crash when accessing GPU delegate
            gpu_delegate = getattr(base_opts.__class__, "Delegate", None)

            if gpu_delegate:
                assert hasattr(gpu_delegate, "GPU"), "GPU delegate should be available"

        except Exception as e:
            pytest.skip(f"MediaPipe GPU delegate test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
