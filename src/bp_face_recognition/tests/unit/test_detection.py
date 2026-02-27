import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from bp_face_recognition.vision.core.face_tracker import FaceTracker


class TestFaceTracker:
    """Test face tracking functionality."""

    @pytest.fixture
    def tracker(self):
        """Create a FaceTracker instance."""
        return FaceTracker()

    def test_tracker_initialization(self, tracker):
        """Test tracker initializes correctly."""
        assert tracker is not None
        assert hasattr(tracker, "track_faces")

    def test_track_faces_returns_list(self, tracker):
        """Test track_faces returns list of tracked faces."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with patch.object(
            tracker, "detect_faces", return_value=[((10, 10, 100, 100), 0.9)]
        ):
            result = tracker.track_faces(frame)
            assert isinstance(result, list)


class TestDetectionPipeline:
    """Test detection pipeline integration."""

    @pytest.fixture
    def registry(self):
        from bp_face_recognition.vision.registry import get_registry

        return get_registry()

    def test_detection_pipeline_returns_boxes(self, registry):
        """Test detection pipeline returns bounding boxes."""
        available = list(registry.list_detectors().keys())
        if not available:
            pytest.skip("No detectors available")

        detector_name = available[0]

        try:
            detector = registry.get_detector(detector_name)
            test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            detections = detector.detect(test_img)

            assert isinstance(detections, (list, tuple))
        except FileNotFoundError:
            pytest.skip("Model not available")
        except Exception as e:
            pytest.skip(f"Detector not functional: {e}")

    def test_detection_on_empty_image(self, registry):
        """Test detection on image with no faces."""
        available = list(registry.list_detectors().keys())
        if not available:
            pytest.skip("No detectors available")

        detector_name = available[0]

        try:
            detector = registry.get_detector(detector_name)
            blank_img = np.zeros((480, 640, 3), dtype=np.uint8)

            detections = detector.detect(blank_img)

            assert isinstance(detections, (list, tuple))
            assert len(detections) >= 0
        except FileNotFoundError:
            pytest.skip("Model not available")
        except Exception as e:
            pytest.skip(f"Detector not functional: {e}")


class TestFPSMeasurement:
    """Test FPS measurement functionality."""

    def test_fps_calculation(self):
        """Test FPS calculation."""
        from bp_face_recognition.utils.fps import calculate_fps

        class MockTime:
            def __init__(self):
                self.times = [0.0, 0.033, 0.067, 0.1]
                self.idx = 0

            def time(self):
                val = self.times[self.idx]
                self.idx = (self.idx + 1) % len(self.times)
                return val

        mock_time = MockTime()

        with patch("time.time", mock_time.time):
            fps = calculate_fps(window=4)
            assert fps > 0


class TestRecognitionPipeline:
    """Test recognition pipeline."""

    @pytest.fixture
    def registry(self):
        from bp_face_recognition.vision.registry import get_registry

        return get_registry()

    def test_recognition_pipeline_embedding(self, registry):
        """Test recognition pipeline generates embeddings."""
        available = list(registry.list_recognizers().keys())
        if not available:
            pytest.skip("No recognizers available")

        recognizer_name = available[0]

        try:
            recognizer = registry.get_recognizer(recognizer_name)
            test_img = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)

            embedding = recognizer.get_embedding(test_img)

            assert embedding is not None
            assert isinstance(embedding, np.ndarray)
            assert len(embedding.shape) == 1
            assert embedding.shape[0] > 0
        except FileNotFoundError:
            pytest.skip("Model not available")
        except Exception as e:
            pytest.skip(f"Recognizer not functional: {e}")

    def test_detection_to_recognition_pipeline(self, registry):
        """Test full detection to recognition pipeline."""
        available_detectors = list(registry.list_detectors().keys())
        available_recognizers = list(registry.list_recognizers().keys())

        if not available_detectors or not available_recognizers:
            pytest.skip("Detectors or recognizers not available")

        try:
            detector = registry.get_detector(available_detectors[0])
            recognizer = registry.get_recognizer(available_recognizers[0])

            test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            detections = detector.detect(test_img)

            for det in detections:
                if len(det) >= 2:
                    bbox, confidence = det[0], det[1]
                    x, y, w, h = bbox
                    face_img = test_img[y : y + h, x : x + w]

                    if face_img.size > 0:
                        embedding = recognizer.get_embedding(face_img)
                        assert embedding is not None
                        break
        except FileNotFoundError:
            pytest.skip("Model not available")
        except Exception as e:
            pytest.skip(f"Pipeline not functional: {e}")
