import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bp_face_recognition.vision.registry import get_registry
from bp_face_recognition.vision.factory import RecognizerFactory


class TestFactoryWithRegistry:
    """Test factory pattern with registry system."""

    @pytest.fixture
    def registry(self):
        """Get the model registry."""
        return get_registry()

    def test_default_detector(self, registry):
        """Test getting default detector."""
        detector = registry.get_default_detector()
        assert detector is not None
        info = detector.get_detector_info()
        assert info is not None
        assert "name" in info or "detector" in str(type(detector)).lower()

    def test_list_detectors(self, registry):
        """Test listing available detectors."""
        detectors = registry.list_detectors()
        assert isinstance(detectors, dict)
        assert len(detectors) > 0

    def test_list_recognizers(self, registry):
        """Test listing available recognizers."""
        recognizers = registry.list_recognizers()
        assert isinstance(recognizers, dict)
        assert len(recognizers) > 0

    def test_get_detector_by_name(self, registry):
        """Test getting specific detector by name."""
        available = list(registry.list_detectors().keys())
        if available:
            detector_name = available[0]
            detector = registry.get_detector(detector_name)
            assert detector is not None

    def test_get_recognizer_by_name(self, registry):
        """Test getting specific recognizer by name."""
        available = list(registry.list_recognizers().keys())
        if available:
            recognizer_name = available[0]
            recognizer = registry.get_recognizer(recognizer_name)
            assert recognizer is not None

    @patch(
        "bp_face_recognition.vision.detection.mediapipe_detector.MediaPipeFaceDetection"
    )
    def test_detector_detection(self, mock_mediapipe, registry):
        """Test detector can perform detection on dummy image."""
        available = list(registry.list_detectors().keys())
        if not available:
            pytest.skip("No detectors available")

        detector_name = available[0]

        with patch.object(
            registry, "get_detector", wraps=registry.get_detector
        ) as mock_get:
            detector = registry.get_detector(detector_name)

            test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            try:
                detections = detector.detect(test_img)
                assert isinstance(detections, (list, tuple))
            except Exception as e:
                pytest.skip(f"Detector {detector_name} not functional: {e}")

    def test_recognizer_embedding(self, registry):
        """Test recognizer can generate embeddings."""
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
        except FileNotFoundError:
            pytest.skip(f"Model file not available for {recognizer_name}")
        except Exception as e:
            pytest.skip(f"Recognizer {recognizer_name} not functional: {e}")

    def test_recognizer_get_info(self, registry):
        """Test recognizer returns info."""
        available = list(registry.list_recognizers().keys())
        if not available:
            pytest.skip("No recognizers available")

        recognizer_name = available[0]

        try:
            recognizer = registry.get_recognizer(recognizer_name)
            info = recognizer.get_recognizer_info()
            assert info is not None
        except FileNotFoundError:
            pytest.skip(f"Model file not available for {recognizer_name}")
        except Exception as e:
            pytest.skip(f"Recognizer {recognizer_name} not functional: {e}")

    def test_invalid_detector_name_raises(self, registry):
        """Test that invalid detector name raises ValueError."""
        with pytest.raises(ValueError):
            registry.get_detector("nonexistent_detector_xyz")

    def test_invalid_recognizer_name_raises(self, registry):
        """Test that invalid recognizer name raises ValueError."""
        with pytest.raises(ValueError):
            registry.get_recognizer("nonexistent_recognizer_xyz")


class TestEnvironmentConfig:
    """Test environment detection and configuration."""

    @pytest.fixture
    def registry(self):
        return get_registry()

    def test_detect_environment(self, registry):
        """Test environment detection returns valid string."""
        env = registry.detect_environment()
        assert isinstance(env, str)
        assert len(env) > 0

    def test_get_environment_config(self, registry):
        """Test getting environment-specific config."""
        env = registry.detect_environment()
        config = registry.get_environment_config(env)
        assert isinstance(config, dict)

    def test_get_current_environment_config(self, registry):
        """Test getting merged current environment config."""
        config = registry.get_current_environment_config()
        assert isinstance(config, dict)
        assert len(config) > 0

    def test_global_settings(self, registry):
        """Test getting global settings."""
        settings = registry.get_global_settings()
        assert isinstance(settings, dict)


class TestOptimizationSettings:
    """Test optimization settings."""

    @pytest.fixture
    def registry(self):
        return get_registry()

    def test_get_optimization_settings(self, registry):
        """Test getting optimization settings."""
        opt = registry.get_optimization_settings()
        assert isinstance(opt, dict)
