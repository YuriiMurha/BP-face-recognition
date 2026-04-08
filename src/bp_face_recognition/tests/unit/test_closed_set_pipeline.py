"""Tests for ClosedSetPipelineService."""

import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Pre-mock face_recognition before any bp_face_recognition imports trigger it
if "face_recognition" not in sys.modules:
    sys.modules["face_recognition"] = MagicMock()
if "face_recognition_models" not in sys.modules:
    sys.modules["face_recognition_models"] = MagicMock()

from bp_face_recognition.services.closed_set_pipeline_service import (  # noqa: E402
    ClosedSetPipelineService,
)


CLASS_NAMES_14 = [
    "Stranger_1", "Stranger_10", "Stranger_11", "Stranger_12",
    "Stranger_14", "Stranger_2", "Stranger_3", "Stranger_4",
    "Stranger_5", "Stranger_6", "Stranger_7", "Stranger_8",
    "Stranger_9", "Yurii",
]


def _make_pipeline(detector, recognizer, threshold=0.7):
    """Create ClosedSetPipelineService with injected mocks (bypass factory)."""
    with patch(
        "bp_face_recognition.services.closed_set_pipeline_service.RecognizerFactory"
    ) as mock_factory:
        mock_factory.get_detector.return_value = detector
        mock_factory.get_recognizer.return_value = recognizer
        return ClosedSetPipelineService(
            detector_type="mediapipe_v1",
            recognizer_type="facenet_pu",
            confidence_threshold=threshold,
        )


@pytest.fixture
def mock_detector():
    """Create a mock face detector."""
    detector = MagicMock()
    detector.detect_with_confidence.return_value = [
        ((50, 50, 100, 100), 0.95),
    ]
    return detector


@pytest.fixture
def mock_recognizer():
    """Create a mock recognizer with recognize() method."""
    recognizer = MagicMock()
    recognizer.recognize.return_value = ("Yurii", 0.95)
    recognizer.class_names = CLASS_NAMES_14
    return recognizer


@pytest.fixture
def dummy_image():
    """Create a dummy BGR image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def pipeline(mock_detector, mock_recognizer):
    """Create ClosedSetPipelineService with mocked components."""
    return _make_pipeline(mock_detector, mock_recognizer)


class TestClosedSetPipelineInit:
    """Tests for ClosedSetPipelineService initialization."""

    def test_init_valid_recognizer(self, pipeline):
        """Verify pipeline initializes with a recognizer that has recognize()."""
        assert pipeline.detector_type == "mediapipe_v1"
        assert pipeline.recognizer_type == "facenet_pu"
        assert pipeline.confidence_threshold == 0.7
        assert not hasattr(pipeline, "database_service")

    def test_init_invalid_recognizer(self):
        """Verify ValueError for recognizer without recognize() method."""
        bad_recognizer = MagicMock(spec=[])  # No recognize attribute

        with pytest.raises(ValueError, match="does not support closed-set"):
            _make_pipeline(MagicMock(), bad_recognizer)


class TestClosedSetProcessImage:
    """Tests for process_image()."""

    def test_process_image_recognized(self, pipeline, dummy_image):
        """Face above threshold should be recognized."""
        result = pipeline.process_image(dummy_image)

        assert result["success"] is True
        assert result["mode"] == "closed-set"
        assert result["recognition_result"]["num_faces"] == 1
        assert result["recognition_result"]["num_recognized"] == 1

        face = result["recognition_result"]["faces"][0]
        assert face["identity"] == "Yurii"
        assert face["recognized"] is True
        assert face["recognition_confidence"] == 0.95

    def test_process_image_below_threshold(self, mock_detector, dummy_image):
        """Face below confidence threshold should be Unknown."""
        low_conf_recognizer = MagicMock()
        low_conf_recognizer.recognize.return_value = ("Stranger_1", 0.4)
        low_conf_recognizer.class_names = ["Stranger_1", "Yurii"]

        service = _make_pipeline(mock_detector, low_conf_recognizer, threshold=0.7)
        result = service.process_image(dummy_image)

        face = result["recognition_result"]["faces"][0]
        assert face["identity"] == "Unknown"
        assert face["recognized"] is False
        assert face["raw_prediction"] == "Stranger_1"
        assert result["recognition_result"]["num_recognized"] == 0

    def test_process_image_no_faces(self, pipeline, dummy_image):
        """No faces detected should return empty results."""
        pipeline.detector.detect_with_confidence.return_value = []

        result = pipeline.process_image(dummy_image)

        assert result["success"] is True
        assert result["recognition_result"]["num_faces"] == 0
        assert result["recognition_result"]["num_recognized"] == 0
        assert result["recognition_result"]["faces"] == []

    def test_result_format_compatible(self, pipeline, dummy_image):
        """Result should have same top-level keys as PipelineService for UI compatibility."""
        result = pipeline.process_image(dummy_image)

        # Keys that main.py depends on
        assert "success" in result
        assert "detection_result" in result
        assert "recognition_result" in result
        assert "processing_time" in result

        # recognition_result structure
        rr = result["recognition_result"]
        assert "faces" in rr
        assert "num_faces" in rr
        assert "num_recognized" in rr

        # Each face should have expected keys
        face = rr["faces"][0]
        assert "box" in face
        assert "identity" in face
        assert "recognized" in face
        assert "recognition_confidence" in face


class TestClosedSetClassNames:
    """Tests for class name access."""

    def test_get_class_names(self, pipeline):
        """Should return the 14 training classes."""
        names = pipeline.get_class_names()
        assert len(names) == 14
        assert "Yurii" in names
        assert "Stranger_1" in names

    def test_get_system_info(self, pipeline):
        """System info should include closed-set metadata."""
        info = pipeline.get_system_info()
        assert info["mode"] == "closed-set"
        assert info["num_classes"] == 14
        assert info["detector_type"] == "mediapipe_v1"
        assert info["recognizer_type"] == "facenet_pu"
