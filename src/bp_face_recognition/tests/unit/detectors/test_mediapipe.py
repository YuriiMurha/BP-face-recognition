"""Test MediaPipe detector functionality (Unit Tests)."""

import pytest
from unittest.mock import patch, MagicMock
from bp_face_recognition.models.factory import RecognizerFactory


class TestMediaPipeDetector:
    """Test MediaPipe detector functionality using mocks."""

    def test_mtcnn_fallback(self):
        """Test MTCNN fallback works."""
        with patch("bp_face_recognition.models.factory.MTCNNDetector") as mock_mtcnn:
            mock_mtcnn.return_value = MagicMock()
            detector = RecognizerFactory.get_detector("mtcnn")
            assert detector is not None, "MTCNN detector should be created"
            mock_mtcnn.assert_called_once()

    def test_mediapipe_creation_success(self):
        """Test MediaPipe detector creation when successful."""
        with patch(
            "bp_face_recognition.models.factory.MediaPipeDetector"
        ) as mock_mediapipe:
            mock_mediapipe.return_value = MagicMock()
            detector = RecognizerFactory.get_detector("mediapipe")
            assert detector is not None, "MediaPipe detector should be created"
            mock_mediapipe.assert_called_once()

    def test_mediapipe_creation_failure(self):
        """Test MediaPipe detector creation failure handling."""
        with patch(
            "bp_face_recognition.models.factory.MediaPipeDetector",
            side_effect=Exception("ExternalFile"),
        ):
            with pytest.raises(Exception) as exc_info:
                RecognizerFactory.get_detector("mediapipe")
            assert "ExternalFile" in str(exc_info.value)

    def test_optimized_detector(self):
        """Test optimized detector selection."""
        with patch(
            "bp_face_recognition.models.factory.RecognizerFactory._create_optimal_detector"
        ) as mock_optimized:
            mock_optimized.return_value = MagicMock()
            detector = RecognizerFactory.get_optimized_detector()
            assert detector is not None, "Optimized detector should be created"
            mock_optimized.assert_called_once()

    def test_detector_types(self):
        """Test different detector types can be created."""
        with patch(
            "bp_face_recognition.models.factory.MTCNNDetector"
        ) as mock_mtcnn, patch(
            "bp_face_recognition.models.factory.HaarDetector"
        ) as mock_haar, patch(
            "bp_face_recognition.models.factory.DlibHOGDetector"
        ) as mock_dlib_hog:
            mock_mtcnn.return_value = MagicMock()
            mock_haar.return_value = MagicMock()
            mock_dlib_hog.return_value = MagicMock()

            mtcnn = RecognizerFactory.get_detector("mtcnn")
            haar = RecognizerFactory.get_detector("haar")
            dlib_hog = RecognizerFactory.get_detector("dlib_hog")

            assert mtcnn is not None, "MTCNN detector should work"
            assert haar is not None, "Haar detector should work"
            assert dlib_hog is not None, "Dlib HOG detector should work"

    def test_factory_error_handling(self):
        """Test factory error handling for invalid detector types."""
        with pytest.raises(ValueError):
            RecognizerFactory.get_detector("invalid_detector")
