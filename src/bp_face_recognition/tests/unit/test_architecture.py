import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from bp_face_recognition.models.factory import RecognizerFactory
from bp_face_recognition.models.recognition_service import RecognitionService
from bp_face_recognition.models.model import CustomFaceRecognizer
from bp_face_recognition.database.database import FaceDatabase


@pytest.fixture
def mock_tf_model():
    with patch("tensorflow.keras.models.load_model") as mock_load:
        mock_model = MagicMock()
        # Mock layer with name 'face_embedding'
        mock_layer = MagicMock()
        mock_layer.output = MagicMock()
        mock_model.get_layer.return_value = mock_layer
        mock_load.return_value = mock_model
        yield mock_load, mock_model


@pytest.fixture
def mock_database():
    db = MagicMock(spec=FaceDatabase)
    db.get_all_embeddings.return_value = [(1, np.random.rand(512).astype(np.float32))]
    return db


def test_recognizer_factory():
    # Test factory instantiation for 'facenet'
    with patch("bp_face_recognition.models.factory.FaceNetRecognizer") as mock_facenet:
        _ = RecognizerFactory.get_recognizer("facenet")
        mock_facenet.assert_called_once()

    # Test factory instantiation for 'custom'
    with patch("bp_face_recognition.models.model.CustomFaceRecognizer") as mock_custom:
        _ = RecognizerFactory.get_recognizer("custom")
        mock_custom.assert_called_once()


def test_custom_face_recognizer_named_layer(mock_tf_model):
    mock_load, mock_model = mock_tf_model
    _ = CustomFaceRecognizer(model_path="dummy.keras")

    # Verify it tried to get the layer by name
    mock_model.get_layer.assert_called_with("face_embedding")


def test_recognition_service_logic(mock_database):
    mock_tracker = MagicMock()
    # Mock detection of one face
    mock_tracker.detect_faces.return_value = [((10, 10, 50, 50), 0.9)]
    # Mock embedding extraction
    mock_tracker.get_embedding.return_value = np.random.rand(512).astype(np.float32)

    service = RecognitionService(tracker=mock_tracker, database=mock_database)

    # Create a dummy frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    results = service.process_frame(frame)

    assert len(results) == 1
    assert "box" in results[0]
    assert "label" in results[0]
    assert results[0]["confidence"] == 0.9


def test_recognition_service_stranger(mock_database):
    mock_tracker = MagicMock()
    mock_tracker.detect_faces.return_value = [((10, 10, 50, 50), 0.9)]
    # Return an embedding very different from the one in mock_database
    mock_tracker.get_embedding.return_value = np.ones(512, dtype=np.float32) * 100

    # Set high threshold to ensure it's a stranger
    service = RecognitionService(
        tracker=mock_tracker, database=mock_database, threshold=0.1
    )

    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Mock add_face to return a new ID
    mock_database.add_face.return_value = 99

    results = service.process_frame(frame)

    assert results[0]["label"] == "99"
    mock_database.add_face.assert_called_once()
