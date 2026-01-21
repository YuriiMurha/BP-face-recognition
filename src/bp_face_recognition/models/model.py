import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from typing import List, Tuple, Optional
from bp_face_recognition.config.settings import settings
from bp_face_recognition.models.interfaces import FaceDetector, FaceRecognizer
from bp_face_recognition.models.methods.mtcnn_detector import MTCNNDetector
from bp_face_recognition.models.methods.facenet_recognizer import FaceNetRecognizer


class CustomFaceRecognizer(FaceRecognizer):
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = str(settings.MODELS_DIR / "seccam_2_final.keras")

        try:
            full_model = tf.keras.models.load_model(model_path, compile=False)
            # Create a sub-model that outputs the features before the classification head
            # In our build_model, the layer before output is a Dropout or Dense(512)
            # Let's find the 'dropout' or the dense layer before the end
            self.model = Model(
                inputs=full_model.input, outputs=full_model.layers[-3].output
            )
        except Exception as e:
            print(f"Warning: Could not load custom model from {model_path}: {e}")
            self.model = None

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(512)  # Default size for our custom model features
        face_image = cv2.resize(face_image, (224, 224))
        face_image = tf.image.convert_image_dtype(face_image, tf.float32)[None, ...]
        return self.model.predict(face_image, verbose=0)[0]


class FaceTracker:
    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        recognizer: Optional[FaceRecognizer] = None,
    ):
        self.detector = detector or MTCNNDetector()
        self.recognizer = recognizer or CustomFaceRecognizer()

    def detect_faces(
        self, image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        # This wrapper might still be useful if we want to return confidence
        if hasattr(self.detector, "detect_with_confidence"):
            return self.detector.detect_with_confidence(image)  # type: ignore
        else:
            boxes = self.detector.detect(image)
            return [(box, 1.0) for box in boxes]  # Default confidence 1.0

    def get_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        return self.recognizer.get_embedding(face_crop)

    def evaluate_embeddings(self, test_dataset, threshold=0.7):
        # Implementation remains similar but uses self.recognizer
        embeddings_db = {}
        correct = 0
        total = 0

        for images, labels in test_dataset:
            # Note: recognizer might need to handle batch for efficiency
            # For now keeping it simple as per original code
            for i in range(images.shape[0]):
                img = images[i].numpy()
                emb = self.get_embedding(img)
                label = labels[i].numpy()

                if label not in embeddings_db:
                    embeddings_db[label] = emb
                else:
                    known_emb = embeddings_db[label]
                    similarity = np.dot(emb, known_emb) / (
                        np.linalg.norm(emb) * np.linalg.norm(known_emb)
                    )
                    if similarity > threshold:
                        correct += 1
                    total += 1

        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy at threshold {threshold}: {accuracy:.4f}")
        return accuracy
