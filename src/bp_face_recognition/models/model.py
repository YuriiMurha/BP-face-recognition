import cv2
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from bp_face_recognition.config.settings import settings
from bp_face_recognition.models.interfaces import FaceDetector, FaceRecognizer
from bp_face_recognition.models.methods.mtcnn_detector import MTCNNDetector

class CustomFaceRecognizer(FaceRecognizer):
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Default path from settings if available or hardcoded fallback
            model_path = str(settings.ROOT_DIR / "custom_model.keras")
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def get_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        face_crop = cv2.resize(face_crop, (224, 224))  # VGG16 input size
        face_crop = tf.image.convert_image_dtype(face_crop, tf.float32)[None, ...]
        return self.model.predict(face_crop)[0]

class FaceTracker:
    def __init__(self, detector: FaceDetector = None, recognizer: FaceRecognizer = None):
        self.detector = detector or MTCNNDetector()
        self.recognizer = recognizer or CustomFaceRecognizer()

    def detect_faces(self, image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        # This wrapper might still be useful if we want to return confidence
        if hasattr(self.detector, 'detect_with_confidence'):
             return self.detector.detect_with_confidence(image)
        else:
            boxes = self.detector.detect(image)
            return [(box, 1.0) for box in boxes] # Default confidence 1.0

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
                    similarity = np.dot(emb, known_emb) / (np.linalg.norm(emb) * np.linalg.norm(known_emb))
                    if similarity > threshold:
                        correct += 1
                    total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy at threshold {threshold}: {accuracy:.4f}")
        return accuracy
