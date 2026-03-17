import numpy as np
import tensorflow as tf
from bp_face_recognition.vision.interfaces import FaceRecognizer


class MetricRecognizer(FaceRecognizer):
    """
    Recognizer that uses Euclidean Distance against a gallery of embeddings.
    Suitable for Open-Set recognition.
    """

    def __init__(self, model_path, gallery=None, threshold=0.5):
        """
        gallery: dict {name: [embeddings]}
        """
        self.model = tf.keras.models.load_model(model_path)
        self.gallery = gallery or {}
        self.threshold = threshold

    def recognize(self, face_img):
        embedding = self.get_embedding(face_img)

        if not self.gallery:
            return "Unknown", 0.0

        min_dist = float("inf")
        best_name = "Unknown"

        for name, known_embeddings in self.gallery.items():
            for known_emb in known_embeddings:
                # Euclidean distance on L2-normalized vectors
                dist = np.linalg.norm(embedding - known_emb)
                if dist < min_dist:
                    min_dist = dist
                    best_name = name

        if min_dist > self.threshold:
            return "Unknown", float(min_dist)

        # Convert distance to confidence (inverted)
        confidence = 1.0 - (min_dist / self.threshold) * 0.5
        return best_name, float(confidence)

    def get_embedding(self, face_img):
        img = tf.image.resize(face_img, (224, 224))
        img = np.expand_dims(img, axis=0) / 255.0
        return self.model.predict(img, verbose=0)[0]

    def add_to_gallery(self, name, face_img):
        emb = self.get_embedding(face_img)
        if name not in self.gallery:
            self.gallery[name] = []
        self.gallery[name].append(emb)
