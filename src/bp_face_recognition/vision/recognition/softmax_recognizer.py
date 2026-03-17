import numpy as np
import tensorflow as tf
from bp_face_recognition.vision.interfaces import FaceRecognizer


class SoftmaxRecognizer(FaceRecognizer):
    """
    Recognizer that uses Softmax output probabilities and Entropy
    to detect "Unknown" individuals.
    """

    def __init__(self, model_path, labels, threshold=0.7, entropy_threshold=0.5):
        self.model = tf.keras.models.load_model(model_path)
        self.labels = labels
        self.threshold = threshold
        self.entropy_threshold = entropy_threshold

    def recognize(self, face_img):
        # Preprocess
        img = tf.image.resize(face_img, (224, 224))
        img = np.expand_dims(img, axis=0) / 255.0

        # Predict
        probs = self.model.predict(img, verbose=0)[0]

        # Calculate Max Probability and Entropy
        max_prob = np.max(probs)
        label_idx = np.argmax(probs)

        # Entropy calculation: -sum(p * log(p))
        # Higher entropy = more uncertainty
        entropy = -np.sum(probs * np.log(probs + 1e-10)) / np.log(len(probs))

        if max_prob < self.threshold or entropy > self.entropy_threshold:
            return "Unknown", 1.0 - max_prob

        return self.labels[label_idx], float(max_prob)

    def get_embedding(self, face_img):
        # For softmax models, the "embedding" is the layer before the softmax head
        # We'd need to reconstruct the model to get this reliably.
        # For now, we return the softmax vector itself as a "soft embedding"
        img = tf.image.resize(face_img, (224, 224))
        img = np.expand_dims(img, axis=0) / 255.0
        return self.model.predict(img, verbose=0)[0]
