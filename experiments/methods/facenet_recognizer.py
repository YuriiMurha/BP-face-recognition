import numpy as np
import tensorflow as tf
import cv2
from typing import Optional
from bp_face_recognition.models.interfaces import FaceRecognizer
from bp_face_recognition.config.settings import settings


class FaceNetRecognizer(FaceRecognizer):
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = str(settings.MODELS_DIR / "FaceNet/facenet_keras.h5")

        # Load the pre-trained FaceNet model with safe_mode=False because it contains Lambda layers
        self.model = tf.keras.models.load_model(
            model_path, compile=False, safe_mode=False
        )

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract 128D embedding using FaceNet.
        """
        # FaceNet expects 160x160 RGB images
        face_image = cv2.resize(face_image, (160, 160))
        if face_image.shape[-1] == 4:  # Handle BGRA
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGRA2RGB)
        elif len(face_image.shape) == 2:  # Handle Grayscale
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        else:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Standardize pixel values
        face_pixels = face_image.astype("float32")
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std

        samples = np.expand_dims(face_pixels, axis=0)
        yhat = self.model.predict(samples, verbose=0)
        return yhat[0]
