"""
TensorFlow Lite Recognizer for optimized face recognition.
Supports float16 and int8 quantized models.
"""

import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Optional, Tuple

from bp_face_recognition.vision.interfaces import FaceRecognizer
from bp_face_recognition.vision.recognition.base import BaseRecognizer

logger = logging.getLogger(__name__)


class TFLiteRecognizer(BaseRecognizer):
    """
    TensorFlow Lite recognizer for face recognition.
    Supports float16, int8, and dynamic range quantization.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
    ):
        """
        Initialize TFLite recognizer.

        Args:
            model_path: Path to TFLite model file
            input_size: Expected input image size (height, width)
            normalize: Whether to normalize pixel values
        """
        super().__init__(input_size=input_size, normalize=normalize)
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """Load TFLite model."""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self._initialized = True
            logger.info(f"Loaded TFLite model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            raise

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Get face embedding from face image.

        Args:
            face_image: Input face image (expected BGR, already cropped)

        Returns:
            Face embedding vector
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized")

        # Preprocess using base class logic (handles BGR to RGB, resize, normalize)
        img = self._preprocess_face(face_image)

        # Add batch dimension if needed
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)

        # Set input
        self.interpreter.set_tensor(self.input_details[0]["index"], img)

        # Invoke
        self.interpreter.invoke()

        # Get output
        embedding = self.interpreter.get_tensor(self.output_details[0]["index"])

        return embedding.flatten()

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Deprecated: Use _preprocess_face from BaseRecognizer instead.
        """
        return self._preprocess_face(image)

    def get_input_size(self) -> Tuple[int, int]:
        """Get model input size."""
        return self.input_size
