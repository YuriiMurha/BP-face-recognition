"""
Keras Metric Recognizer for custom-trained embedding models.
Supports models trained with Triplet Loss or other metric learning approaches.
"""

import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from bp_face_recognition.vision.interfaces import FaceRecognizer
from bp_face_recognition.vision.recognition.base import BaseRecognizer

logger = logging.getLogger(__name__)


def create_embedding_model(
    backbone_type="EfficientNetB0", embedding_dim=128, input_shape=(224, 224, 3)
):
    """
    Creates a feature extractor model for Metric Learning.
    Uses ImageNet pre-trained weights.
    """
    if backbone_type == "EfficientNetB0":
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    elif backbone_type == "MobileNetV3Small":
        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_type}")

    # Freeze backbone
    base_model.trainable = False

    # Create model with functional API for better compatibility
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(embedding_dim, name="embedding_dense")(x)
    # Manual L2 normalization using Lambda with explicit output_shape
    outputs = layers.Lambda(
        lambda v: v / (tf.norm(v, axis=-1, keepdims=True) + 1e-8),
        output_shape=(embedding_dim,),
        name="l2_norm",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="metric_model")
    return model


class KerasMetricRecognizer(BaseRecognizer):
    """
    Recognizer using custom-trained Keras models for metric learning.
    Uses pre-trained EfficientNet/MobileNet as feature extractor.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        backbone: str = "EfficientNetB0",
        embedding_dim: int = 128,
        weights_path: Optional[str] = None,
    ):
        """
        Initialize Keras Metric Recognizer.

        Args:
            model_path: Path to trained Keras model (.keras or .h5) - currently unused
            input_size: Expected input image size (height, width)
            normalize: Whether to normalize pixel values
            backbone: Backbone architecture (EfficientNetB0 or MobileNetV3Small)
            embedding_dim: Embedding dimension (128 or 64)
            weights_path: Optional path to weights file - currently unused
        """
        super().__init__(input_size=input_size, normalize=normalize)
        self.model_path = model_path
        self.weights_path = weights_path
        self.backbone = backbone
        self._embedding_size = embedding_dim
        self.model = None

        # Always create fresh model for testing pipeline
        self._create_model(backbone, embedding_dim)

    def _create_model(self, backbone: str, embedding_dim: int) -> None:
        """Create or load metric model."""
        try:
            # First create the model architecture
            self.model = create_embedding_model(
                backbone_type=backbone, embedding_dim=embedding_dim
            )

            # Build model with dummy input to ensure weights are loadable
            dummy_input = tf.zeros((1, 224, 224, 3))
            _ = self.model(dummy_input, training=False)

            # Try to load trained weights
            model_path = self.model_path
            weights_path = self.weights_path
            weights_loaded = False

            if weights_path and Path(weights_path).exists():
                try:
                    self.model.load_weights(weights_path)
                    logger.info(f"SUCCESS: Loaded trained weights from {weights_path}")
                    weights_loaded = True
                except Exception as e:
                    logger.warning(f"Could not load weights from {weights_path}: {e}")
            elif model_path and Path(model_path).exists():
                try:
                    # Try loading with custom_objects for L2NormalizeLayer
                    from bp_face_recognition.vision.training.metric.model import (
                        L2NormalizeLayer,
                    )

                    self.model = tf.keras.models.load_model(
                        model_path,
                        custom_objects={"L2NormalizeLayer": L2NormalizeLayer},
                    )
                    logger.info(f"SUCCESS: Loaded trained model from {model_path}")
                    weights_loaded = True
                except Exception as e:
                    logger.warning(f"Could not load model from {model_path}: {e}")

            self.model.trainable = False
            self._embedding_size = embedding_dim
            self._initialized = True

            if weights_loaded:
                logger.info(
                    f"Model ready with TRAINED weights, backbone: {backbone}, embedding: {embedding_dim}D"
                )
            else:
                logger.info(
                    f"Model ready with IMAGENET weights (trained weights not loaded), backbone: {backbone}, embedding: {embedding_dim}D"
                )
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Get face embedding from face image.

        Args:
            face_image: Input face image (expected BGR, already cropped)

        Returns:
            128D or 64D embedding vector (L2-normalized)
        """
        if not self._initialized or self.model is None:
            logger.error("Model not initialized")
            return np.array([])

        try:
            # Preprocess using base class logic
            img = self._preprocess_face(face_image)

            # Add batch dimension if needed
            if len(img.shape) == 3:
                img = np.expand_dims(img, axis=0)

            # Get embedding
            embedding = self.model(img, training=False)

            # Flatten and return
            return embedding.numpy().flatten()

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return np.array([])

    def get_embedding_size(self) -> int:
        """Get embedding dimension."""
        return self._embedding_size

    def get_recognizer_info(self) -> Dict[str, Any]:
        info = super().get_recognizer_info()
        info.update(
            {
                "model_type": "Keras-MetricLearning",
                "embedding_size": self._embedding_size,
                "backbone": self.backbone,
                "model_path": self.model_path,
                "weights_path": self.weights_path,
                "normalization": "L2-normalized",
                "note": "Using ImageNet pre-trained weights - custom weights need retraining",
            }
        )
        return info
