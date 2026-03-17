"""Finetuned Recognizer for FaceNet Models

Wraps fine-tuned Keras FaceNet models for integration with the registry system.
Supports Transfer Learning (TL), Progressive Unfreezing (PU), and Triplet Loss (TLoss) models.

Usage:
    from bp_face_recognition.vision.recognition.finetuned_recognizer import FinetunedRecognizer

    recognizer = FinetunedRecognizer(
        model_path="models/finetuned/facenet_progressive_v1.0.keras",
        class_names=["Stranger_1", "Stranger_2", ..., "Yurii"]
    )

    identity, confidence = recognizer.recognize(face_image)
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Union, Optional
from PIL import Image

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from bp_face_recognition.vision.recognition.base import BaseRecognizer


class FinetunedRecognizer(BaseRecognizer):
    """
    Recognizer wrapper for fine-tuned FaceNet models.

    Supports:
    - Transfer Learning (TL): Frozen base + trainable head
    - Progressive Unfreezing (PU): Gradual unfreezing with 4 phases
    - Triplet Loss (TLoss): Metric learning with embeddings

    Attributes:
        model: Loaded Keras model
        class_names: List of identity names
        input_size: Model input size (default: 160x160)
        preprocessing: Preprocessing method (facenet_standard: [-1, 1])
    """

    def __init__(
        self,
        model_path: str,
        class_names: Optional[List[str]] = None,
        input_size: Tuple[int, int] = (160, 160),
        preprocessing: str = "facenet_standard",
        **kwargs,
    ):
        """
        Initialize FinetunedRecognizer.

        Args:
            model_path: Path to .keras model file
            class_names: List of class names (if None, loads from dataset_info)
            input_size: Input image size (default: 160, 160)
            preprocessing: Preprocessing method (facenet_standard scales to [-1, 1])
            **kwargs: Additional arguments passed to BaseRecognizer
        """
        super().__init__(**kwargs)

        self.model_path = Path(model_path)
        self.input_size = input_size
        self.preprocessing = preprocessing

        # Load class names
        if class_names is None:
            self.class_names = self._load_class_names()
        else:
            self.class_names = class_names

        # Load model
        self.model = self._load_model()

        print(f"✓ FinetunedRecognizer initialized")
        print(f"  Model: {self.model_path.name}")
        print(f"  Classes: {len(self.class_names)}")
        print(f"  Input size: {self.input_size}")

    def _load_model(self) -> tf.keras.Model:
        """Load Keras model from file with FaceNet compatibility."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Use the robust loader which properly handles weights
        try:
            from bp_face_recognition.utils.facenet_loader import (
                load_finetuned_facenet_robust,
            )

            model = load_finetuned_facenet_robust(
                str(self.model_path), num_classes=len(self.class_names)
            )
            print(f"OK Model loaded: {self.model_path.name}")
            return model
        except Exception as e:
            print(f"ERROR Failed to load model: {e}")
            raise RuntimeError(f"Could not load model from {self.model_path}")

    def _load_class_names(self) -> List[str]:
        """Load class names from model directory or use defaults."""
        # Try to load from dataset_info in model directory
        dataset_info_path = self.model_path.parent / "dataset_info.json"

        if dataset_info_path.exists():
            with open(dataset_info_path) as f:
                info = json.load(f)
                if "class_names" in info:
                    return info["class_names"]

        # Default class names based on training dataset
        # These should match the 14 identities used in training
        return [
            "Stranger_1",
            "Stranger_10",
            "Stranger_11",
            "Stranger_12",
            "Stranger_14",
            "Stranger_2",
            "Stranger_3",
            "Stranger_4",
            "Stranger_5",
            "Stranger_6",
            "Stranger_7",
            "Stranger_8",
            "Stranger_9",
            "Yurii",
        ]

    def preprocess_image(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """
        Preprocess image for FaceNet model.

        FaceNet preprocessing:
        1. Resize to 160x160
        2. Convert to float32
        3. Normalize to [0, 1]
        4. Scale to [-1, 1] (facenet_standard)

        Args:
            image: Input image (numpy array, PIL Image, or path)

        Returns:
            Preprocessed image array of shape (1, 160, 160, 3)
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
            image = np.array(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure numpy array
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"Image must be numpy array, PIL Image, or path. Got: {type(image)}"
            )

        # Resize to input size
        if image.shape[:2] != self.input_size:
            image = tf.image.resize(image, self.input_size)
            image = image.numpy()

        # Ensure float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Normalize based on preprocessing type
        if self.preprocessing == "facenet_standard":
            # Normalize to [0, 1] then scale to [-1, 1]
            if image.max() > 1.0:
                image = image / 255.0
            image = (image - 0.5) * 2.0  # Scale to [-1, 1]
        elif self.preprocessing == "normalized":
            # Just normalize to [0, 1]
            if image.max() > 1.0:
                image = image / 255.0

        # Add batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        return image

    def recognize(
        self, face_image: Union[np.ndarray, str, Path], return_confidence: bool = True
    ) -> Union[str, Tuple[str, float]]:
        """
        Recognize identity from face image.

        Args:
            face_image: Face image (numpy array, PIL Image, or path)
            return_confidence: Whether to return confidence score

        Returns:
            Identity name, or (identity, confidence) tuple
        """
        # Preprocess
        processed = self.preprocess_image(face_image)

        # Predict
        predictions = self.model.predict(processed, verbose=0)

        # Get top prediction
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        identity = self.class_names[class_idx]

        if return_confidence:
            return identity, confidence
        return identity

    def recognize_batch(
        self,
        face_images: List[Union[np.ndarray, str, Path]],
        return_confidences: bool = True,
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """
        Recognize multiple faces in batch.

        Args:
            face_images: List of face images
            return_confidences: Whether to return confidence scores

        Returns:
            List of identities, or list of (identity, confidence) tuples
        """
        results = []
        for img in face_images:
            result = self.recognize(img, return_confidence=return_confidences)
            results.append(result)
        return results

    def get_embedding(self, face_image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """
        Get embedding representation of face (for triplet loss models).

        Note: For classification models (TL, PU), this returns the model output
        before softmax. For TLoss models, returns the actual embedding.

        Args:
            face_image: Face image

        Returns:
            Embedding vector
        """
        processed = self.preprocess_image(face_image)

        # For models with embedding layer, extract from appropriate layer
        # For now, return model output (works for classification models)
        embedding = self.model.predict(processed, verbose=0)

        return embedding[0]

    def benchmark(self, num_iterations: int = 100) -> dict:
        """
        Benchmark inference speed.

        Args:
            num_iterations: Number of inference iterations

        Returns:
            Dictionary with timing statistics
        """
        import time

        # Create dummy input
        dummy_input = np.random.rand(1, *self.input_size, 3).astype(np.float32)
        dummy_input = (dummy_input - 0.5) * 2.0  # Scale to [-1, 1]

        # Warmup
        for _ in range(10):
            self.model.predict(dummy_input, verbose=0)

        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            self.model.predict(dummy_input, verbose=0)
            times.append(time.time() - start)

        return {
            "avg_time_ms": np.mean(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "min_time_ms": np.min(times) * 1000,
            "max_time_ms": np.max(times) * 1000,
            "fps": 1.0 / np.mean(times),
        }

    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return len(self.class_names)

    def __repr__(self):
        return (
            f"FinetunedRecognizer("
            f"model='{self.model_path.name}', "
            f"classes={self.num_classes}, "
            f"input_size={self.input_size})"
        )


# Factory function for easy creation from config
def create_finetuned_recognizer(
    model_file: str, config: dict = None
) -> FinetunedRecognizer:
    """
    Factory function to create FinetunedRecognizer from config.

    Args:
        model_file: Path to model file (relative to project root)
        config: Configuration dictionary with keys like 'input_size', 'class_names'

    Returns:
        FinetunedRecognizer instance
    """
    # Build full path
    project_root = Path(__file__).parent.parent.parent.parent
    model_path = project_root / model_file

    # Extract config
    kwargs = {}
    if config:
        if "input_size" in config:
            kwargs["input_size"] = tuple(config["input_size"])
        if "class_names" in config.get("metadata", {}):
            kwargs["class_names"] = config["metadata"]["class_names"]

    return FinetunedRecognizer(model_path=str(model_path), **kwargs)


if __name__ == "__main__":
    # Test the recognizer
    print("Testing FinetunedRecognizer...")

    # Example with progressive model
    model_path = Path(
        "src/bp_face_recognition/models/finetuned/facenet_progressive_v1.0.keras"
    )

    if model_path.exists():
        recognizer = FinetunedRecognizer(model_path=str(model_path))

        # Benchmark
        print("\nBenchmarking...")
        stats = recognizer.benchmark(num_iterations=50)
        print(f"  Average: {stats['avg_time_ms']:.1f} ms")
        print(f"  FPS: {stats['fps']:.1f}")

        print("\n✓ FinetunedRecognizer test complete!")
    else:
        print(f"⚠ Model not found: {model_path}")
        print("  Run training first: make train-facenet-pu")
