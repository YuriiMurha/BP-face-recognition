import cv2
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional
from bp_face_recognition.models.interfaces import FaceRecognizer


class TFLiteRecognizer(FaceRecognizer):
    """
    TensorFlow Lite recognizer for optimized face recognition.
    Supports both float16 and int8 quantized models.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize TFLite recognizer.

        Args:
            model_path: Path to quantized TFLite model
        """
        if model_path is None:
            # Try to find a quantized model in models directory
            from bp_face_recognition.config.settings import settings

            models_dir = settings.MODELS_DIR

            # Look for float16 first, then int8, then dynamic
            for suffix in ["_float16.tflite", "_int8.tflite", "_dynamic.tflite"]:
                candidates = list(models_dir.glob(f"**/*{suffix}"))
                if candidates:
                    model_path = str(candidates[-1])  # Use the most recent
                    print(f"Auto-detected quantized model: {model_path}")
                    break

        if not model_path:
            raise ValueError("No TFLite model found")

        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.embedding_size = None

        print(f"Loading TFLite model: {model_path}")

        try:
            # Load TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Get embedding dimension from output shape
            if self.output_details:
                output_shape = self.output_details[0]["shape"]
                if len(output_shape) >= 2:
                    self.embedding_size = output_shape[1]
                else:
                    self.embedding_size = 512  # Default fallback

            print(f"✅ TFLite model loaded successfully")
            print(f"📊 Input shape: {self.input_details[0]['shape']}")
            print(f"📊 Output shape: {self.output_details[0]['shape']}")
            print(f"📊 Embedding size: {self.embedding_size}")

        except Exception as e:
            print(f"❌ Failed to load TFLite model: {e}")
            self.interpreter = None

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding using quantized TFLite model.

        Args:
            face_image: Input face image (BGR format)

        Returns:
            Face embedding vector
        """
        if self.interpreter is None:
            print("Warning: TFLite model not loaded, returning zeros")
            return np.zeros(512, dtype=np.float32)

        try:
            # Preprocess image (assuming model expects 224x224 RGB)
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))

            # Convert to float32 and normalize
            face_normalized = face_resized.astype(np.float32) / 255.0

            # Add batch dimension
            input_data = np.expand_dims(face_normalized, axis=0)

            # Set input tensor
            input_index = self.interpreter.get_input_details()[0]["index"]
            self.interpreter.set_tensor(input_index, input_data)

            # Run inference
            self.interpreter.invoke()

            # Get output embedding
            output_index = self.output_details[0]["index"]
            embedding = self.interpreter.get_tensor(output_index)

            # Flatten if needed (remove batch dimension)
            if embedding.ndim > 1:
                embedding = embedding.flatten()

            return embedding[0] if embedding.ndim > 1 else embedding

        except Exception as e:
            print(f"❌ Embedding extraction failed: {e}")
            return np.zeros(self.embedding_size or 512, dtype=np.float32)

    def get_model_info(self) -> dict:
        """
        Get information about the loaded TFLite model.

        Returns:
            Dictionary with model information
        """
        if self.interpreter is None:
            return {"error": "Model not loaded"}

        return {
            "model_path": self.model_path,
            "input_shape": self.input_details[0]["shape"]
            if self.input_details
            else None,
            "input_dtype": self.input_details[0]["dtype"]
            if self.input_details
            else None,
            "output_shape": self.output_details[0]["shape"]
            if self.output_details
            else None,
            "output_dtype": self.output_details[0]["dtype"]
            if self.output_details
            else None,
            "embedding_size": self.embedding_size,
            "tensor_details": {
                "input": self.input_details[0] if self.input_details else None,
                "output": self.output_details[0] if self.output_details else None,
            },
        }

    def __del__(self):
        """Clean up TFLite resources."""
        if hasattr(self, "interpreter"):
            self.interpreter.__del__()
