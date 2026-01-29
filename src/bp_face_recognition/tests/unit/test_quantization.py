"""Test quantization functionality (Unit Tests)."""

import pytest
import tempfile
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.quantize_model import ModelQuantizer, quantize_model


class TestModelQuantizer:
    """Test ModelQuantizer class functionality."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer"),
                tf.keras.layers.Rescaling(1.0 / 255),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(512, name="embedding_layer"),
            ]
        )
        return model

    @pytest.fixture
    def temp_model_file(self, simple_model):
        """Create a temporary model file."""
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
            temp_path = f.name

        simple_model.save(temp_path)
        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def quantizer(self, temp_model_file):
        """Create ModelQuantizer instance."""
        return ModelQuantizer(temp_model_file)

    def test_quantizer_initialization(self, temp_model_file):
        """Test quantizer initializes correctly."""
        quantizer = ModelQuantizer(temp_model_file)
        assert quantizer.model is not None
        assert quantizer.model_path.exists()
        assert quantizer.output_dir.exists()

    def test_representative_dataset_generator(self, quantizer):
        """Test representative dataset generator."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy image files
            for i in range(3):
                dummy_path = Path(temp_dir) / f"dummy_{i}.jpg"
                dummy_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                from PIL import Image

                Image.fromarray(dummy_array).save(dummy_path)

            generator = quantizer.create_representative_dataset(temp_dir, num_samples=5)

            # Test generator produces correct number of samples
            samples = list(generator())
            assert len(samples) == 5

            # Test each sample has correct shape
            for sample in samples:
                assert len(sample) == 1
                assert sample[0].shape == (224, 224, 3)
                assert sample[0].dtype == np.float32

    def test_representative_dataset_no_images(self, quantizer):
        """Test representative dataset with no images falls back to dummy data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = quantizer.create_representative_dataset(temp_dir, num_samples=3)
            samples = list(generator())
            assert len(samples) == 3

    def test_float16_quantization(self, quantizer):
        """Test float16 quantization works."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = quantizer.quantize_float16()

            # Check output file exists
            assert os.path.exists(output_path)

            # Check it's a valid TFLite model
            interpreter = tf.lite.Interpreter(model_path=output_path)
            interpreter.allocate_tensors()

            # Check input/output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            assert len(input_details) > 0
            assert len(output_details) > 0

    def test_dynamic_range_quantization(self, quantizer):
        """Test dynamic range quantization works."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = quantizer.quantize_dynamic_range()

            # Check output file exists
            assert os.path.exists(output_path)

            # Check it's a valid TFLite model
            interpreter = tf.lite.Interpreter(model_path=output_path)
            interpreter.allocate_tensors()

    def test_int8_quantization_with_dataset(self, quantizer):
        """Test int8 quantization with representative dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy image files
            for i in range(3):
                dummy_path = Path(temp_dir) / f"dummy_{i}.jpg"
                dummy_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                from PIL import Image

                Image.fromarray(dummy_array).save(dummy_path)

            output_path = quantizer.quantize_int8(temp_dir)

            # Check output file exists
            assert os.path.exists(output_path)

            # Check it's a valid TFLite model
            interpreter = tf.lite.Interpreter(model_path=output_path)
            interpreter.allocate_tensors()

            # Check input/output types for int8 quantization
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Should be uint8 for int8 quantization
            assert input_details[0]["dtype"] == np.uint8
            assert output_details[0]["dtype"] == np.uint8

    def test_int8_quantization_no_dataset_raises_error(self, quantizer):
        """Test int8 quantization without dataset raises error."""
        with pytest.raises(ValueError, match="Dataset directory required"):
            quantizer.quantize_int8(None)

    def test_quantize_model_function(self, temp_model_file):
        """Test main quantize_model function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test float16
            output_path = quantize_model(
                temp_model_file, "float16", output_dir=temp_dir
            )
            assert os.path.exists(output_path)
            assert output_path.endswith("_float16.tflite")

    def test_quantize_model_invalid_type(self, temp_model_file):
        """Test quantize_model with invalid type raises error."""
        with pytest.raises(ValueError, match="Supported types"):
            quantize_model(temp_model_file, "invalid_type")

    def test_model_size_comparison(self, quantizer):
        """Test that quantization reduces model size."""
        original_size = quantizer.model_path.stat().st_size

        output_path = quantizer.quantize_float16()
        quantized_size = os.path.getsize(output_path)

        # Quantized model should be smaller
        assert quantized_size < original_size


class TestQuantizationIntegration:
    """Test quantization integration with TFLite recognizer."""

    @pytest.fixture
    def quantized_model_path(self):
        """Create a quantized model for testing."""
        # Create simple model
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer"),
                tf.keras.layers.Rescaling(1.0 / 255),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(512, name="embedding_layer"),
            ]
        )

        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
            model_path = f.name

        model.save(model_path)

        # Quantize it
        quantized_path = quantize_model(model_path, "float16")

        # Cleanup original
        os.unlink(model_path)

        yield quantized_path

        # Cleanup quantized
        if os.path.exists(quantized_path):
            os.unlink(quantized_path)

    def test_tflite_recognizer_loads_quantized_model(self, quantized_model_path):
        """Test TFLiteRecognizer can load quantized models."""
        from bp_face_recognition.models.methods.tflite_recognizer import (
            TFLiteRecognizer,
        )

        recognizer = TFLiteRecognizer(quantized_model_path)
        assert recognizer.interpreter is not None
        assert recognizer.embedding_size is not None

    def test_tflite_recognizer_inference(self, quantized_model_path):
        """Test TFLiteRecognizer can perform inference."""
        from bp_face_recognition.models.methods.tflite_recognizer import (
            TFLiteRecognizer,
        )

        recognizer = TFLiteRecognizer(quantized_model_path)

        # Create dummy face image
        face_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        embedding = recognizer.get_embedding(face_image)
        assert embedding is not None
        assert len(embedding) == 512  # Should match model output

    def test_tflite_recognizer_model_info(self, quantized_model_path):
        """Test TFLiteRecognizer model info."""
        from bp_face_recognition.models.methods.tflite_recognizer import (
            TFLiteRecognizer,
        )

        recognizer = TFLiteRecognizer(quantized_model_path)
        info = recognizer.get_model_info()

        assert "model_path" in info
        assert "input_shape" in info
        assert "output_shape" in info
        assert "embedding_size" in info
