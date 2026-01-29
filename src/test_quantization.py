"""
Test script to create a simple model and test quantization.
"""

import tensorflow as tf
import numpy as np
from scripts.quantize_model import quantize_model


def create_simple_model():
    """Create a simple CNN model for testing quantization."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer"),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(512, name="embedding_layer"),  # Face embedding output
        ]
    )

    return model


def test_quantization():
    """Test quantization on a simple model."""
    print("Creating simple test model...")
    model = create_simple_model()

    # Save the model
    test_model_path = "test_model.keras"
    model.save(test_model_path)
    print(f"Test model saved: {test_model_path}")

    try:
        # Test float16 quantization
        print("\nTesting float16 quantization...")
        float16_path = quantize_model(test_model_path, "float16")
        print(f"Float16 quantized model: {float16_path}")

        # Test dynamic range quantization
        print("\nTesting dynamic range quantization...")
        dynamic_path = quantize_model(test_model_path, "dynamic")
        print(f"Dynamic quantized model: {dynamic_path}")

        print("\nQuantization tests completed successfully!")

    except Exception as e:
        print(f"Quantization test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_quantization()
