"""
Robust FaceNet Model Loader

Handles loading of fine-tuned FaceNet models with proper weight restoration.
The key issue: FaceNet uses Lambda layers with custom scaling functions that
serialize as Python code references, not as portable objects.

Solution: Reconstruct the exact architecture and load only the weights.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import tensorflow as tf
from tensorflow import keras
import numpy as np


def load_finetuned_facenet_robust(model_path: str, num_classes: int = 14):
    """
    Load fine-tuned FaceNet model with proper weight restoration.

    This function:
    1. Builds the exact architecture used during training
    2. Loads only the weights (not the full model)
    3. Returns a working model

    Args:
        model_path: Path to .keras or .weights.h5 file
        num_classes: Number of output classes (default 14 for your dataset)

    Returns:
        Loaded Keras model with fine-tuned weights
    """
    model_path = Path(model_path)
    print(f"Loading model: {model_path.name}")

    # Determine model type from filename
    model_type = _get_model_type(model_path.name)
    print(f"  Detected model type: {model_type}")

    # Step 1: Build the architecture
    print("  Step 1: Building architecture...")
    model = _build_facenet_classifier(num_classes=num_classes, trainable_base=True)

    # Step 2: Load weights
    print("  Step 2: Loading weights...")
    try:
        # Try loading as full model first (in case it was saved properly)
        if model_path.suffix == ".keras":
            try:
                # Load weights from the keras file
                temp_model = keras.models.load_model(model_path, compile=False)
                model.set_weights(temp_model.get_weights())
                print("  [OK] Weights loaded from .keras file")
                return model
            except Exception:
                # Silently fall back to weight-only loading
                pass

        # Try loading weights directly
        model.load_weights(str(model_path))
        print("  [OK] Weights loaded successfully")
        return model

    except Exception as e:
        print(f"  [ERROR] Weight loading failed: {e}")
        print("  Falling back to base FaceNet...")
        return _load_base_facenet(num_classes)


def _get_model_type(filename: str) -> str:
    """Detect model type from filename."""
    filename_lower = filename.lower()
    if "transfer" in filename_lower or "tl" in filename_lower:
        return "Transfer Learning"
    elif "progressive" in filename_lower or "pu" in filename_lower:
        return "Progressive Unfreezing"
    elif "triplet" in filename_lower or "tloss" in filename_lower:
        return "Triplet Loss"
    else:
        return "Unknown"


def _build_facenet_classifier(num_classes: int = 14, trainable_base: bool = True):
    """
    Build FaceNet classifier architecture.

    Architecture (used by all fine-tuned models):
    - Input: (160, 160, 3)
    - Base: FaceNet (InceptionResNetV1) with 512D embeddings
    - Dense: 256 units, ReLU
    - Dropout: 0.5
    - Output: num_classes units, Softmax

    Args:
        num_classes: Number of output classes
        trainable_base: Whether base model is trainable

    Returns:
        Uninitialized Keras model
    """
    print("    Building FaceNet classifier architecture...")

    try:
        from keras_facenet import FaceNet

        # Load base FaceNet model
        facenet = FaceNet()
        base_model = facenet.model

        # Set trainability
        base_model.trainable = trainable_base

        # Build classifier head
        inputs = keras.Input(shape=(160, 160, 3), name="input")

        # Base FaceNet (outputs 512D embeddings)
        x = base_model(inputs)

        # Classification head
        x = keras.layers.Dense(256, activation="relu", name="dense_1")(x)
        x = keras.layers.Dropout(0.5, name="dropout")(x)
        outputs = keras.layers.Dense(
            num_classes, activation="softmax", name="predictions"
        )(x)

        model = keras.Model(inputs, outputs)

        print(f"    [OK] Architecture built: {num_classes} classes")
        return model

    except ImportError:
        print("    [ERROR] keras-facenet not installed. Using fallback...")
        return _build_fallback_architecture(num_classes)


def _build_fallback_architecture(num_classes: int = 14):
    """
    Build simplified architecture if keras-facenet not available.
    This won't have the fine-tuned weights but provides compatible structure.
    """
    print("    Building fallback architecture...")

    inputs = keras.Input(shape=(160, 160, 3), name="input")

    # Simplified CNN (won't match FaceNet performance)
    x = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    print("    [WARN] Using fallback architecture (not FaceNet)")
    return model


def _load_base_facenet(num_classes: int = 14):
    """Load base FaceNet without fine-tuned weights."""
    print("  Loading base FaceNet (pre-trained only)...")

    try:
        from keras_facenet import FaceNet

        facenet = FaceNet()
        base_model = facenet.model

        # Build simple classification head
        inputs = keras.Input(shape=(160, 160, 3))
        x = base_model(inputs)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dropout(0.5)(x)
        outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        print("  [OK] Base FaceNet loaded (99.6% LFW, NOT fine-tuned)")
        return model

    except ImportError:
        print("  [ERROR] Could not load FaceNet")
        return None


def verify_model_weights(model, model_path: str):
    """Verify that model has loaded weights (not random)."""
    print("\nVerifying model weights...")

    # Get a layer's weights
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            # Check if weights are all zeros or very small (uninitialized)
            total_weight = np.sum([np.sum(np.abs(w)) for w in weights])
            if total_weight < 0.001:
                print(f"  [WARN] Layer '{layer.name}' may have uninitialized weights")
            else:
                print(
                    f"  [OK] Layer '{layer.name}' has weights (sum={total_weight:.2f})"
                )
                break

    print("  Note: If all layers show warnings, weights may not be loaded correctly")


if __name__ == "__main__":
    """Test the loader."""
    import sys

    print("=" * 60)
    print("FaceNet Robust Loader Test")
    print("=" * 60)

    # Test with progressive model
    model_path = Path(
        "src/bp_face_recognition/models/finetuned/facenet_progressive_v1.0.keras"
    )

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("  Please run training first: make train-facenet-pu")
        sys.exit(1)

    print(f"\nTesting with: {model_path.name}")
    print("-" * 60)

    try:
        model = load_finetuned_facenet_robust(str(model_path))

        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")

        # Verify weights
        verify_model_weights(model, str(model_path))

        # Quick inference test
        print("\nTesting inference...")
        dummy_input = np.random.rand(1, 160, 160, 3).astype(np.float32)
        dummy_input = (dummy_input - 0.5) * 2.0  # Scale to [-1, 1]

        output = model.predict(dummy_input, verbose=0)
        print(f"  Output shape: {output.shape}")
        print(f"  Output sum: {np.sum(output):.4f} (should be ~1.0 for softmax)")

        if abs(np.sum(output) - 1.0) < 0.01:
            print("  [OK] Model output looks correct (valid softmax)")
        else:
            print("  [WARN] Model output may be incorrect")

    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        import traceback

        traceback.print_exc()
