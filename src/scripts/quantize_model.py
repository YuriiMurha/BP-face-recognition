"""
TensorFlow Lite quantization script for face recognition models.
Supports float16 and int8 quantization strategies.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Any, Callable, Optional
from pathlib import Path


class ModelQuantizer:
    """Handle model quantization for TensorFlow Lite conversion."""

    def __init__(self, model_path: str, output_dir: Optional[str] = None):
        """
        Initialize quantizer.

        Args:
            model_path: Path to trained Keras model
            output_dir: Directory to save quantized models
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir) if output_dir else self.model_path.parent

        print(f"Loading model: {self.model_path}")

        # Load the model
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        print(f"Model loaded successfully")

    def create_representative_dataset(
        self, dataset_dir: str, num_samples: int = 100
    ) -> Callable[[], Any]:
        """
        Create a representative dataset for quantization.

        Args:
            dataset_dir: Directory containing validation images
            num_samples: Number of samples to generate

        Returns:
            Generator function yielding representative data
        """

        def data_generator():
            # Get image files from dataset
            dataset_path = Path(dataset_dir)
            if not dataset_path.exists():
                print(f"Warning: Dataset directory {dataset_dir} not found")
                # Use dummy data as fallback
                for _ in range(num_samples):
                    yield [np.random.randint(0, 255, (224, 224, 3)).astype(np.float32)]
                return

            image_files = list(dataset_path.glob("**/*.jpg")) + list(
                dataset_path.glob("**/*.png")
            )
            if not image_files:
                print(f"Warning: No images found in {dataset_dir}")
                # Use dummy data
                for _ in range(num_samples):
                    yield [np.random.randint(0, 255, (224, 224, 3)).astype(np.float32)]
                return

            # Use available images (cycling if needed)
            for i in range(num_samples):
                img_path = image_files[i % len(image_files)]
                try:
                    img = tf.keras.preprocessing.image.load_img(
                        img_path, target_size=(224, 224)
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array.astype(np.float32) / 255.0
                    yield [img_array]
                except Exception as e:
                    print(f"Warning: Could not load {img_path}: {e}")
                    yield [np.random.randint(0, 255, (224, 224, 3)).astype(np.float32)]

        return data_generator

    def quantize_float16(self, output_suffix: str = "_float16.tflite") -> str:
        """
        Apply float16 quantization for minimal accuracy loss.

        Args:
            output_suffix: Suffix for output file

        Returns:
            Path to quantized model
        """
        print("\\n" + "=" * 50)
        print("FLOAT16 QUANTIZATION")
        print("=" * 50)

        output_path = self.output_dir / (self.model_path.stem + output_suffix)

        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Apply float16 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        print("Converting to float16...")
        quantized_model = converter.convert()

        # Save the quantized model
        with open(output_path, "wb") as f:
            f.write(quantized_model)

        print(f"Float16 model saved: {output_path}")

        # Get model size info
        original_size = self.model_path.stat().st_size
        quantized_size = output_path.stat().st_size
        compression_ratio = (1 - quantized_size / original_size) * 100

        print(
            f"Size reduction: {compression_ratio:.1f}% ({original_size} -> {quantized_size} bytes)"
        )

        return str(output_path)

    def quantize_int8(
        self, dataset_dir: str, output_suffix: str = "_int8.tflite"
    ) -> str:
        """
        Apply full integer quantization for maximum compression.

        Args:
            dataset_dir: Directory containing representative data
            output_suffix: Suffix for output file

        Returns:
            Path to quantized model
        """
        print("\\n" + "=" * 50)
        print("INT8 QUANTIZATION")
        print("=" * 50)

        output_path = self.output_dir / (self.model_path.stem + output_suffix)

        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Apply int8 quantization with representative dataset
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.create_representative_dataset(
            dataset_dir
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        print("Converting to int8...")
        quantized_model = converter.convert()

        # Save the quantized model
        with open(output_path, "wb") as f:
            f.write(quantized_model)

        print(f"Int8 model saved: {output_path}")

        # Get model size info
        original_size = self.model_path.stat().st_size
        quantized_size = output_path.stat().st_size
        compression_ratio = (1 - quantized_size / original_size) * 100

        print(
            f"Size reduction: {compression_ratio:.1f}% ({original_size} -> {quantized_size} bytes)"
        )

        return str(output_path)

    def quantize_dynamic_range(self, output_suffix: str = "_dynamic.tflite") -> str:
        """
        Apply dynamic range quantization (no representative dataset needed).

        Args:
            output_suffix: Suffix for output file

        Returns:
            Path to quantized model
        """
        print("\\n" + "=" * 50)
        print("DYNAMIC RANGE QUANTIZATION")
        print("=" * 50)

        output_path = self.output_dir / (self.model_path.stem + output_suffix)

        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Apply dynamic range quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        print("Converting with dynamic range quantization...")
        quantized_model = converter.convert()

        # Save the quantized model
        with open(output_path, "wb") as f:
            f.write(quantized_model)

        print(f"Dynamic model saved: {output_path}")

        # Get model size info
        original_size = self.model_path.stat().st_size
        quantized_size = output_path.stat().st_size
        compression_ratio = (1 - quantized_size / original_size) * 100

        print(
            f"Size reduction: {compression_ratio:.1f}% ({original_size} -> {quantized_size} bytes)"
        )

        return str(output_path)


def quantize_model(
    model_path: str,
    quantization_type: str = "float16",
    dataset_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    Main function to quantize a face recognition model.

    Args:
        model_path: Path to trained Keras model
        quantization_type: Type of quantization ('float16', 'int8', 'dynamic')
        dataset_dir: Directory with representative data (required for int8)
        output_dir: Output directory for quantized models

    Returns:
        Path to quantized model file
    """
    print(f"Starting model quantization...")
    print(f"Model: {model_path}")
    print(f"Quantization type: {quantization_type}")

    quantizer = ModelQuantizer(model_path, output_dir)

    if quantization_type == "float16":
        return quantizer.quantize_float16()

    elif quantization_type == "int8":
        if not dataset_dir:
            print("Error: int8 quantization requires representative dataset directory")
            raise ValueError("Dataset directory required for int8 quantization")
        return quantizer.quantize_int8(dataset_dir)

    elif quantization_type == "dynamic":
        return quantizer.quantize_dynamic_range()

    else:
        print(f"Error: Unknown quantization type '{quantization_type}'")
        raise ValueError(f"Supported types: float16, int8, dynamic")


def main():
    """Example usage of model quantization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quantize face recognition models for TFLite"
    )
    parser.add_argument("--model", required=True, help="Path to trained Keras model")
    parser.add_argument(
        "--type",
        choices=["float16", "int8", "dynamic"],
        default="float16",
        help="Quantization type",
    )
    parser.add_argument("--dataset", help="Dataset directory for int8 quantization")
    parser.add_argument("--output", help="Output directory for quantized models")

    args = parser.parse_args()

    try:
        quantized_path = quantize_model(
            args.model, args.type, args.dataset, args.output
        )
        print(f"\\nQuantization completed successfully!")
        print(f"Output: {quantized_path}")

    except Exception as e:
        print(f"\\nQuantization failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
