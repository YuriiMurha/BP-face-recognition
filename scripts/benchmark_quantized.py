"""
Benchmark quantized models for BP Face Recognition.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bp_face_recognition.vision.benchmark.model_benchmark import benchmark_model_variant


def test_quantized_model():
    """Test quantized EfficientNetB0 model."""
    model_path = "src/bp_face_recognition/models/efficientnetb0_seccam_2_cpu_final_dynamic.tflite"
    model_name = "efficientnetb0_seccam_2_20epochs_quantized"
    architecture = "EfficientNetB0"
    training_epochs = 20

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Quantized model not ready...")
        return

    print("Starting quantized model benchmark...")
    metrics = benchmark_model_variant(
        model_path=model_path,
        model_name=model_name,
        architecture=architecture,
        training_epochs=training_epochs,
        quantized=True,
    )

    print(f"Quantized benchmark completed!")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Inference: {metrics.inference_time_ms:.2f}ms")
    print(f"Size: {metrics.model_size_mb:.1f}MB")

    return metrics


if __name__ == "__main__":
    test_quantized_model()
