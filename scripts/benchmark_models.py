"""
Model benchmarking utilities and test scripts for BP Face Recognition.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bp_face_recognition.vision.benchmark.model_benchmark import (
    ModelBenchmarkTracker,
    benchmark_model_variant,
)


def test_current_model():
    """Test the current training model."""
    model_path = (
        "src/bp_face_recognition/models/efficientnetb0_seccam_2_cpu_final.keras"
    )
    model_name = "efficientnetb0_seccam_2_20epochs_baseline"
    architecture = "EfficientNetB0"
    training_epochs = 20

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Waiting for training to complete...")
        return

    print("Starting baseline model benchmark...")
    metrics = benchmark_model_variant(
        model_path=model_path,
        model_name=model_name,
        architecture=architecture,
        training_epochs=training_epochs,
        quantized=False,
    )

    print(f"Baseline benchmark completed!")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Inference: {metrics.inference_time_ms:.2f}ms")
    print(f"Size: {metrics.model_size_mb:.1f}MB")

    return metrics


def generate_full_report():
    """Generate comprehensive benchmark report."""
    tracker = ModelBenchmarkTracker()
    report_path = tracker.generate_report()
    print(f"📄 Report generated: {report_path}")
    return report_path


def compare_all_models():
    """Compare all benchmarked models."""
    tracker = ModelBenchmarkTracker()
    comparison = tracker.compare_models()

    print("\n🏆 Model Performance Comparison:")
    print("=" * 50)

    if "error" in comparison:
        print(comparison["error"])
        return

    # Show top performers
    for metric in ["accuracy", "inference_time_ms", "model_size_mb"]:
        comp = comparison["performance_comparison"][metric]
        if metric == "accuracy":
            print(f"\n🥇 Best Accuracy: {comp['best'][0]} - {comp['best'][1]:.4f}")
        elif metric == "inference_time_ms":
            print(f"⚡ Fastest Inference: {comp['best'][0]} - {comp['best'][1]:.2f}ms")
        elif metric == "model_size_mb":
            print(f"💾 Smallest Model: {comp['best'][0]} - {comp['best'][1]:.1f}MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BP Face Recognition Model Benchmarking"
    )
    parser.add_argument(
        "--action",
        choices=["test", "report", "compare"],
        required=True,
        help="Action to perform",
    )

    args = parser.parse_args()

    if args.action == "test":
        test_current_model()
    elif args.action == "report":
        generate_full_report()
    elif args.action == "compare":
        compare_all_models()
