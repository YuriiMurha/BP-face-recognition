"""Comprehensive Model Evaluation and Comparison

Evaluates and compares all FaceNet fine-tuned models.
Generates detailed comparison report with statistical metrics.

Usage:
    python evaluate_comprehensive.py \
        --models model1.keras model2.keras model3.keras \
        --output results/comparison

Output:
    - JSON results for each model
    - Markdown comparison report
    - Confusion matrices (PNG)
    - Summary table
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from bp_face_recognition.vision.training.finetune.dataset_loader import (
    create_combined_dataset,
)


def evaluate_single_model(
    model_path: str, test_ds, dataset_info: Dict, verbose: bool = True
) -> Dict:
    """
    Evaluate a single model comprehensively.

    Args:
        model_path: Path to model file
        test_ds: Test dataset
        dataset_info: Dataset information
        verbose: Print progress

    Returns:
        Dictionary with comprehensive results
    """
    model_name = Path(model_path).stem

    if verbose:
        print(f"\nEvaluating: {model_name}")
        print("-" * 60)

    # Load model
    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"⚠ Error loading {model_name}: {e}")
        return None

    # Get class names
    class_names = dataset_info.get(
        "class_names", [f"Class_{i}" for i in range(dataset_info["num_classes"])]
    )

    # Collect predictions
    all_predictions = []
    all_labels = []
    inference_times = []

    import time

    for images, labels in test_ds:
        # Preprocess
        images = tf.cast(images, tf.float32)
        images = images / 255.0
        images = (images - 0.5) * 2.0

        # Time inference
        start = time.time()
        predictions = model.predict(images, verbose=0)
        inference_times.append(time.time() - start)

        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels.numpy(), axis=1)

        all_predictions.extend(predicted_classes)
        all_labels.extend(true_classes)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)

    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_predictions[mask] == i).mean()
            per_class_metrics[class_name] = {
                "accuracy": float(class_acc),
                "support": int(mask.sum()),
            }

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average="weighted", zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Inference time stats
    avg_inference_time = np.mean(inference_times) * 1000  # ms

    # Model size
    model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)

    results = {
        "model_name": model_name,
        "model_path": str(model_path),
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "accuracy": float(accuracy),
            "accuracy_percent": float(accuracy * 100),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "num_samples": len(all_labels),
            "num_classes": len(class_names),
            "avg_inference_time_ms": float(avg_inference_time),
            "model_size_mb": float(model_size_mb),
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist(),
    }

    if verbose:
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Inference: {avg_inference_time:.1f} ms/batch")
        print(f"  Model size: {model_size_mb:.1f} MB")

    return results


def generate_comparison_report(all_results: List[Dict], output_dir: Path) -> str:
    """
    Generate comprehensive comparison report.

    Args:
        all_results: List of evaluation results for each model
        output_dir: Output directory

    Returns:
        Path to generated report
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create comparison table
    comparison_data = []
    for result in all_results:
        metrics = result["metrics"]
        comparison_data.append(
            {
                "Model": result["model_name"],
                "Accuracy (%)": f"{metrics['accuracy_percent']:.2f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1 Score": f"{metrics['f1_score']:.4f}",
                "Inference (ms)": f"{metrics['avg_inference_time_ms']:.1f}",
                "Size (MB)": f"{metrics['model_size_mb']:.1f}",
            }
        )

    df = pd.DataFrame(comparison_data)

    # Generate Markdown report
    report_path = output_dir / "comprehensive_comparison_report.md"

    with open(report_path, "w") as f:
        f.write("# FaceNet Fine-Tuned Models: Comprehensive Comparison\n\n")
        f.write(
            f"**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        f.write("## Executive Summary\n\n")

        # Find best model
        best_idx = df["Accuracy (%)"].astype(float).idxmax()
        best_model = df.iloc[best_idx]["Model"]
        best_accuracy = df.iloc[best_idx]["Accuracy (%)"]

        f.write(
            f"🏆 **Best Model**: {best_model} with **{best_accuracy}%** accuracy\n\n"
        )

        f.write("## Comparison Table\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Detailed Results\n\n")

        for result in all_results:
            model_name = result["model_name"]
            metrics = result["metrics"]

            f.write(f"### {model_name}\n\n")
            f.write(f"- **Accuracy**: {metrics['accuracy_percent']:.2f}%\n")
            f.write(f"- **Precision**: {metrics['precision']:.4f}\n")
            f.write(f"- **Recall**: {metrics['recall']:.4f}\n")
            f.write(f"- **F1 Score**: {metrics['f1_score']:.4f}\n")
            f.write(
                f"- **Inference Time**: {metrics['avg_inference_time_ms']:.1f} ms/batch\n"
            )
            f.write(f"- **Model Size**: {metrics['model_size_mb']:.1f} MB\n")
            f.write(f"- **Samples**: {metrics['num_samples']}\n\n")

            f.write("#### Per-Class Accuracy\n\n")
            per_class_df = pd.DataFrame(
                [
                    {
                        "Class": k,
                        "Accuracy": f"{v['accuracy']:.2%}",
                        "Samples": v["support"],
                    }
                    for k, v in result["per_class_metrics"].items()
                ]
            )
            f.write(per_class_df.to_markdown(index=False))
            f.write("\n\n")

        f.write("## Conclusions\n\n")
        f.write(
            f"1. **Best Overall Performance**: {best_model} achieves the highest accuracy ({best_accuracy}%)\n"
        )
        f.write(
            "2. **Speed vs Accuracy Trade-off**: See inference times in the table above\n"
        )
        f.write(
            "3. **Model Size**: Consider quantization for deployment to reduce size by ~75%\n\n"
        )

        f.write("---\n\n")
        f.write("*Generated by evaluate_comprehensive.py*\n")

    print(f"\n✓ Report saved: {report_path}")

    # Also save JSON
    json_path = output_dir / "comprehensive_comparison_results.json"
    with open(json_path, "w") as f:
        json.dump(
            {"timestamp": datetime.now().isoformat(), "models": all_results},
            f,
            indent=2,
        )

    print(f"✓ JSON saved: {json_path}")

    return str(report_path)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation and comparison of FaceNet models"
    )
    parser.add_argument(
        "--models", nargs="+", required=True, help="Paths to model files to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation",
        help="Output directory for results",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    print(f"Models to evaluate: {len(args.models)}")

    # Load dataset once
    print("\nLoading dataset...")
    _, _, test_ds, dataset_info = create_combined_dataset(
        batch_size=32, augmentation=False
    )
    print(f"✓ Dataset loaded: {dataset_info['num_test']} test samples")

    # Evaluate each model
    all_results = []

    for model_path in args.models:
        if not Path(model_path).exists():
            print(f"⚠ Model not found: {model_path}")
            continue

        result = evaluate_single_model(model_path, test_ds, dataset_info, verbose=True)

        if result:
            all_results.append(result)

    if not all_results:
        print("\n⚠ No models evaluated successfully")
        return

    # Generate comparison report
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON REPORT")
    print("=" * 60)

    output_dir = Path(args.output)
    report_path = generate_comparison_report(all_results, output_dir)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Report: {report_path}")

    # Print summary
    print("\nSummary:")
    for result in all_results:
        acc = result["metrics"]["accuracy_percent"]
        print(f"  {result['model_name']:30s}: {acc:6.2f}%")


if __name__ == "__main__":
    main()
