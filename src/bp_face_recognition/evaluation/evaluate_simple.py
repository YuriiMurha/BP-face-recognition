"""Simple Model Evaluation

Fast evaluation of fine-tuned FaceNet models.
Provides basic accuracy metrics for quick assessment.

Usage:
    python evaluate_simple.py --model path/to/model.keras

Output:
    - Accuracy percentage
    - Per-class accuracy
    - JSON results file
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report

from bp_face_recognition.vision.training.finetune.dataset_loader import (
    create_combined_dataset,
)


def load_facenet_model(model_path: str):
    """
    Load FaceNet model with proper handling of custom layers.
    """
    from bp_face_recognition.utils.facenet_loader import load_finetuned_facenet

    return load_finetuned_facenet(model_path)


def evaluate_model_simple(model_path: str, verbose: bool = True) -> Dict:
    """
    Simple evaluation of a fine-tuned model.

    Args:
        model_path: Path to .keras model file
        verbose: Print progress

    Returns:
        Dictionary with evaluation results
    """
    if verbose:
        print("=" * 60)
        print("SIMPLE MODEL EVALUATION")
        print("=" * 60)
        print(f"Model: {Path(model_path).name}")

    # Load model using FaceNet compatibility layer
    if verbose:
        print("\nLoading model...")

    model = load_facenet_model(model_path)

    if verbose:
        print(f"✓ Model loaded")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")

    # Load dataset
    if verbose:
        print("\nLoading dataset...")

    _, _, test_ds, dataset_info = create_combined_dataset(
        batch_size=32, augmentation=False
    )

    num_classes = dataset_info["num_classes"]
    class_names = dataset_info.get(
        "class_names", [f"Class_{i}" for i in range(num_classes)]
    )

    if verbose:
        print(f"✓ Dataset loaded")
        print(f"  Test samples: {dataset_info['num_test']}")
        print(f"  Classes: {num_classes}")

    # Collect embeddings (since we're using base FaceNet)
    if verbose:
        print("\nEvaluating (using embeddings + KNN)...")

    # Get training data for KNN
    train_ds, _, _, _ = create_combined_dataset(batch_size=32, augmentation=False)

    train_embeddings = []
    train_labels = []

    for images, labels in train_ds:
        images = tf.cast(images, tf.float32)
        images = images / 255.0
        images = (images - 0.5) * 2.0

        embeddings = model.predict(images, verbose=0)
        train_embeddings.append(embeddings)
        train_labels.extend(np.argmax(labels.numpy(), axis=1))

        if len(train_labels) >= 2000:  # Limit for speed
            break

    train_embeddings = np.vstack(train_embeddings)
    train_labels = np.array(train_labels)

    # Test data
    test_embeddings = []
    test_labels = []

    for images, labels in test_ds:
        images = tf.cast(images, tf.float32)
        images = images / 255.0
        images = (images - 0.5) * 2.0

        embeddings = model.predict(images, verbose=0)
        test_embeddings.append(embeddings)
        test_labels.extend(np.argmax(labels.numpy(), axis=1))

    test_embeddings = np.vstack(test_embeddings)
    test_labels = np.array(test_labels)

    # KNN classifier
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_embeddings, train_labels)

    predictions = knn.predict(test_embeddings)
    accuracy = accuracy_score(test_labels, predictions)

    # Per-class accuracy
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        mask = test_labels == i
        if mask.sum() > 0:
            class_acc = (predictions[mask] == i).mean()
            per_class_accuracy[class_name] = float(class_acc)

    # Classification report
    report = classification_report(
        test_labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Prepare results
    results = {
        "model_name": Path(model_path).stem,
        "model_path": str(model_path),
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "accuracy": float(accuracy),
            "accuracy_percent": float(accuracy * 100),
            "num_samples": len(test_labels),
            "num_classes": num_classes,
        },
        "per_class_accuracy": per_class_accuracy,
        "classification_report": report,
        "note": "Evaluated using FaceNet embeddings + KNN classifier",
    }

    # Print results
    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Samples evaluated: {len(test_labels)}")
        print("\nPer-Class Accuracy:")
        for class_name, class_acc in sorted(per_class_accuracy.items()):
            print(f"  {class_name:15s}: {class_acc:.4f} ({class_acc*100:5.1f}%)")
        print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Simple evaluation of fine-tuned FaceNet models"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to .keras model file"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file path (optional)"
    )
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Evaluate
    results = evaluate_model_simple(args.model, verbose=not args.quiet)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        if not args.quiet:
            print(f"\n✓ Results saved to: {output_path}")
    else:
        # Default output path
        model_name = Path(args.model).stem
        default_output = f"results/evaluation/{model_name}_simple_eval.json"
        Path(default_output).parent.mkdir(parents=True, exist_ok=True)

        with open(default_output, "w") as f:
            json.dump(results, f, indent=2)

        if not args.quiet:
            print(f"\n✓ Results saved to: {default_output}")

    return results


if __name__ == "__main__":
    main()
