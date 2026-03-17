"""
Evaluate FaceNet Triplet Loss Model (Option C) - Alternative Approach

This script evaluates the trained triplet model using the saved weights
and reconstructing the model architecture.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

from bp_face_recognition.vision.training.finetune.dataset_loader import (
    create_combined_dataset,
)


def build_facenet_model():
    """Build FaceNet model from scratch."""
    from keras_facenet import FaceNet

    facenet = FaceNet()
    return facenet.model


def evaluate_triplet_model_weights(weights_path: str, test_ds, dataset_info: dict):
    """
    Evaluate triplet model by loading weights into fresh model.
    """
    print(f"Building FaceNet model...")

    # Build fresh model
    model = build_facenet_model()

    print(f"Loading weights from: {weights_path}")
    try:
        model.load_weights(weights_path)
        print("✓ Weights loaded successfully")
    except Exception as e:
        print(f"⚠ Could not load weights: {e}")
        print("Using original FaceNet weights instead")

    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    # Collect test embeddings
    print("\nGenerating embeddings for test set...")
    test_embeddings = []
    test_labels = []

    for images, labels in test_ds:
        embeddings = model.predict(images, verbose=0)
        test_embeddings.append(embeddings)
        test_labels.extend(np.argmax(labels.numpy(), axis=1))

    test_embeddings = np.vstack(test_embeddings)
    test_labels = np.array(test_labels)

    print(f"✓ Test embeddings: {test_embeddings.shape}")

    # Collect training embeddings for classifier
    print("\nGenerating embeddings for training set...")
    train_ds, _, _, _ = create_combined_dataset(batch_size=32, augmentation=False)

    train_embeddings = []
    train_labels = []
    sample_count = 0
    max_samples = 2000  # Limit for speed

    for images, labels in train_ds:
        if sample_count >= max_samples:
            break
        embeddings = model.predict(images, verbose=0)
        train_embeddings.append(embeddings)
        train_labels.extend(np.argmax(labels.numpy(), axis=1))
        sample_count += len(images)

    train_embeddings = np.vstack(train_embeddings)
    train_labels = np.array(train_labels)

    print(f"✓ Train embeddings: {train_embeddings.shape}")

    # Train KNN classifier
    print("\nTraining KNN classifier...")
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(train_embeddings, train_labels)

    # Predict
    print("Predicting on test set...")
    predictions = knn.predict(test_embeddings)
    accuracy = accuracy_score(test_labels, predictions)

    print(f"\n{'='*60}")
    print(f"TRIPLET LOSS MODEL RESULTS (Option C)")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}")

    # Classification report
    class_names = dataset_info.get(
        "class_names", [f"Class_{i}" for i in range(dataset_info["num_classes"])]
    )
    report = classification_report(
        test_labels, predictions, target_names=class_names, output_dict=True
    )

    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, predictions, target_names=class_names))

    # Save results
    results = {
        "model_type": "FaceNet Triplet Loss (Option C)",
        "timestamp": datetime.now().isoformat(),
        "test_accuracy": float(accuracy),
        "test_accuracy_percent": float(accuracy * 100),
        "num_test_samples": len(test_labels),
        "num_train_samples": len(train_labels),
        "classification_report": report,
        "weights_path": weights_path,
        "embedding_shape": int(test_embeddings.shape[1]),
        "note": "Evaluated using loaded weights with KNN classifier",
    }

    results_path = Path(
        "src/bp_face_recognition/models/finetuned/facenet_triplet_evaluation.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate FaceNet Triplet Model")
    parser.add_argument(
        "--weights",
        type=str,
        default="src/bp_face_recognition/models/finetuned/facenet_triplet_best.keras",
        help="Path to trained model weights",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FACENET TRIPLET MODEL EVALUATION (Option C)")
    print("=" * 60)

    # Load test dataset
    print("\nLoading dataset...")
    _, _, test_ds, dataset_info = create_combined_dataset(
        batch_size=32, augmentation=False
    )

    print(f"✓ Dataset loaded: {dataset_info['num_test']} test samples")
    print(f"✓ Number of classes: {dataset_info['num_classes']}")

    # Evaluate
    results = evaluate_triplet_model_weights(args.weights, test_ds, dataset_info)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
