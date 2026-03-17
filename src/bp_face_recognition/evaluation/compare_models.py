"""
Model Comparison Evaluation Script

Evaluates both EfficientNetB0 (128D) and FaceNet (512D) models on the same dataset
for fair comparison. Generates JSON reports with accuracy, inference time, and
embedding quality metrics.

Usage:
    python compare_models.py --dataset webcam --output results/comparison/
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

from bp_face_recognition.config.settings import settings
from bp_face_recognition.vision.core.face_tracker import FaceTracker


def load_dataset(
    dataset_name: str,
) -> Tuple[List[Path], List[int], List[Path], List[int]]:
    """
    Load train and test datasets.

    Returns:
        train_paths, train_labels, test_paths, test_labels
    """
    dataset_path = settings.CROPPED_DIR / dataset_name
    train_path = dataset_path / "train"
    test_path = dataset_path / "test"

    if not train_path.exists() or not test_path.exists():
        raise ValueError(
            f"Dataset {dataset_name} not found or missing train/test split"
        )

    # Load train images
    train_images = []
    train_labels = []
    for img_path in train_path.glob("*.jpg"):
        try:
            # Extract label from filename (format: name.label.idx.jpg)
            label = int(img_path.name.split(".")[-2])
            train_images.append(img_path)
            train_labels.append(label)
        except Exception:
            continue

    # Load test images
    test_images = []
    test_labels = []
    for img_path in test_path.glob("*.jpg"):
        try:
            label = int(img_path.name.split(".")[-2])
            test_images.append(img_path)
            test_labels.append(label)
        except Exception:
            continue

    print(f"Dataset: {dataset_name}")
    print(f"  Train: {len(train_images)} images")
    print(f"  Test: {len(test_images)} images")
    print(f"  Classes: {len(set(train_labels))}")

    return train_images, train_labels, test_images, test_labels


def evaluate_model(
    model_name: str,
    recognizer_type: str,
    train_images: List[Path],
    train_labels: List[int],
    test_images: List[Path],
    test_labels: List[int],
    batch_size: int = 32,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate a face recognition model.

    Args:
        model_name: Name of the model (for reporting)
        recognizer_type: Type of recognizer ('efficientnet' or 'facenet')
        train_images: List of training image paths
        train_labels: List of training labels
        test_images: List of test image paths
        test_labels: List of test labels
        batch_size: Batch size for processing
        verbose: Print progress

    Returns:
        Dictionary with evaluation results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

    # Initialize tracker with specified recognizer
    # Note: We need to modify the config or use a different approach
    # For now, we'll use the model directly

    results = {
        "model_name": model_name,
        "recognizer_type": recognizer_type,
        "timestamp": datetime.now().isoformat(),
        "dataset_stats": {
            "train_samples": len(train_images),
            "test_samples": len(test_images),
            "num_classes": len(set(train_labels)),
        },
    }

    # Load model based on type
    if recognizer_type == "efficientnet":
        model_path = str(settings.MODELS_DIR / "metric_efficientnetb0_128d_final.keras")
        embedding_dim = 128
    elif recognizer_type == "facenet":
        model_path = str(
            settings.MODELS_DIR / "finetuned" / "facenet_finetuned_pu_final.keras"
        )
        embedding_dim = 512
    else:
        raise ValueError(f"Unknown recognizer type: {recognizer_type}")

    if verbose:
        print(f"Loading model from: {model_path}")

    # Load the model
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        if verbose:
            print(f"✓ Model loaded successfully")
            print(f"  Input shape: {model.input_shape}")
            print(f"  Output shape: {model.output_shape}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        results["error"] = str(e)
        return results

    # Generate embeddings for training set
    if verbose:
        print(f"\nGenerating embeddings for training set...")

    train_embeddings = []
    train_inference_times = []

    for i in range(0, len(train_images), batch_size):
        batch_paths = train_images[i : i + batch_size]
        batch_imgs = []

        for img_path in batch_paths:
            try:
                import cv2

                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.resize(img, (224, 224))
                batch_imgs.append(img)
            except Exception:
                continue

        if not batch_imgs:
            continue

        # Preprocess
        batch_tensor = tf.image.convert_image_dtype(np.array(batch_imgs), tf.float32)

        # Generate embeddings with timing
        start_time = time.time()
        embeddings = model.predict(batch_tensor, verbose=0)
        inference_time = time.time() - start_time

        train_embeddings.append(embeddings)
        train_inference_times.append(inference_time)

        if verbose and (i // batch_size) % 10 == 0:
            print(
                f"  Progress: {min(i + batch_size, len(train_images))}/{len(train_images)}"
            )

    train_embeddings = np.vstack(train_embeddings)
    avg_train_time = np.mean(train_inference_times) / batch_size

    if verbose:
        print(f"✓ Training embeddings: {train_embeddings.shape}")
        print(f"  Avg inference time: {avg_train_time*1000:.2f}ms per image")

    # Average embeddings per class for gallery
    unique_labels = sorted(list(set(train_labels)))
    gallery_embeddings = []

    for label in unique_labels:
        mask = np.array(train_labels) == label
        gallery_embeddings.append(np.mean(train_embeddings[mask], axis=0))

    gallery_embeddings = np.array(gallery_embeddings)

    # Generate embeddings for test set
    if verbose:
        print(f"\nGenerating embeddings for test set...")

    test_embeddings = []
    test_inference_times = []
    y_true = []

    for i in range(0, len(test_images), batch_size):
        batch_paths = test_images[i : i + batch_size]
        batch_imgs = []
        batch_labels = []

        for img_path in batch_paths:
            try:
                import cv2

                label = int(img_path.name.split(".")[-2])
                if label not in unique_labels:
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.resize(img, (224, 224))
                batch_imgs.append(img)
                batch_labels.append(unique_labels.index(label))
            except Exception:
                continue

        if not batch_imgs:
            continue

        batch_tensor = tf.image.convert_image_dtype(np.array(batch_imgs), tf.float32)

        start_time = time.time()
        embeddings = model.predict(batch_tensor, verbose=0)
        inference_time = time.time() - start_time

        test_embeddings.append(embeddings)
        y_true.extend(batch_labels)
        test_inference_times.append(inference_time)

        if verbose and (i // batch_size) % 10 == 0:
            print(
                f"  Progress: {min(i + batch_size, len(test_images))}/{len(test_images)}"
            )

    test_embeddings = np.vstack(test_embeddings)
    y_true = np.array(y_true)
    avg_test_time = np.mean(test_inference_times) / batch_size

    if verbose:
        print(f"✓ Test embeddings: {test_embeddings.shape}")
        print(f"  Avg inference time: {avg_test_time*1000:.2f}ms per image")

    # Compute cosine similarities
    all_scores = []
    for emb in test_embeddings:
        # Normalize
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        gallery_norm = gallery_embeddings / (
            np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Cosine similarity
        scores = np.dot(gallery_norm, emb_norm)
        all_scores.append(scores)

    all_scores = np.array(all_scores)
    y_pred = np.argmax(all_scores, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Top-K accuracy
    top_k_results = {}
    for k in [1, 3, 5]:
        if k <= all_scores.shape[1]:
            top_k_indices = np.argsort(all_scores, axis=1)[:, -k:]
            correct = np.any(top_k_indices == y_true[:, None], axis=1)
            top_k_results[f"top_{k}"] = float(np.mean(correct))

    # Embedding diversity (max similarity between different classes)
    unique_embeddings = []
    for label in unique_labels:
        mask = np.array(train_labels) == label
        unique_embeddings.append(np.mean(train_embeddings[mask], axis=0))

    unique_embeddings = np.array(unique_embeddings)
    similarities = np.dot(unique_embeddings, unique_embeddings.T)
    np.fill_diagonal(similarities, 0)  # Remove self-similarity
    max_similarity = float(np.max(similarities))
    min_similarity = float(np.min(similarities))
    avg_similarity = float(np.mean(similarities))

    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS - {model_name}")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nTop-K Accuracy:")
        for k, acc in top_k_results.items():
            print(f"  {k}: {acc:.4f} ({acc*100:.2f}%)")
        print(f"\nInference Time:")
        print(f"  Training avg: {avg_train_time*1000:.2f}ms")
        print(f"  Testing avg: {avg_test_time*1000:.2f}ms")
        print(f"\nEmbedding Diversity:")
        print(f"  Max similarity: {max_similarity:.4f}")
        print(f"  Min similarity: {min_similarity:.4f}")
        print(f"  Avg similarity: {avg_similarity:.4f}")
        print(f"{'='*60}")

    # Compile results
    results.update(
        {
            "metrics": {
                "accuracy": float(accuracy),
                "accuracy_percent": float(accuracy * 100),
                "top_k_accuracy": top_k_results,
                "inference_time_ms": {
                    "train_avg": float(avg_train_time * 1000),
                    "test_avg": float(avg_test_time * 1000),
                },
                "embedding_diversity": {
                    "max_similarity": max_similarity,
                    "min_similarity": min_similarity,
                    "avg_similarity": avg_similarity,
                },
                "embedding_dim": embedding_dim,
            }
        }
    )

    return results


def generate_comparison_report(
    efficientnet_results: Dict, facenet_results: Dict, output_path: Path
):
    """
    Generate a comparison report in Markdown format.
    """
    report = f"""# Model Comparison Report: EfficientNetB0 vs FaceNet

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

| Model | Type | Dimensions | Dataset |
|-------|------|------------|---------|
| EfficientNetB0 | Custom Metric Learning | 128D | {efficientnet_results['dataset_stats']['train_samples']} train / {efficientnet_results['dataset_stats']['test_samples']} test |
| FaceNet PU | Fine-tuned (Progressive Unfreezing) | 512D | {facenet_results['dataset_stats']['train_samples']} train / {facenet_results['dataset_stats']['test_samples']} test |

## Accuracy Comparison

| Model | Top-1 | Top-3 | Top-5 |
|-------|-------|-------|-------|
| **EfficientNetB0** | {efficientnet_results['metrics']['accuracy_percent']:.2f}% | {efficientnet_results['metrics']['top_k_accuracy'].get('top_3', 0)*100:.2f}% | {efficientnet_results['metrics']['top_k_accuracy'].get('top_5', 0)*100:.2f}% |
| **FaceNet PU** | {facenet_results['metrics']['accuracy_percent']:.2f}% | {facenet_results['metrics']['top_k_accuracy'].get('top_3', 0)*100:.2f}% | {facenet_results['metrics']['top_k_accuracy'].get('top_5', 0)*100:.2f}% |

## Performance Comparison

| Model | Inference Time (ms) | Embedding Dim |
|-------|---------------------|---------------|
| **EfficientNetB0** | {efficientnet_results['metrics']['inference_time_ms']['test_avg']:.2f}ms | {efficientnet_results['metrics']['embedding_dim']}D |
| **FaceNet PU** | {facenet_results['metrics']['inference_time_ms']['test_avg']:.2f}ms | {facenet_results['metrics']['embedding_dim']}D |

## Embedding Quality

| Model | Max Similarity | Min Similarity | Avg Similarity |
|-------|----------------|----------------|----------------|
| **EfficientNetB0** | {efficientnet_results['metrics']['embedding_diversity']['max_similarity']:.4f} | {efficientnet_results['metrics']['embedding_diversity']['min_similarity']:.4f} | {efficientnet_results['metrics']['embedding_diversity']['avg_similarity']:.4f} |
| **FaceNet PU** | {facenet_results['metrics']['embedding_diversity']['max_similarity']:.4f} | {facenet_results['metrics']['embedding_diversity']['min_similarity']:.4f} | {facenet_results['metrics']['embedding_diversity']['avg_similarity']:.4f} |

> **Note:** Lower max similarity indicates better embedding diversity and discrimination.

## Analysis

### Accuracy
- **Winner:** {'FaceNet PU' if facenet_results['metrics']['accuracy'] > efficientnet_results['metrics']['accuracy'] else 'EfficientNetB0'}
- **Difference:** {abs(facenet_results['metrics']['accuracy'] - efficientnet_results['metrics']['accuracy'])*100:.2f} percentage points

### Speed
- **Faster:** {'FaceNet PU' if facenet_results['metrics']['inference_time_ms']['test_avg'] < efficientnet_results['metrics']['inference_time_ms']['test_avg'] else 'EfficientNetB0'}
- **Speedup:** {max(efficientnet_results['metrics']['inference_time_ms']['test_avg'], facenet_results['metrics']['inference_time_ms']['test_avg']) / min(efficientnet_results['metrics']['inference_time_ms']['test_avg'], facenet_results['metrics']['inference_time_ms']['test_avg']):.2f}x

### Embedding Quality
- **Better Diversity:** {'FaceNet PU' if facenet_results['metrics']['embedding_diversity']['max_similarity'] < efficientnet_results['metrics']['embedding_diversity']['max_similarity'] else 'EfficientNetB0'}
- Lower maximum similarity between different classes indicates better discrimination.

## Conclusions

{'FaceNet PU (fine-tuned with Progressive Unfreezing) demonstrates superior performance across all metrics.' if facenet_results['metrics']['accuracy'] > efficientnet_results['metrics']['accuracy'] and facenet_results['metrics']['embedding_diversity']['max_similarity'] < efficientnet_results['metrics']['embedding_diversity']['max_similarity'] else 'Both models show different strengths. Further runtime testing is recommended.'}

## Next Steps

1. **Runtime Testing:** Test both models in live camera scenarios
2. **Robustness Testing:** Evaluate performance under different conditions (lighting, angles)
3. **User Experience:** Compare recognition speed and accuracy in real-world usage

---

*This report was automatically generated by the model comparison evaluation system.*
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\n✓ Comparison report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare EfficientNetB0 and FaceNet models"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="webcam",
        help="Dataset name (webcam, seccam_2, etc.)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison",
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for processing"
    )
    parser.add_argument(
        "--skip-efficientnet",
        action="store_true",
        help="Skip EfficientNetB0 evaluation",
    )
    parser.add_argument(
        "--skip-facenet", action="store_true", help="Skip FaceNet evaluation"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MODEL COMPARISON EVALUATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    # Load dataset
    try:
        train_images, train_labels, test_images, test_labels = load_dataset(
            args.dataset
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Evaluate EfficientNetB0
    efficientnet_results = None
    if not args.skip_efficientnet:
        try:
            efficientnet_results = evaluate_model(
                model_name="EfficientNetB0 (128D Metric Learning)",
                recognizer_type="efficientnet",
                train_images=train_images,
                train_labels=train_labels,
                test_images=test_images,
                test_labels=test_labels,
                batch_size=args.batch_size,
            )
        except Exception as e:
            print(f"\n✗ EfficientNetB0 evaluation failed: {e}")
            efficientnet_results = {"error": str(e)}

    # Evaluate FaceNet
    facenet_results = None
    if not args.skip_facenet:
        try:
            facenet_results = evaluate_model(
                model_name="FaceNet PU (512D Fine-tuned)",
                recognizer_type="facenet",
                train_images=train_images,
                train_labels=train_labels,
                test_images=test_images,
                test_labels=test_labels,
                batch_size=args.batch_size,
            )
        except Exception as e:
            print(f"\n✗ FaceNet evaluation failed: {e}")
            facenet_results = {"error": str(e)}

    # Save individual results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if efficientnet_results and "error" not in efficientnet_results:
        with open(output_dir / "efficientnet_results.json", "w") as f:
            json.dump(efficientnet_results, f, indent=2)
        print(f"\n✓ EfficientNetB0 results saved")

    if facenet_results and "error" not in facenet_results:
        with open(output_dir / "facenet_results.json", "w") as f:
            json.dump(facenet_results, f, indent=2)
        print(f"✓ FaceNet results saved")

    # Generate comparison report
    if efficientnet_results and facenet_results:
        if "error" not in efficientnet_results and "error" not in facenet_results:
            generate_comparison_report(
                efficientnet_results,
                facenet_results,
                output_dir / "comparison_report.md",
            )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
