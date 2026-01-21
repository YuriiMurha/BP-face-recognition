import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from bp_face_recognition.config.settings import settings
from bp_face_recognition.models.model import FaceTracker


def calculate_top_k_accuracy(y_true, scores, k_list=[1, 3, 5]):
    """
    Calculates Top-K accuracy.
    scores: array of shape (n_samples, n_classes)
    y_true: array of shape (n_samples,) with class indices
    """
    results = {}
    for k in k_list:
        if k > scores.shape[1]:
            continue
        # Get top k indices for each row
        top_k_indices = np.argsort(scores, axis=1)[:, -k:]
        # Check if true label is in top k
        correct = np.any(top_k_indices == np.array(y_true)[:, None], axis=1)
        results[f"Top-{k}"] = np.mean(correct)
    return results


def evaluate_recognition(dataset_name):
    print(f"Evaluating recognition for dataset: {dataset_name}")
    tracker = FaceTracker()

    test_path = settings.CROPPED_DIR / dataset_name / "test"
    train_path = settings.CROPPED_DIR / dataset_name / "train"

    if not test_path.exists() or not train_path.exists():
        print(f"Error: Paths not found for {dataset_name}")
        return

    # Build gallery
    print("Building gallery from training set...")
    gallery_embeddings = []
    gallery_labels = []

    train_images = list(train_path.glob("*.jpg"))

    # Batch processing for speed
    BATCH_SIZE = 32
    for i in range(0, len(train_images), BATCH_SIZE):
        batch_paths = train_images[i : i + BATCH_SIZE]
        batch_imgs = []
        batch_labels = []
        for img_path in batch_paths:
            try:
                label = int(img_path.name.split(".")[-2])
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.resize(img, (224, 224))
                batch_imgs.append(img)
                batch_labels.append(label)
            except Exception:
                continue

        if not batch_imgs:
            continue

        # Convert to tensor and predict
        batch_tensor = tf.image.convert_image_dtype(np.array(batch_imgs), tf.float32)

        recognizer = tracker.recognizer
        if hasattr(recognizer, "model") and recognizer.model is not None:
            embeddings = recognizer.model.predict(batch_tensor, verbose=0)
        else:
            embeddings = [tracker.get_embedding(img) for img in batch_imgs]

        gallery_embeddings.extend(embeddings)
        gallery_labels.extend(batch_labels)

        if (i // BATCH_SIZE) % 10 == 0:
            print(
                f"Gallery: {min(i + BATCH_SIZE, len(train_images))}/{len(train_images)}"
            )

    gallery_embeddings = np.array(gallery_embeddings)
    unique_labels = sorted(list(set(gallery_labels)))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}

    # Average embeddings per label in gallery
    averaged_gallery = []
    for label in unique_labels:
        mask = np.array(gallery_labels) == label
        averaged_gallery.append(np.mean(gallery_embeddings[mask], axis=0))
    averaged_gallery = np.array(averaged_gallery)

    # Evaluate test set
    print("Evaluating test set...")
    y_true = []
    all_scores = []

    test_images = list(test_path.glob("*.jpg"))
    for i in range(0, len(test_images), BATCH_SIZE):
        batch_paths = test_images[i : i + BATCH_SIZE]
        batch_imgs = []
        batch_true_labels = []
        for img_path in batch_paths:
            try:
                true_label = int(img_path.name.split(".")[-2])
                if true_label not in label_to_idx:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.resize(img, (224, 224))
                batch_imgs.append(img)
                batch_true_labels.append(label_to_idx[true_label])
            except Exception:
                continue

        if not batch_imgs:
            continue

        batch_tensor = tf.image.convert_image_dtype(np.array(batch_imgs), tf.float32)

        if hasattr(recognizer, "model") and recognizer.model is not None:
            embeddings = recognizer.model.predict(batch_tensor, verbose=0)
        else:
            embeddings = [tracker.get_embedding(img) for img in batch_imgs]

        for emb, true_idx in zip(embeddings, batch_true_labels):
            # Compute cosine similarities
            dots = np.dot(averaged_gallery, emb)
            norms = np.linalg.norm(averaged_gallery, axis=1) * np.linalg.norm(emb)
            scores = dots / (norms + 1e-8)  # Avoid division by zero

            y_true.append(true_idx)
            all_scores.append(scores)

        if (i // BATCH_SIZE) % 10 == 0:
            print(
                f"Testing: {min(i + BATCH_SIZE, len(test_images))}/{len(test_images)}"
            )

    y_true = np.array(y_true)
    all_scores = np.array(all_scores)

    # Metrics
    y_pred = np.argmax(all_scores, axis=1)

    top_k = calculate_top_k_accuracy(y_true, all_scores)
    print("\nTop-K Accuracy:")
    for k, acc in top_k.items():
        print(f"{k}: {acc:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(unique_labels)),
        target_names=[str(l) for l in unique_labels],
        output_dict=True,
    )
    df_report = pd.DataFrame(report).transpose()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=unique_labels, yticklabels=unique_labels
    )
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Save assets
    output_dir = settings.ASSETS_DIR / "reports" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    metrics_path = output_dir / "recognition_report.md"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"# Recognition Evaluation Report: {dataset_name}\n\n")
        f.write("## Top-K Accuracy\n")
        for k, acc in top_k.items():
            f.write(f"- **{k}**: {acc:.4f}\n")
        f.write("\n## Classification Report\n")
        markdown_report = df_report.to_markdown()
        f.write(markdown_report if markdown_report else "")
        f.write(f"\n\n![Confusion Matrix](confusion_matrix.png)\n")

    print(f"\nReport saved to: {metrics_path}")


if __name__ == "__main__":
    import cv2  # Ensure cv2 is available

    if len(sys.argv) > 1:
        evaluate_recognition(sys.argv[1])
    else:
        evaluate_recognition("seccam_2")
