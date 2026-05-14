"""Embedding-quality evaluation for the three FaceNet fine-tuning approaches.

Extracts 512D FaceNet-backbone embeddings from the held-out test split and
computes geometry metrics that are independent of the classification head:

- Average intra-class L2 distance
- Average inter-class centroid L2 distance
- Silhouette score (cosine)
- Separation ratio (inter / intra)

All three approaches are compared at the SAME architectural point (the FaceNet
backbone output, 512D) so the comparison isolates the effect of fine-tuning on
the backbone representation. Because Transfer Learning leaves the backbone
frozen, its embeddings are equal to vanilla pre-trained FaceNet — i.e. its row
in the result table doubles as the baseline.

Output:
    results/embedding_quality.json
    results/embedding_quality_report.md
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import silhouette_score

from bp_face_recognition.config.settings import settings
from bp_face_recognition.utils.facenet_loader import load_finetuned_facenet_robust
from bp_face_recognition.vision.training.finetune.dataset_loader import (
    create_combined_dataset,
)


PROJECT_ROOT = settings.ROOT_DIR
FINETUNED_DIR = PROJECT_ROOT / "src" / "bp_face_recognition" / "models" / "finetuned"
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MODELS = [
    {
        "key": "TL",
        "name": "Transfer Learning",
        "path": FINETUNED_DIR / "facenet_transfer_v1.0.keras",
        "kind": "classifier",
    },
    {
        "key": "PU",
        "name": "Progressive Unfreezing",
        "path": FINETUNED_DIR / "facenet_progressive_v1.0.keras",
        "kind": "classifier",
    },
    {
        "key": "TLoss",
        "name": "Triplet Loss",
        # After the retrain fix, the embedding model is saved weights-only
        # because FaceNet's Lambda layers are not JSON-serializable.
        "path_weights": FINETUNED_DIR / "facenet_triplet_v1.0.weights.h5",
        "path_best_weights": FINETUNED_DIR / "facenet_triplet_best.weights.h5",
        "kind": "triplet",
    },
]


def build_classifier_embedder(model_path: Path) -> keras.Model:
    """Load a TL/PU classifier and return its inner FaceNet base directly.

    In Keras 3, you cannot rewire a parent model's input through a nested
    sub-model — the sub-model owns its own input tensor. The nested base IS
    itself a keras.Model with input (160,160,3) and output 512D, so we can use
    it directly without wrapping.
    """
    classifier = load_finetuned_facenet_robust(str(model_path))
    # _build_facenet_classifier stacks: Input → base_model → Dense(256) → Dropout → Dense(N)
    # layers[0] is the InputLayer, layers[1] is the nested FaceNet base model
    return classifier.layers[1]


def build_triplet_embedder(path_weights: Path, path_best_weights: Path) -> keras.Model:
    """Rebuild the FaceNet backbone and load the triplet-trained weights into it.

    The trainer saves the embedding model with `save_weights()` (not `save()`)
    because FaceNet contains Lambda layers whose function references are not
    JSON-serializable in the .keras zip format. To use the saved checkpoint we
    rebuild the architecture via keras_facenet and load weights by name.
    """
    from keras_facenet import FaceNet

    # Prefer the post-training v1.0 weights; fall back to the per-epoch best.
    chosen = path_weights if path_weights.exists() else path_best_weights
    if not chosen.exists():
        raise FileNotFoundError(
            f"No TLoss weights found. Expected {path_weights} or {path_best_weights}."
        )

    facenet = FaceNet()
    base_model = facenet.model
    # `path_weights` is the embedding-only weight file from save_weights() —
    # it has the SAME architecture as a fresh FaceNet base, so load_weights
    # by topological order. Keras 3's .weights.h5 format does not support
    # by_name=True (that flag is legacy-.h5 only).
    base_model.load_weights(str(chosen))
    return base_model


def collect_test_set() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Materialize the 1,062-sample test split as numpy arrays."""
    _, _, test_ds, info = create_combined_dataset(batch_size=32, augmentation=False)
    images_list, labels_list = [], []
    for imgs, lbls in test_ds:
        images_list.append(imgs.numpy())
        labels_list.append(np.argmax(lbls.numpy(), axis=1))
    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return images, labels, info["class_names"]


def extract_embeddings(embedder: keras.Model, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Run the embedder on all images, return (N, 512)."""
    return embedder.predict(images, batch_size=batch_size, verbose=0)


def intra_class_distance(embs: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[int, float]]:
    """Mean pairwise L2 distance within each class, averaged over classes.

    Per-class value uses centroid-based mean distance (E[||x - mu||]) — equivalent
    in trend to mean pairwise distance but O(N) instead of O(N^2) per class.
    """
    per_class: Dict[int, float] = {}
    for cls in np.unique(labels):
        members = embs[labels == cls]
        if len(members) < 2:
            continue
        centroid = members.mean(axis=0)
        dists = np.linalg.norm(members - centroid, axis=1)
        per_class[int(cls)] = float(dists.mean())
    overall = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return overall, per_class


def inter_class_distance(embs: np.ndarray, labels: np.ndarray) -> float:
    """Mean L2 distance between class centroids."""
    classes = np.unique(labels)
    centroids = np.stack([embs[labels == c].mean(axis=0) for c in classes])
    dists = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    return float(np.mean(dists)) if dists else 0.0


def silhouette_cosine(embs: np.ndarray, labels: np.ndarray, max_samples: int = 2000) -> float:
    """Silhouette score with cosine distance; subsample for tractability."""
    if len(embs) > max_samples:
        idx = np.random.default_rng(seed=0).choice(len(embs), size=max_samples, replace=False)
        embs, labels = embs[idx], labels[idx]
    return float(silhouette_score(embs, labels, metric="cosine"))


def evaluate_model(spec: Dict, images: np.ndarray, labels: np.ndarray) -> Dict:
    print(f"\n  [{spec['key']}] {spec['name']}")
    if spec["kind"] == "classifier":
        embedder = build_classifier_embedder(spec["path"])
    else:
        embedder = build_triplet_embedder(spec["path_weights"], spec["path_best_weights"])

    print(f"    Embedder output shape: {embedder.output_shape}")
    print(f"    Extracting embeddings for {len(images)} samples...")
    embs = extract_embeddings(embedder, images)
    print(f"    Embedding matrix: {embs.shape}")

    intra, intra_per_class = intra_class_distance(embs, labels)
    inter = inter_class_distance(embs, labels)
    sil = silhouette_cosine(embs, labels)
    ratio = inter / intra if intra > 0 else float("inf")

    print(
        f"    intra={intra:.3f}  inter={inter:.3f}  "
        f"silhouette(cos)={sil:.4f}  separation={ratio:.3f}"
    )

    return {
        "key": spec["key"],
        "name": spec["name"],
        "embedding_dim": int(embs.shape[1]),
        "num_samples": int(embs.shape[0]),
        "intra_class_distance": round(intra, 4),
        "inter_class_distance": round(inter, 4),
        "silhouette_cosine": round(sil, 4),
        "separation_ratio": round(ratio, 4),
        "intra_per_class": {str(k): round(v, 4) for k, v in intra_per_class.items()},
    }


def render_markdown(rows: List[Dict], class_names: List[str]) -> str:
    out = ["# Embedding Quality Comparison\n"]
    out.append(
        "All three models evaluated at the FaceNet backbone output (512D), "
        "on the 1,062-sample held-out test set. Distances are L2; silhouette "
        "uses cosine. Transfer Learning's backbone is frozen during training, "
        "so its row also serves as the vanilla pre-trained FaceNet baseline.\n"
    )

    out.append("| Metric | " + " | ".join(r["name"] for r in rows) + " |")
    out.append("|---" * (len(rows) + 1) + "|")

    def line(label: str, key: str, fmt: str = "{:.4f}") -> str:
        return (
            f"| {label} | "
            + " | ".join(fmt.format(r[key]) for r in rows)
            + " |"
        )

    out.append(line("Avg Intra-class Distance (L2, lower is better)", "intra_class_distance", "{:.3f}"))
    out.append(line("Avg Inter-class Distance (L2, higher is better)", "inter_class_distance", "{:.3f}"))
    out.append(line("Silhouette Score (cosine, higher is better)", "silhouette_cosine"))
    out.append(line("Separation Ratio (inter / intra)", "separation_ratio", "{:.3f}"))

    return "\n".join(out) + "\n"


def main():
    print("=" * 60)
    print("EMBEDDING QUALITY EVALUATION (FaceNet backbone, 512D)")
    print("=" * 60)

    print("Loading test split...")
    images, labels, class_names = collect_test_set()
    print(f"  Test samples: {len(images)}, classes: {len(class_names)}")

    rows = []
    for spec in MODELS:
        try:
            rows.append(evaluate_model(spec, images, labels))
        except Exception as exc:
            print(f"  [error] {spec['key']} failed: {exc}")
            rows.append({"key": spec["key"], "name": spec["name"], "error": str(exc)})

    out_payload = {
        "embedding_extraction_point": "FaceNet backbone output (512D)",
        "num_test_samples": int(len(images)),
        "class_names": class_names,
        "models": rows,
    }
    json_path = OUTPUT_DIR / "embedding_quality.json"
    with open(json_path, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nWrote {json_path}")

    md_path = OUTPUT_DIR / "embedding_quality_report.md"
    valid_rows = [r for r in rows if "error" not in r]
    with open(md_path, "w") as f:
        f.write(render_markdown(valid_rows, class_names))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
