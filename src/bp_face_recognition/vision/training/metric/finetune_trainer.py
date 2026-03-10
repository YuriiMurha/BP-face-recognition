"""
Experiment C: Fine-tune EfficientNetB0 as classifier, then use embeddings.

Strategy:
  1. Train EfficientNetB0 with softmax on all identities (classification loss)
  2. Phase 1: Frozen backbone, train embedding + classifier head
  3. Phase 2: Unfreeze top layers, fine-tune with low LR
  4. Save backbone+embedding weights (strip softmax head)

This avoids triplet loss collapse by using stable cross-entropy training.
The model is then used identically to the metric model - embedding space
is discriminative because the backbone learned to separate identities.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable()
class L2NormalizeLayer(layers.Layer):
    """Custom L2 normalization layer for better serialization compatibility."""

    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


def discover_identities(dataset_paths):
    """
    Scan dataset directories and build identity → file list map.
    Filename format: {Identity_Name}_{uuid}.{aug_idx}.jpg
    """
    identity_to_files = {}

    for ds_path in dataset_paths:
        p = Path(ds_path)
        if not p.exists():
            print(f"[WARN] Dataset path not found: {ds_path}")
            continue

        # Collect images from root and train/ subdirectory
        dirs_to_scan = [p]
        if (p / "train").exists():
            dirs_to_scan = [p / "train"]

        for scan_dir in dirs_to_scan:
            for img_file in sorted(scan_dir.glob("*.jpg")):
                # Parse identity from filename
                # LFW format: Alejandro_Toledo_0000.0.jpg -> Alejandro_Toledo
                # Custom format: Yurii_uuid.0.jpg -> Yurii
                filename = img_file.name  # e.g., "Alejandro_Toledo_0000.0.jpg"
                # Remove extension: "Alejandro_Toledo_0000.0"
                base = filename.rsplit(".", 1)[0]
                # Remove augmentation suffix: "Alejandro_Toledo_0000.0" -> "Alejandro_Toledo_0000"
                # or "Yurii_uuid.0" -> "Yurii_uuid"
                if "." in base:
                    base = base.rsplit(".", 1)[0]  # "Alejandro_Toledo_0000"

                # Split by underscore
                parts = base.split("_")

                # Check if last part is a numeric index (LFW format) or UUID (custom format)
                last_part = parts[-1]
                is_uuid = len(last_part) >= 8 and any(
                    c in last_part for c in ["-", "a", "b", "c", "d", "e", "f"]
                )
                is_numeric = last_part.isdigit()

                if is_numeric:
                    # LFW: Alejandro_Toledo_0000 -> Alejandro_Toledo
                    identity = "_".join(parts[:-1])
                elif is_uuid:
                    # Custom: Yurii_uuid -> Yurii
                    identity = "_".join(parts[:-1])
                else:
                    # Single word identity or compound name
                    identity = base

                if identity not in identity_to_files:
                    identity_to_files[identity] = []
                identity_to_files[identity].append(str(img_file))

    # Filter: need at least 2 images per identity for val split
    identity_to_files = {k: v for k, v in identity_to_files.items() if len(v) >= 2}
    return identity_to_files


def build_tf_dataset(
    identity_to_files,
    identity_to_idx,
    img_size=(224, 224),
    val_split=0.15,
    batch_size=32,
    subset="train",
):
    """Build tf.data.Dataset from file lists with proper train/val split."""
    all_paths, all_labels = [], []

    for identity, files in identity_to_files.items():
        label = identity_to_idx[identity]
        for f in files:
            all_paths.append(f)
            all_labels.append(label)

    # Deterministic shuffle before split
    rng = np.random.RandomState(42)
    idxs = rng.permutation(len(all_paths))
    all_paths = np.array(all_paths)[idxs]
    all_labels = np.array(all_labels)[idxs]

    split = int(len(all_paths) * (1 - val_split))
    if subset == "train":
        paths, labels = all_paths[:split], all_labels[:split]
    else:
        paths, labels = all_paths[split:], all_labels[split:]

    print(f"[DATA] {subset}: {len(paths)} images, {len(identity_to_files)} identities")

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if subset == "train":
        ds = ds.shuffle(buffer_size=min(10000, len(paths)), seed=42)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(num_classes, embedding_dim=128, backbone="EfficientNetB0"):
    """
    Build classifier model:
      EfficientNetB0 → GAP → Dense(embedding_dim) → L2Norm → Dense(num_classes, softmax)

    The embedding layer output is what we save for inference.
    """
    inputs = tf.keras.Input(shape=(224, 224, 3))

    if backbone == "EfficientNetB0":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # Freeze backbone initially
    base.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(embedding_dim, name="embedding_dense")(x)
    embedding = tf.keras.layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1), name="l2_norm"
    )(x)
    output = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="classifier"
    )(embedding)

    full_model = tf.keras.Model(inputs=inputs, outputs=output)
    embedding_model = tf.keras.Model(inputs=inputs, outputs=embedding)

    return full_model, embedding_model, base


def finetune_train(
    dataset_paths,
    backbone="EfficientNetB0",
    embedding_dim=128,
    epochs_phase1=15,
    epochs_phase2=10,
    batch_size=32,
):
    print(f"\n[TRAIN] === Experiment C: Fine-tune Classifier ===")
    print(f"[TRAIN] Datasets: {dataset_paths}")
    print(f"[TRAIN] Backbone: {backbone}, Embedding: {embedding_dim}D")
    print(f"[TRAIN] Phase1 epochs: {epochs_phase1}, Phase2 epochs: {epochs_phase2}")

    # 1. Discover identities
    identity_to_files = discover_identities(dataset_paths)
    num_classes = len(identity_to_files)
    identity_to_idx = {
        name: i for i, name in enumerate(sorted(identity_to_files.keys()))
    }
    print(
        f"[DATA] Found {num_classes} identities, {sum(len(v) for v in identity_to_files.values())} images"
    )

    if num_classes < 2:
        raise ValueError("Need at least 2 identities to train classifier")

    # 2. Build datasets
    train_ds = build_tf_dataset(
        identity_to_files, identity_to_idx, batch_size=batch_size, subset="train"
    )
    val_ds = build_tf_dataset(
        identity_to_files, identity_to_idx, batch_size=batch_size, subset="val"
    )

    # Special handling: create model in a way that's compatible across platforms
    # The issue is WSL's Keras version uses different layer serialization
    # We'll use Sequential with explicit layers
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # Create base model (same as before)
    if backbone == "EfficientNetB0":
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(224, 224, 3), include_top=False, weights="imagenet"
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    base_model.trainable = False

    # Create the embedding model with explicit layers
    embedding_model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(embedding_dim, name="embedding_dense"),
            # Use custom L2NormalizeLayer for better serialization
            L2NormalizeLayer(axis=1, name="l2_norm"),
        ],
        name="embedding_model",
    )

    # Create classification model for training
    classifier_model = models.Sequential(
        [
            embedding_model,
            layers.Dense(num_classes, activation="softmax", name="classifier"),
        ],
        name="classifier_model",
    )

    print(f"[MODEL] Parameters: {classifier_model.count_params():,}")

    model_dir = Path("src/bp_face_recognition/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save both .keras (full model) and .weights.h5
    model_path = model_dir / f"metric_{backbone.lower()}_{embedding_dim}d_final.keras"
    weights_path = model_dir / f"metric_{backbone.lower()}_{embedding_dim}d.weights.h5"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(weights_path),
            save_weights_only=True,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1
        ),
    ]

    # --- Phase 1: Train embedding head only ---
    print("\n[PHASE 1] Training embedding head (backbone frozen)...")
    classifier_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history1 = classifier_model.fit(
        train_ds, validation_data=val_ds, epochs=epochs_phase1, callbacks=callbacks
    )

    # --- Phase 2: Unfreeze top layers + fine-tune ---
    print("\n[PHASE 2] Fine-tuning top layers of backbone...")
    # Unfreeze top 30 layers of the base model (which is part of embedding_model)
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Recompile classifier model
    classifier_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = classifier_model.fit(
        train_ds, validation_data=val_ds, epochs=epochs_phase2, callbacks=callbacks
    )

    # 4. Save embedding model (no softmax head) as .keras file
    print("\n[SAVE] Saving embedding model...")
    embedding_model.save(str(model_path))
    print(f"[SAVE] Model saved to: {model_path}")

    # Also save weights separately (for compatibility)
    embedding_model.save_weights(str(weights_path))
    print(f"[SAVE] Weights saved to: {weights_path}")

    # Quick sanity check: are embeddings diverse?
    print("\n[CHECK] Embedding diversity sanity check...")
    sample_batch = next(iter(val_ds))
    sample_imgs = sample_batch[0][:4]
    embs = embedding_model(sample_imgs, training=False).numpy()
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            sim = float(np.dot(embs[i], embs[j]))
            print(f"  Similarity(img{i}, img{j}): {sim:.4f}")

    # Check model performance
    final_val_acc = max(history2.history["val_accuracy"])
    if final_val_acc < 0.3:
        print(f"\n[WARNING] Final validation accuracy: {final_val_acc:.4f} (< 0.3)")
        print("The model may not have learned well. Consider:")
        print("  - Training more epochs")
        print("  - Adding more training data")
        print("  - Adjusting learning rate")

    # Save training history
    import json

    history_path = Path("data/logs/finetune_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(
            {
                "val_accuracy": [
                    float(x)
                    for x in history1.history["val_accuracy"]
                    + history2.history["val_accuracy"]
                ],
                "val_loss": [
                    float(x)
                    for x in history1.history["val_loss"] + history2.history["val_loss"]
                ],
            },
            f,
        )
    print(f"[SAVE] Training history saved to: {history_path}")

    print("\n[DONE] Fine-tune training complete!")
    return str(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment C: Fine-tune Classifier for Embeddings"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="lfw,webcam,seccam",
        help="Comma-separated dataset names",
    )
    parser.add_argument("--backbone", type=str, default="EfficientNetB0")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--epochs-phase1", type=int, default=15)
    parser.add_argument("--epochs-phase2", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    dataset_names = [d.strip() for d in args.datasets.split(",")]
    dataset_paths = []
    for name in dataset_names:
        p = Path(f"data/datasets/augmented/{name}")
        if p.exists():
            dataset_paths.append(str(p))
        else:
            print(f"[WARN] Dataset not found: {name}")

    if not dataset_paths:
        raise SystemExit("No valid datasets found.")

    model_path = finetune_train(
        dataset_paths=dataset_paths,
        backbone=args.backbone,
        embedding_dim=args.dim,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        batch_size=args.batch_size,
    )
    print(f"\nTraining complete! Model saved to: {model_path}")
