import os
import argparse
import tensorflow as tf
from pathlib import Path
from bp_face_recognition.vision.training.metric.model import create_embedding_model
from bp_face_recognition.vision.training.metric.loss import TripletModel
from bp_face_recognition.vision.training.metric.data_loader import TripletDataLoader


def train_metric(
    dataset_paths,
    backbone="EfficientNetB0",
    embedding_dim=128,
    epochs=20,
    batch_size=8,
    subset="train",
):
    """
    Train metric learning model.

    Args:
        dataset_paths: Single dataset name (str) or comma-separated list of dataset names
                      Examples: "lfw", "lfw,webcam,seccam_2"
        backbone: Backbone architecture
        embedding_dim: Embedding dimension
        epochs: Number of epochs
        batch_size: Batch size
        subset: Which subset to use (train, val, test)
    """
    # Parse dataset paths - can be comma-separated list
    if isinstance(dataset_paths, str):
        dataset_names = [d.strip() for d in dataset_paths.split(",")]
    else:
        dataset_names = dataset_paths

    # Convert dataset names to full paths
    full_paths = []
    for name in dataset_names:
        # Try augmented/{name}/{subset} first
        path = Path(f"data/datasets/augmented/{name}")
        if not path.exists():
            # Try cropped/{name}/{subset}
            path = Path(f"data/datasets/cropped/{name}")
            if not path.exists():
                # Try as absolute path
                path = Path(name)

        if path.exists():
            full_paths.append(str(path))
        else:
            print(f"Warning: Dataset not found: {name}")

    if not full_paths:
        raise ValueError(f"No valid datasets found: {dataset_names}")

    print(f"[TRAIN] Starting Metric Learning training...")
    print(f"[TRAIN] Datasets: {dataset_names}")
    print(f"[TRAIN] Subset: {subset}")
    print(f"[TRAIN] Architecture: {backbone} -> {embedding_dim}D")

    # 1. Create Data Loader with multiple datasets
    loader = TripletDataLoader(full_paths, subset=subset)
    train_ds = loader.get_dataset(batch_size=batch_size)

    # 2. Create Model
    base_model = create_embedding_model(
        backbone_type=backbone, embedding_dim=embedding_dim
    )
    triplet_model = TripletModel(base_model, margin=0.2)

    # 3. Compile
    triplet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    # 3b. Build the model by passing dummy data
    dummy_input = tf.zeros((1, 224, 224, 3))
    triplet_model(dummy_input, training=False)

    # 4. Train
    steps_per_epoch = 10  # Reduced for quick verification

    # Create model directory if needed
    model_dir = Path("src/bp_face_recognition/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        model_dir / f"metric_{backbone.lower()}_{embedding_dim}d.weights.h5"
    )

    triplet_model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                save_weights_only=True,
                monitor="loss",
                save_best_only=True,
            )
        ],
    )

    # 5. Save final backbone (the embedding model, not the triplet wrapper)
    final_path = model_dir / f"metric_{backbone.lower()}_{embedding_dim}d_final.keras"
    base_model.save(str(final_path))
    print(f"[TRAIN] Training complete. Backbone saved to {final_path}")

    # Also save weights separately for compatibility
    weights_path = model_dir / f"metric_{backbone.lower()}_{embedding_dim}d.weights.h5"
    base_model.save_weights(str(weights_path))
    print(f"[TRAIN] Weights saved to {weights_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Metric Learning Model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="lfw",
        help="Dataset name(s), comma-separated: 'lfw', 'lfw,webcam,seccam_2'",
    )
    parser.add_argument("--backbone", type=str, default="EfficientNetB0")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--subset", type=str, default="train", help="Subset to use: train, val, or test"
    )

    args = parser.parse_args()

    train_metric(
        dataset_paths=args.dataset,
        backbone=args.backbone,
        embedding_dim=args.dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        subset=args.subset,
    )
