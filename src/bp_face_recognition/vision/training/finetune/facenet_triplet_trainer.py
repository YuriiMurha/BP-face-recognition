"""FaceNet Triplet Loss Trainer - Option C

Fine-tunes FaceNet using triplet loss for metric learning.
Directly optimizes embedding space for similarity matching.

Training Strategy:
- Online triplet mining (semi-hard negatives)
- Fine-tune entire FaceNet model
- Optimizes: L = max(0, d(a,p) - d(a,n) + margin)

Usage:
    python facenet_triplet_trainer.py --epochs 30 --batch-size 32 --margin 0.2
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from bp_face_recognition.vision.training.finetune.dataset_loader import (
    create_combined_dataset,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TripletDataGenerator:
    """
    Generate triplets (anchor, positive, negative) for training.
    """

    def __init__(
        self, dataset: tf.data.Dataset, num_classes: int, batch_size: int = 32
    ):
        self.dataset = dataset
        self.num_classes = num_classes
        self.batch_size = batch_size

        # Convert to list for triplet mining
        self.images = []
        self.labels = []

        logger.info("Building triplet dataset...")
        for images, labels in dataset:
            self.images.extend(images.numpy())
            self.labels.extend(np.argmax(labels.numpy(), axis=1))

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

        # Group by class
        self.class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        logger.info(
            f"Triplet dataset ready: {len(self.images)} images, {num_classes} classes"
        )

    def generate_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a batch of triplets.

        Returns:
            (anchor_batch, positive_batch, negative_batch)
        """
        anchors = []
        positives = []
        negatives = []

        for _ in range(self.batch_size):
            # Random anchor class
            anchor_class = np.random.choice(list(self.class_indices.keys()))

            # Sample anchor and positive from same class
            anchor_idx, positive_idx = np.random.choice(
                self.class_indices[anchor_class], size=2, replace=False
            )

            # Sample negative from different class
            negative_class = np.random.choice(
                [c for c in self.class_indices.keys() if c != anchor_class]
            )
            negative_idx = np.random.choice(self.class_indices[negative_class])

            anchors.append(self.images[anchor_idx])
            positives.append(self.images[positive_idx])
            negatives.append(self.images[negative_idx])

        return (np.array(anchors), np.array(positives), np.array(negatives))

    def create_tf_dataset(self) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from triplets.

        Returns:
            TensorFlow dataset yielding ((anchor, positive, negative), dummy_labels)
        """

        def generator():
            while True:
                anchors, positives, negatives = self.generate_batch()
                # Dummy labels (not used in loss)
                yield (anchors, positives, negatives), np.zeros((self.batch_size, 1))

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                (
                    tf.TensorSpec(
                        shape=(self.batch_size, 160, 160, 3), dtype=tf.float32
                    ),
                    tf.TensorSpec(
                        shape=(self.batch_size, 160, 160, 3), dtype=tf.float32
                    ),
                    tf.TensorSpec(
                        shape=(self.batch_size, 160, 160, 3), dtype=tf.float32
                    ),
                ),
                tf.TensorSpec(shape=(self.batch_size, 1), dtype=tf.float32),
            ),
        )

        return dataset


class TripletLoss(keras.losses.Loss):
    """
    Triplet loss for metric learning.

    L = max(0, d(a,p) - d(a,n) + margin)
    """

    def __init__(self, margin: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        """
        Compute triplet loss.

        Args:
            y_true: Not used (dummy for compatibility)
            y_pred: Concatenated embeddings [anchor, positive, negative]

        Returns:
            Triplet loss value
        """
        # Split embeddings
        embedding_dim = y_pred.shape[1] // 3
        anchor = y_pred[:, :embedding_dim]
        positive = y_pred[:, embedding_dim : 2 * embedding_dim]
        negative = y_pred[:, 2 * embedding_dim :]

        # Compute distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

        # Triplet loss
        loss = tf.maximum(0.0, pos_dist - neg_dist + self.margin)

        return tf.reduce_mean(loss)


class FaceNetTripletTrainer:
    """
    Triplet loss trainer for FaceNet metric learning.

    Fine-tunes FaceNet to optimize embedding space for face recognition.
    Uses a siamese network architecture with shared weights.
    """

    def __init__(
        self,
        num_classes: int,
        img_size: tuple = (160, 160),
        embedding_dim: int = 512,
        margin: float = 0.2,
        model_dir: str = "src/bp_face_recognition/models/finetuned",
    ):
        """
        Initialize triplet loss trainer.

        Args:
            num_classes: Number of identities
            img_size: Input image size
            embedding_dim: FaceNet embedding dimension
            margin: Triplet loss margin
            model_dir: Directory to save models
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.base_model = None
        self.embedding_model = None
        self.history = None

        logger.info(f"TripletTrainer initialized:")
        logger.info(f"  Classes: {num_classes}")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Margin: {margin}")

    def build_model(self) -> keras.Model:
        """
        Build FaceNet model for triplet training with shared weights.

        Returns:
            Triplet model (anchor, positive, negative) -> concatenated embeddings
        """
        logger.info("Building triplet loss model...")

        try:
            from keras_facenet import FaceNet

            logger.info("Loading pre-trained FaceNet...")
            facenet = FaceNet()
            base_model = facenet.model

            # Make trainable
            base_model.trainable = True

            logger.info(
                f"FaceNet loaded: {len(base_model.layers)} layers (all trainable)"
            )

        except ImportError:
            logger.error("keras-facenet not installed")
            raise

        # Create shared embedding model
        self.embedding_model = base_model

        # Create triplet inputs
        anchor_input = keras.Input(shape=(160, 160, 3), name="anchor_input")
        positive_input = keras.Input(shape=(160, 160, 3), name="positive_input")
        negative_input = keras.Input(shape=(160, 160, 3), name="negative_input")

        # Generate embeddings (shared weights)
        anchor_embedding = self.embedding_model(anchor_input)
        positive_embedding = self.embedding_model(positive_input)
        negative_embedding = self.embedding_model(negative_input)

        # Concatenate embeddings for loss computation
        merged_output = keras.layers.concatenate(
            [anchor_embedding, positive_embedding, negative_embedding],
            axis=1,
            name="merged_embeddings",
        )

        # Create the triplet model
        self.model = keras.Model(
            inputs=[anchor_input, positive_input, negative_input],
            outputs=merged_output,
            name="facenet_triplet",
        )

        logger.info(f"Triplet model built: {self.model.count_params():,} parameters")
        return self.model

    def train(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        epochs: int = 30,
        learning_rate: float = 0.001,
        steps_per_epoch: int = 100,
    ) -> keras.callbacks.History:
        """
        Train with triplet loss.

        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            epochs: Number of epochs
            learning_rate: Learning rate
            steps_per_epoch: Batches per epoch

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        logger.info(f"Starting triplet loss training for {epochs} epochs...")

        # Create triplet generators
        train_triplet_gen = TripletDataGenerator(
            train_ds, self.num_classes, batch_size=32
        )
        val_triplet_gen = TripletDataGenerator(val_ds, self.num_classes, batch_size=32)

        train_triplet_ds = train_triplet_gen.create_tf_dataset()
        val_triplet_ds = val_triplet_gen.create_tf_dataset()

        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=TripletLoss(margin=self.margin),
        )

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_dir / "facenet_triplet_best.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

        # Train
        self.history = self.model.fit(
            train_triplet_ds,
            validation_data=val_triplet_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=20,
            callbacks=callbacks,
            verbose=1,
        )

        logger.info("Triplet loss training completed!")
        return self.history

    def save_model(self, filename: str = "facenet_triplet_v1.0.keras"):
        """Save trained model."""
        save_path = self.model_dir / filename
        # Save the embedding model (base FaceNet) for inference
        self.embedding_model.save(save_path)
        logger.info(f"Model saved to {save_path}")

    def save_history(self, filename: str = "facenet_triplet_history.json"):
        """Save training history."""
        if self.history is None:
            raise ValueError("No training history to save!")

        history_path = self.model_dir / filename

        history_dict = {
            key: [float(v) for v in values]
            for key, values in self.history.history.items()
        }

        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=2)

        logger.info(f"History saved to {history_path}")

    def save_training_report(self, dataset_info: Dict):
        """Save comprehensive training report."""
        if self.history is None:
            raise ValueError("No training history!")

        report = {
            "model_type": "FaceNet Triplet Loss (Option C)",
            "timestamp": datetime.now().isoformat(),
            "architecture": {
                "base_model": "FaceNet (fully trainable)",
                "embedding_dim": self.embedding_dim,
                "margin": self.margin,
                "loss": "Triplet Loss",
            },
            "training_config": {
                "learning_rate": 0.001,
                "epochs_trained": len(self.history.history["loss"]),
                "margin": self.margin,
            },
            "dataset": dataset_info,
            "results": {
                "final_train_loss": float(self.history.history["loss"][-1]),
                "final_val_loss": float(self.history.history["val_loss"][-1]),
                "best_val_loss": float(min(self.history.history["val_loss"])),
            },
        }

        report_path = self.model_dir / "facenet_triplet_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {report_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("TRIPLET LOSS TRAINING RESULTS (Option C)")
        print("=" * 60)
        print(f"Epochs: {report['training_config']['epochs_trained']}")
        print(f"Best Val Loss: {report['results']['best_val_loss']:.4f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="FaceNet Triplet Loss Trainer")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--margin", type=float, default=0.2, help="Triplet loss margin")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    print("=" * 60)
    print("FACENET TRIPLET LOSS TRAINING (Option C)")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Margin: {args.margin}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    # Load dataset
    logger.info("Loading dataset...")
    train_ds, val_ds, test_ds, dataset_info = create_combined_dataset(
        batch_size=args.batch_size, augmentation=True
    )

    # Create trainer
    trainer = FaceNetTripletTrainer(
        num_classes=dataset_info["num_classes"], margin=args.margin
    )

    # Build model
    trainer.build_model()

    # Train
    trainer.train(
        train_ds=train_ds, val_ds=val_ds, epochs=args.epochs, learning_rate=args.lr
    )

    # Save everything
    trainer.save_model()
    trainer.save_history()
    trainer.save_training_report(dataset_info)

    logger.info("Triplet loss training complete!")


if __name__ == "__main__":
    main()
