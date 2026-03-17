"""FaceNet Transfer Learning Trainer - Option A

Trains a classification head on top of frozen FaceNet base.
This is the safest fine-tuning approach with minimal risk.

Architecture:
    FaceNet (Frozen) -> GlobalAveragePooling -> Dense(256) -> Dropout -> Dense(N classes) -> Softmax

Usage:
    python facenet_transfer_trainer.py --epochs 20 --batch-size 32
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from bp_face_recognition.vision.training.finetune.dataset_loader import (
    create_combined_dataset,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FaceNetTransferTrainer:
    """
    Transfer learning trainer for FaceNet.

    Freezes FaceNet base and trains only the classification head.
    """

    def __init__(
        self,
        num_classes: int,
        img_size: tuple = (160, 160),
        learning_rate: float = 0.001,
        dropout_rate: float = 0.5,
        hidden_units: int = 256,
        model_dir: str = "src/bp_face_recognition/models/finetuned",
    ):
        """
        Initialize transfer learning trainer.

        Args:
            num_classes: Number of output classes (identities)
            img_size: Input image size
            learning_rate: Learning rate for training
            dropout_rate: Dropout rate for regularization
            hidden_units: Number of units in hidden dense layer
            model_dir: Directory to save trained models
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.hidden_units = hidden_units
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.history = None

        logger.info(f"TransferTrainer initialized:")
        logger.info(f"  Classes: {num_classes}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Hidden units: {hidden_units}")

    def build_model(self) -> keras.Model:
        """
        Build transfer learning model.

        Loads pre-trained FaceNet and adds classification head.

        Returns:
            Compiled Keras model
        """
        logger.info("Building transfer learning model...")

        try:
            # Import FaceNet
            from keras_facenet import FaceNet

            # Load pre-trained FaceNet
            logger.info("Loading pre-trained FaceNet...")
            facenet = FaceNet()
            base_model = facenet.model

            logger.info(f"FaceNet base model loaded:")
            logger.info(f"  Input shape: {base_model.input_shape}")
            logger.info(f"  Output shape: {base_model.output_shape}")

        except ImportError:
            logger.error(
                "keras-facenet not installed. Install with: uv pip install keras-facenet"
            )
            raise

        # Freeze base model
        base_model.trainable = False
        logger.info(f"Frozen {len(base_model.layers)} layers in base model")

        # Build classification head
        inputs = keras.Input(shape=(*self.img_size, 3))

        # Use FaceNet for feature extraction (outputs 512D embeddings directly)
        x = base_model(inputs, training=False)

        # FaceNet outputs 512D embeddings, no pooling needed
        # Add classification head on top of embeddings
        x = keras.layers.Dense(self.hidden_units, activation="relu")(x)
        x = keras.layers.Dropout(self.dropout_rate)(x)

        # Output layer
        outputs = keras.layers.Dense(
            self.num_classes, activation="softmax", name="predictions"
        )(x)

        # Create model
        model = keras.Model(inputs, outputs, name="facenet_transfer")

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model

        logger.info(f"Model built successfully:")
        logger.info(f"  Total layers: {len(model.layers)}")
        logger.info(f"  Trainable params: {model.count_params():,}")

        return model

    def train(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        epochs: int = 20,
        early_stopping_patience: int = 5,
        reduce_lr_patience: int = 3,
    ) -> keras.callbacks.History:
        """
        Train the transfer learning model.

        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        logger.info(f"Starting training for {epochs} epochs...")

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-6,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_dir / "facenet_transfer_best.keras"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ]

        # Train
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        logger.info("Training completed!")

        return self.history

    def evaluate(self, test_ds: tf.data.Dataset) -> Dict:
        """
        Evaluate model on test set.

        Args:
            test_ds: Test dataset

        Returns:
            Evaluation metrics dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        logger.info("Evaluating on test set...")

        results = self.model.evaluate(test_ds, verbose=1, return_dict=True)

        logger.info(f"Test results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")

        return results

    def save_model(self, filename: str = "facenet_transfer_v1.0.keras"):
        """
        Save trained model.

        Args:
            filename: Filename for saved model
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        save_path = self.model_dir / filename
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")

    def save_history(self, filename: str = "facenet_transfer_history.json"):
        """
        Save training history.

        Args:
            filename: Filename for history JSON
        """
        if self.history is None:
            raise ValueError("No training history to save!")

        history_path = self.model_dir / filename

        # Convert numpy arrays to lists for JSON serialization
        history_dict = {
            key: [float(v) for v in values]
            for key, values in self.history.history.items()
        }

        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=2)

        logger.info(f"Training history saved to {history_path}")

    def save_training_report(self, dataset_info: Dict, test_results: Dict):
        """
        Save comprehensive training report.

        Args:
            dataset_info: Dataset information dictionary
            test_results: Test evaluation results
        """
        report = {
            "model_type": "FaceNet Transfer Learning (Option A)",
            "timestamp": datetime.now().isoformat(),
            "architecture": {
                "base_model": "FaceNet (frozen)",
                "hidden_units": self.hidden_units,
                "dropout_rate": self.dropout_rate,
                "num_classes": self.num_classes,
            },
            "training_config": {
                "learning_rate": self.learning_rate,
                "epochs_trained": len(self.history.history["loss"])
                if self.history
                else 0,
                "early_stopping": True,
            },
            "dataset": dataset_info,
            "results": {
                "final_train_accuracy": float(self.history.history["accuracy"][-1])
                if self.history
                else None,
                "final_val_accuracy": float(self.history.history["val_accuracy"][-1])
                if self.history
                else None,
                "best_val_accuracy": float(max(self.history.history["val_accuracy"]))
                if self.history
                else None,
                "test_accuracy": test_results.get("accuracy"),
                "test_loss": test_results.get("loss"),
            },
        }

        report_path = self.model_dir / "facenet_transfer_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Training report saved to {report_path}")

        # Also print summary
        print("\n" + "=" * 60)
        print("TRANSFER LEARNING RESULTS (Option A)")
        print("=" * 60)
        print(
            f"Model: FaceNet + Dense({self.hidden_units}) + Dropout({self.dropout_rate})"
        )
        print(f"Classes: {self.num_classes}")
        print(f"Epochs: {report['training_config']['epochs_trained']}")
        print(
            f"\nBest Validation Accuracy: {report['results']['best_val_accuracy']:.4f}"
        )
        print(f"Test Accuracy: {report['results']['test_accuracy']:.4f}")
        print(f"Test Loss: {report['results']['test_loss']:.4f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="FaceNet Transfer Learning Trainer")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--hidden-units", type=int, default=256, help="Hidden layer units"
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument(
        "--webcam-dir", type=str, default="data/datasets/augmented/webcam"
    )
    parser.add_argument(
        "--seccam-dir", type=str, default="data/datasets/augmented/seccam_2"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FACENET TRANSFER LEARNING (Option A)")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Hidden units: {args.hidden_units}")
    print(f"Dropout: {args.dropout}")
    print("=" * 60)

    # Load dataset
    logger.info("Loading dataset...")
    train_ds, val_ds, test_ds, dataset_info = create_combined_dataset(
        webcam_dir=args.webcam_dir,
        seccam_dir=args.seccam_dir,
        batch_size=args.batch_size,
        augmentation=True,
    )

    # Create trainer
    trainer = FaceNetTransferTrainer(
        num_classes=dataset_info["num_classes"],
        learning_rate=args.lr,
        hidden_units=args.hidden_units,
        dropout_rate=args.dropout,
    )

    # Build model
    trainer.build_model()

    # Train
    history = trainer.train(train_ds=train_ds, val_ds=val_ds, epochs=args.epochs)

    # Evaluate
    test_results = trainer.evaluate(test_ds)

    # Save everything
    trainer.save_model()
    trainer.save_history()
    trainer.save_training_report(dataset_info, test_results)

    logger.info("Transfer learning complete!")


if __name__ == "__main__":
    main()
