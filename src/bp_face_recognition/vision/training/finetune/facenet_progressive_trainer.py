"""FaceNet Progressive Unfreezing Trainer - Option B

Fine-tunes FaceNet by progressively unfreezing layers from top to bottom.
Uses very low learning rates to protect pre-trained weights.

Training Strategy:
    Phase 1: Train classification head only (all base frozen) - 5 epochs
    Phase 2: Unfreeze top 20% of base layers (LR=1e-5) - 5 epochs
    Phase 3: Unfreeze top 40% of base layers (LR=5e-6) - 5 epochs
    Phase 4: Fine-tune all with very low LR (LR=1e-6) - 5 epochs

Usage:
    python facenet_progressive_trainer.py --epochs-per-phase 5 --batch-size 32
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
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


class FaceNetProgressiveTrainer:
    """
    Progressive unfreezing trainer for FaceNet fine-tuning.

    Gradually unfreezes layers from top to bottom with decreasing learning rates
    to adapt pre-trained features to the target domain while preserving knowledge.
    """

    def __init__(
        self,
        num_classes: int,
        img_size: tuple = (160, 160),
        hidden_units: int = 256,
        dropout_rate: float = 0.5,
        model_dir: str = "src/bp_face_recognition/models/finetuned",
    ):
        """
        Initialize progressive unfreezing trainer.

        Args:
            num_classes: Number of output classes
            img_size: Input image size
            hidden_units: Hidden layer units in classification head
            dropout_rate: Dropout rate for regularization
            model_dir: Directory to save models
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.base_model = None
        self.history = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []}

        logger.info(f"ProgressiveTrainer initialized:")
        logger.info(f"  Classes: {num_classes}")
        logger.info(f"  Progressive unfreezing strategy: 4 phases")

    def build_model(self) -> keras.Model:
        """
        Build model with FaceNet base and classification head.

        Returns:
            Compiled Keras model
        """
        logger.info("Building progressive fine-tuning model...")

        try:
            from keras_facenet import FaceNet

            logger.info("Loading pre-trained FaceNet...")
            facenet = FaceNet()
            self.base_model = facenet.model

            logger.info(f"FaceNet loaded: {len(self.base_model.layers)} layers")

        except ImportError:
            logger.error("keras-facenet not installed")
            raise

        # Freeze all layers initially
        self.base_model.trainable = False

        # Build classification head
        inputs = keras.Input(shape=(*self.img_size, 3))
        x = self.base_model(inputs, training=False)
        x = keras.layers.Dense(self.hidden_units, activation="relu")(x)
        x = keras.layers.Dropout(self.dropout_rate)(x)
        outputs = keras.layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = keras.Model(inputs, outputs, name="facenet_progressive")

        logger.info(f"Model built: {len(self.model.layers)} total layers")
        return self.model

    def _count_trainable_layers(self) -> int:
        """Count trainable layers in base model."""
        return sum(1 for layer in self.base_model.layers if layer.trainable)

    def _unfreeze_top_layers(self, percentage: float):
        """
        Unfreeze top percentage of base model layers.

        Args:
            percentage: Percentage of layers to unfreeze (0.0 to 1.0)
        """
        total_layers = len(self.base_model.layers)
        num_unfreeze = int(total_layers * percentage)

        # Freeze all first
        for layer in self.base_model.layers:
            layer.trainable = False

        # Unfreeze top layers
        for layer in self.base_model.layers[-num_unfreeze:]:
            layer.trainable = True

        trainable_count = sum(1 for l in self.base_model.layers if l.trainable)
        logger.info(
            f"Unfroze top {percentage:.0%} of base: {trainable_count}/{total_layers} layers"
        )

    def _compile_model(self, learning_rate: float):
        """Compile model with given learning rate."""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info(f"Compiled with LR={learning_rate}")

    def train_phase(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        epochs: int,
        learning_rate: float,
        phase_name: str,
    ) -> Dict:
        """
        Train a single phase of progressive unfreezing.

        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            epochs: Number of epochs for this phase
            learning_rate: Learning rate for this phase
            phase_name: Name of the phase (for logging)

        Returns:
            Training history for this phase
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE: {phase_name}")
        logger.info(f"Trainable layers: {self._count_trainable_layers()}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"{'='*60}\n")

        # Compile with appropriate LR
        self._compile_model(learning_rate)

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(
                    self.model_dir
                    / f'facenet_progressive_{phase_name.lower().replace(" ", "_")}.keras'
                ),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ]

        # Train
        phase_history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # Append to overall history
        for key in ["accuracy", "loss", "val_accuracy", "val_loss"]:
            self.history[key].extend(phase_history.history[key])

        return phase_history

    def train(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        epochs_per_phase: int = 5,
    ) -> Dict:
        """
        Execute progressive unfreezing training across all phases.

        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            epochs_per_phase: Epochs to train in each phase

        Returns:
            Complete training history
        """
        logger.info("Starting progressive unfreezing training...")

        # Phase 1: Train classification head only (base fully frozen)
        self.train_phase(
            train_ds, val_ds, epochs_per_phase, 0.001, "Phase 1: Head Only"
        )

        # Phase 2: Unfreeze top 20% of base (low LR)
        self._unfreeze_top_layers(0.20)
        self.train_phase(
            train_ds, val_ds, epochs_per_phase, 1e-5, "Phase 2: Top 20% Unfrozen"
        )

        # Phase 3: Unfreeze top 40% of base (very low LR)
        self._unfreeze_top_layers(0.40)
        self.train_phase(
            train_ds, val_ds, epochs_per_phase, 5e-6, "Phase 3: Top 40% Unfrozen"
        )

        # Phase 4: Unfreeze all (extremely low LR)
        self._unfreeze_top_layers(1.0)
        self.train_phase(
            train_ds, val_ds, epochs_per_phase, 1e-6, "Phase 4: Full Fine-tuning"
        )

        logger.info("Progressive unfreezing complete!")
        return self.history

    def evaluate(self, test_ds: tf.data.Dataset) -> Dict:
        """Evaluate on test set."""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        logger.info("Evaluating on test set...")
        results = self.model.evaluate(test_ds, verbose=1, return_dict=True)

        logger.info(f"Test results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")

        return results

    def save_model(self, filename: str = "facenet_progressive_v1.0.keras"):
        """Save final model."""
        save_path = self.model_dir / filename
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")

    def save_history(self, filename: str = "facenet_progressive_history.json"):
        """Save training history."""
        history_path = self.model_dir / filename

        history_dict = {
            key: [float(v) for v in values] for key, values in self.history.items()
        }

        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=2)

        logger.info(f"History saved to {history_path}")

    def save_training_report(self, dataset_info: Dict, test_results: Dict):
        """Save comprehensive training report."""
        report = {
            "model_type": "FaceNet Progressive Unfreezing (Option B)",
            "timestamp": datetime.now().isoformat(),
            "architecture": {
                "base_model": "FaceNet with progressive unfreezing",
                "phases": [
                    "Phase 1: Head only (frozen base)",
                    "Phase 2: Top 20% unfrozen (LR=1e-5)",
                    "Phase 3: Top 40% unfrozen (LR=5e-6)",
                    "Phase 4: Full unfrozen (LR=1e-6)",
                ],
            },
            "training_config": {
                "epochs_per_phase": 5,
                "total_epochs": len(self.history["loss"]),
                "progressive_strategy": True,
            },
            "dataset": dataset_info,
            "results": {
                "final_train_accuracy": float(self.history["accuracy"][-1]),
                "final_val_accuracy": float(self.history["val_accuracy"][-1]),
                "best_val_accuracy": float(max(self.history["val_accuracy"])),
                "test_accuracy": test_results.get("accuracy"),
                "test_loss": test_results.get("loss"),
            },
        }

        report_path = self.model_dir / "facenet_progressive_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {report_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("PROGRESSIVE UNFREEZING RESULTS (Option B)")
        print("=" * 60)
        print(f"Total epochs: {report['training_config']['total_epochs']}")
        print(f"Best Val Accuracy: {report['results']['best_val_accuracy']:.4f}")
        print(f"Test Accuracy: {report['results']['test_accuracy']:.4f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="FaceNet Progressive Unfreezing Trainer"
    )
    parser.add_argument(
        "--epochs-per-phase", type=int, default=5, help="Epochs per unfreezing phase"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--hidden-units", type=int, default=256, help="Hidden layer units"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FACENET PROGRESSIVE UNFREEZING (Option B)")
    print("=" * 60)
    print(f"Epochs per phase: {args.epochs_per_phase}")
    print(f"Total expected epochs: {args.epochs_per_phase * 4}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    # Load dataset
    logger.info("Loading dataset...")
    train_ds, val_ds, test_ds, dataset_info = create_combined_dataset(
        batch_size=args.batch_size, augmentation=True
    )

    # Create trainer
    trainer = FaceNetProgressiveTrainer(
        num_classes=dataset_info["num_classes"], hidden_units=args.hidden_units
    )

    # Build model
    trainer.build_model()

    # Train with progressive unfreezing
    trainer.train(
        train_ds=train_ds, val_ds=val_ds, epochs_per_phase=args.epochs_per_phase
    )

    # Evaluate
    test_results = trainer.evaluate(test_ds)

    # Save everything
    trainer.save_model()
    trainer.save_history()
    trainer.save_training_report(dataset_info, test_results)

    logger.info("Progressive unfreezing training complete!")


if __name__ == "__main__":
    main()
