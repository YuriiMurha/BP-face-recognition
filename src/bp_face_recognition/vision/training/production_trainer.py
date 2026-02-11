#!/usr/bin/env python3
"""
Production Training Script for Face Recognition Models
Supports EfficientNetB0 and MobileNetV3 backbones with comprehensive logging
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV3Small
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from bp_face_recognition.config.settings import settings


class TrainingLogger:
    """Comprehensive training logger with hardware and performance tracking."""

    def __init__(self, backbone, dataset, platform="auto"):
        self.backbone = backbone
        self.dataset = dataset
        self.platform = self._detect_platform() if platform == "auto" else platform
        self.start_time = time.time()

        # Log file setup
        self.log_file = settings.LOGS_DIR / f"{backbone.lower()}_{dataset}_training.log"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.results = {
            "training_info": self._get_training_info(),
            "parameters": {},
            "training_times": {},
            "performance_metrics": {},
        }

    def _detect_platform(self):
        """Detect if GPU is available."""
        gpus = tf.config.list_physical_devices("GPU")
        return "GPU" if gpus else "CPU"

    def _get_training_info(self):
        """Get hardware and environment info."""
        gpus = tf.config.list_physical_devices("GPU")
        return {
            "timestamp": datetime.now().isoformat(),
            "tensorflow_version": tf.__version__,
            "gpu_count": len(gpus),
            "gpu_available": len(gpus) > 0,
            "gpu_details": [str(gpu) for gpu in gpus] if gpus else [],
            "built_with_cuda": tf.test.is_built_with_cuda()
            if hasattr(tf.test, "is_built_with_cuda")
            else False,
            "backbone": self.backbone,
            "dataset": self.dataset,
            "platform": self.platform,
        }

    def log(self, message):
        """Log message to both console and file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)

        with open(self.log_file, "a") as f:
            f.write(formatted + "\n")

    def start_phase(self, phase_name):
        """Log the start of a training phase."""
        self.log(f"=== Phase: {phase_name} ===")
        return time.time()

    def end_phase(self, phase_name, phase_start):
        """Log the end of a training phase."""
        phase_time = time.time() - phase_start
        self.results["training_times"][phase_name.lower().replace(" ", "_")] = (
            phase_time
        )
        self.log(
            f"Phase {phase_name} completed in {phase_time:.2f}s ({phase_time/60:.2f}m)"
        )
        return phase_time


def load_dataset(dataset_name, subset, batch_size=32, logger=None):
    """Load face recognition dataset."""
    base_path = settings.CROPPED_DIR / dataset_name / subset

    if not base_path.exists():
        raise FileNotFoundError(f"Dataset subset not found: {base_path}")

    # Get all image files
    image_files = list(base_path.glob("*.jpg"))
    if not image_files:
        raise ValueError(f"No images found in {base_path}")

    if logger:
        logger.log(f"Found {len(image_files)} images in {subset} set")

    def load_and_preprocess(image_path):
        # Load image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Extract label from filename - simplified approach
        # Use Python string operations instead of TensorFlow for this
        path_str = (
            image_path.numpy().decode("utf-8")
            if hasattr(image_path, "numpy")
            else str(image_path)
        )
        filename = os.path.basename(path_str)

        # Extract person ID from filename (format: uuid.person_id.jpg)
        parts = filename.split(".")
        if len(parts) >= 3:
            label_str = parts[-2]  # Person ID is second to last
        else:
            # Fallback for malformed filenames
            import re

            match = re.search(r"(\d+)", filename)
            label_str = match.group(1) if match else "0"

        try:
            label = int(label_str)
        except ValueError:
            label = 0

        return image, label

    # Create dataset
    dataset = tf.data.Dataset.list_files(str(base_path / "*.jpg"))
    if subset == "train":
        dataset = dataset.shuffle(1000)

    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset


def get_num_classes(dataset_name):
    """Count unique person labels in dataset."""
    max_label = 0

    for subset in ["train", "val", "test"]:
        subset_path = settings.CROPPED_DIR / dataset_name / subset
        if not subset_path.exists():
            continue

        for img_file in subset_path.glob("*.jpg"):
            try:
                # Extract label from filename format: uuid.person.jpg
                label = int(img_file.name.split(".")[-2])
                max_label = max(max_label, label)
            except (IndexError, ValueError):
                continue

    return max_label + 1


def build_model(num_classes, backbone="EfficientNetB0", input_shape=(224, 224, 3)):
    """Build face recognition model."""
    input_layer = Input(shape=input_shape)

    # Build backbone
    if backbone == "EfficientNetB0":
        base_model = EfficientNetB0(
            weights="imagenet", include_top=False, input_tensor=input_layer
        )
    elif backbone == "MobileNetV3Small":
        base_model = MobileNetV3Small(
            weights="imagenet", include_top=False, input_tensor=input_layer
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # Freeze backbone initially
    base_model.trainable = False

    # Add custom layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation="relu", name="face_embedding")(x)
    x = Dropout(0.5)(x)

    # Output layer
    activation = "softmax" if num_classes > 1 else "sigmoid"
    output = Dense(num_classes, activation=activation)(x)

    return Model(inputs=input_layer, outputs=output), base_model


class ProductionTrainer:
    """Production trainer with comprehensive logging and model management."""

    def __init__(
        self,
        dataset_name="seccam_2",
        backbone="EfficientNetB0",
        epochs=20,
        batch_size=32,
        learning_rate=1e-3,
        fine_tune=True,
    ):
        self.dataset_name = dataset_name
        self.backbone = backbone
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.fine_tune = fine_tune

        # Initialize logger
        self.logger = TrainingLogger(backbone, dataset_name)

        # Get dataset info
        self.num_classes = get_num_classes(dataset_name)
        self.logger.log(f"Dataset: {dataset_name}, Classes: {self.num_classes}")

        # Build model
        self.model, self.base_model = build_model(self.num_classes, backbone)
        self.compile_model(learning_rate)

        # Setup model paths
        model_name = f"{backbone.lower()}_{dataset_name}_{self.logger.platform.lower()}"
        self.checkpoint_path = (
            settings.MODELS_DIR / f"checkpoints/{model_name}_best.keras"
        )
        self.final_model_path = settings.MODELS_DIR / f"{model_name}_final.keras"

        # Ensure directories exist
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.final_model_path.parent.mkdir(parents=True, exist_ok=True)

    def compile_model(self, learning_rate):
        """Compile the model with appropriate loss and metrics."""
        loss = (
            "sparse_categorical_crossentropy"
            if self.num_classes > 1
            else "binary_crossentropy"
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=["accuracy"],
        )

    def create_timing_callback(self):
        """Create a callback to track epoch timing."""
        trainer_self = self  # Reference to outer class
        epochs = self.epochs

        class TimingCallback(tf.keras.callbacks.Callback):
            def __init__(self, outer_logger):
                super().__init__()
                self.logger = outer_logger
                self.epochs = epochs

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start = time.time()

            def on_epoch_end(self, epoch, logs=None):
                epoch_time = time.time() - self.epoch_start
                if logs:
                    self.logger.log(
                        f"Epoch {epoch+1}/{self.epochs} - "
                        f"loss: {logs['loss']:.4f}, "
                        f"acc: {logs['accuracy']:.4f}, "
                        f"val_loss: {logs['val_loss']:.4f}, "
                        f"val_acc: {logs['val_accuracy']:.4f} "
                        f"({epoch_time:.2f}s)"
                    )

                    # Store timing
                    self.logger.results["training_times"][f"epoch_{epoch+1}"] = (
                        epoch_time
                    )

        return TimingCallback(self.logger)

    def create_callbacks(self):
        """Create training callbacks."""
        callbacks = [
            ModelCheckpoint(
                str(self.checkpoint_path),
                save_best_only=True,
                monitor="val_accuracy",
                mode="max",
            ),
            EarlyStopping(
                patience=3,
                restore_best_weights=True,
                monitor="val_accuracy",
                mode="max",
            ),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6),
            self.create_timing_callback(),
        ]
        return callbacks

    def train(self):
        """Execute the complete training pipeline."""
        self.logger.log("Starting production training pipeline")

        # Update results parameters
        self.logger.results["parameters"] = {
            "dataset": self.dataset_name,
            "backbone": self.backbone,
            "platform": self.logger.platform,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_classes": self.num_classes,
            "fine_tune": self.fine_tune,
        }

        # Load datasets
        train_ds = load_dataset(
            self.dataset_name, "train", self.batch_size, self.logger
        )
        val_ds = load_dataset(self.dataset_name, "val", self.batch_size, self.logger)

        # Phase 1: Train top layers
        phase1_start = self.logger.start_phase("Training Top Layers")
        history1 = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=self.create_callbacks(),
        )
        self.logger.end_phase("Training Top Layers", phase1_start)

        # Phase 2: Fine-tuning (if enabled)
        if self.fine_tune:
            phase2_start = self.logger.start_phase("Fine-Tuning Backbone")

            # Unfreeze backbone and recompile with lower learning rate
            self.base_model.trainable = True
            self.compile_model(self.learning_rate / 10)

            history2 = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.epochs // 2,
                callbacks=self.create_callbacks(),
            )

            self.logger.end_phase("Fine-Tuning Backbone", phase2_start)

        # Save model and results
        self.save_results(history1)
        return self.final_model_path, self.logger.results

    def save_results(self, history):
        """Save model and training results."""
        total_time = time.time() - self.logger.start_time

        # Save model
        self.model.save(str(self.final_model_path))

        # Update performance metrics
        self.logger.results["performance_metrics"] = {
            "final_val_accuracy": float(max(history.history["val_accuracy"])),
            "final_train_accuracy": float(max(history.history["accuracy"])),
            "final_val_loss": float(min(history.history["val_loss"])),
            "final_train_loss": float(min(history.history["loss"])),
            "model_size_mb": self.final_model_path.stat().st_size / (1024 * 1024),
            "total_training_time_seconds": total_time,
            "total_training_time_minutes": total_time / 60,
        }

        # Save results
        results_path = (
            settings.LOGS_DIR
            / f"{self.backbone.lower()}_{self.dataset_name}_{self.logger.platform.lower()}_results.json"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(self.logger.results, f, indent=2)

        # Final summary
        self.logger.log("Training completed successfully!")
        self.logger.log(f"Model saved to: {self.final_model_path}")
        self.logger.log(f"Results saved to: {results_path}")
        self.logger.log(
            f"Final validation accuracy: {self.logger.results['performance_metrics']['final_val_accuracy']:.4f}"
        )
        self.logger.log(
            f"Total training time: {total_time:.2f}s ({total_time/60:.2f}m)"
        )
        self.logger.log(
            f"Model size: {self.logger.results['performance_metrics']['model_size_mb']:.2f} MB"
        )


def main():
    """Main training function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Production Face Recognition Model Training"
    )

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="seccam_2", help="Dataset name")

    # Model architecture arguments
    parser.add_argument(
        "--backbone",
        type=str,
        default="EfficientNetB0",
        choices=["EfficientNetB0", "MobileNetV3Small"],
        help="Backbone architecture",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Initial learning rate"
    )

    # Training options
    parser.add_argument(
        "--no-fine-tune", action="store_true", help="Disable fine-tuning phase"
    )
    parser.add_argument(
        "--force-cpu", action="store_true", help="Force CPU training (disable GPU)"
    )

    # Output options
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Set environment for CPU-only training
    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Create trainer
    trainer = ProductionTrainer(
        dataset_name=args.dataset,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fine_tune=not args.no_fine_tune,
    )

    # Start training
    try:
        model_path, results = trainer.train()
        print(f"\nTraining completed successfully!")
        print(f"Model: {model_path}")
        print(f"Accuracy: {results['performance_metrics']['final_val_accuracy']:.4f}")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
