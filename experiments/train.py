import os
import argparse
import time
import json
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from pathlib import Path
from datetime import datetime
from bp_face_recognition.config.settings import settings


def load_image_and_label(image_path):
    """
    Reads and decodes an image from a given file path.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Extract the label from the filename (assuming format: name.label.jpg)
    filename = tf.strings.split(image_path, os.sep)[-1]
    label_str = tf.strings.split(filename, ".")[-2]
    label = tf.strings.to_number(label_str, out_type=tf.int32)
    return image, label


def create_dataset(dataset_source, dataset_type, batch_size=32):
    base_path = settings.CROPPED_DIR / dataset_source / dataset_type
    pattern = str(base_path / "*.jpg")

    images = tf.data.Dataset.list_files(pattern, shuffle=(dataset_type == "train"))
    dataset = images.map(load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)

    if dataset_type == "train":
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model(num_classes, backbone="EfficientNetB0", input_shape=(224, 224, 3)):
    input_layer = Input(shape=input_shape)

    if backbone == "EfficientNetB0":
        base_model = EfficientNetB0(
            weights="imagenet", include_top=False, input_tensor=input_layer
        )
    elif backbone == "MobileNetV3Small":
        base_model = tf.keras.applications.MobileNetV3Small(
            weights="imagenet", include_top=False, input_tensor=input_layer
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # Freeze the backbone initially
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation="relu", name="face_embedding")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax" if num_classes > 1 else "sigmoid")(
        x
    )

    return Model(inputs=input_layer, outputs=output), base_model


def get_num_classes(dataset_source):
    max_label = 0
    for subset in ["train", "val", "test"]:
        base_path = settings.CROPPED_DIR / dataset_source / subset
        if not base_path.exists():
            continue
        for f in base_path.glob("*.jpg"):
            try:
                label = int(f.name.split(".")[-2])
                if label > max_label:
                    max_label = label
            except (IndexError, ValueError):
                continue
    return max_label + 1


def get_training_info():
    """Get hardware and environment info for comparison logging."""
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
    }


def train(
    dataset_name,
    epochs=20,
    batch_size=32,
    lr=1e-3,
    fine_tune=True,
    backbone="EfficientNetB0",
):
    """
    Enhanced training function with comparison logging and performance tracking.
    """
    start_time = time.time()

    # Get training info
    training_info = get_training_info()
    platform = "GPU" if training_info["gpu_available"] else "CPU"

    print(f"=== Training Session Started ===")
    print(f"Dataset: {dataset_name}")
    print(f"Backbone: {backbone}")
    print(f"Platform: {platform}")
    print(f"GPU Available: {training_info['gpu_available']}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")

    num_classes = get_num_classes(dataset_name)
    print(f"Detected {num_classes} classes")

    # Create datasets
    train_ds = create_dataset(dataset_name, "train", batch_size)
    val_ds = create_dataset(dataset_name, "val", batch_size)

    # Build model
    model, base_model = build_model(num_classes, backbone)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy"
        if num_classes > 1
        else "binary_crossentropy",
        metrics=["accuracy"],
    )

    # Setup output directories
    model_name = f"{backbone.lower()}_{dataset_name}_{platform.lower()}"
    checkpoint_path = settings.MODELS_DIR / f"checkpoints/{model_name}_best.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Training results log
    results = {
        "training_info": training_info,
        "parameters": {
            "dataset": dataset_name,
            "backbone": backbone,
            "platform": platform,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "num_classes": num_classes,
            "fine_tune": fine_tune,
        },
        "training_times": {},
        "performance_metrics": {},
    }

    # Enhanced callbacks with timing
    class TimingCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            epoch_time = time.time() - self.epoch_start
            results["training_times"][f"epoch_{epoch+1}"] = epoch_time
            print(f"Epoch {epoch+1} time: {epoch_time:.2f}s")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            patience=3, restore_best_weights=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6
        ),
        TimingCallback(),
    ]

    print("\n=== Phase 1: Training top layers ===")
    phase1_start = time.time()
    history1 = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks
    )
    phase1_time = time.time() - phase1_start
    results["training_times"]["phase_1"] = phase1_time
    print(f"Phase 1 completed in {phase1_time:.2f}s")

    if fine_tune:
        print("\n=== Phase 2: Fine-tuning backbone ===")
        # Unfreeze base model
        base_model.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr / 10),
            loss="sparse_categorical_crossentropy"
            if num_classes > 1
            else "binary_crossentropy",
            metrics=["accuracy"],
        )

        phase2_start = time.time()
        history2 = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs // 2, callbacks=callbacks
        )
        phase2_time = time.time() - phase2_start
        results["training_times"]["phase_2"] = phase2_time
        print(f"Phase 2 completed in {phase2_time:.2f}s")

    # Save final model and results
    final_model_path = settings.MODELS_DIR / f"{model_name}_final.keras"
    model.save(str(final_model_path))

    total_time = time.time() - start_time
    results["training_times"]["total"] = total_time
    results["performance_metrics"]["final_val_accuracy"] = max(
        history1.history["val_accuracy"]
    )
    if fine_tune:
        results["performance_metrics"]["final_val_accuracy_ft"] = (
            max(history2.history["val_accuracy"]) if "history2" in locals() else 0
        )
    results["performance_metrics"]["model_size_mb"] = (
        final_model_path.stat().st_size / (1024 * 1024)
    )

    # Save training results
    results_path = settings.LOGS_DIR / f"{model_name}_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Training Complete ===")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
    print(f"Model saved to: {final_model_path}")
    print(f"Results saved to: {results_path}")
    print(
        f"Final validation accuracy: {results['performance_metrics']['final_val_accuracy']:.4f}"
    )

    return final_model_path, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train face recognition model with comparison logging"
    )
    parser.add_argument("--dataset", type=str, default="seccam_2", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        default=True,
        help="Enable fine-tuning of backbone",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="EfficientNetB0",
        choices=["EfficientNetB0", "MobileNetV3Small"],
        help="Backbone architecture",
    )

    args = parser.parse_args()
    train(
        args.dataset,
        args.epochs,
        args.batch_size,
        args.lr,
        fine_tune=args.fine_tune,
        backbone=args.backbone,
    )
