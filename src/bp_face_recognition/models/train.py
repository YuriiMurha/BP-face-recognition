import os
import argparse
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
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


def build_model(num_classes, input_shape=(224, 224, 3)):
    input_layer = Input(shape=input_shape)
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_tensor=input_layer
    )

    # Freeze the backbone
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax" if num_classes > 1 else "sigmoid")(
        x
    )

    return Model(inputs=input_layer, outputs=output)


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


def train(dataset_name, epochs=10, batch_size=32, lr=1e-3):
    print(f"Starting training for dataset: {dataset_name}")

    num_classes = get_num_classes(dataset_name)
    print(f"Detected {num_classes} classes")

    train_ds = create_dataset(dataset_name, "train", batch_size)
    val_ds = create_dataset(dataset_name, "val", batch_size)

    model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy"
        if num_classes > 1
        else "binary_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_path = settings.MODELS_DIR / f"checkpoints/{dataset_name}_best.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=str(settings.LOGS_DIR / "tensorboard")),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    final_model_path = settings.MODELS_DIR / f"{dataset_name}_final.keras"
    model.save(str(final_model_path))
    print(f"Training complete. Model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train face recognition model")
    parser.add_argument("--dataset", type=str, default="seccam_2", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()
    train(args.dataset, args.epochs, args.batch_size, args.lr)
