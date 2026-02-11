"""
Vision data preprocessing module for BP Face Recognition.
Provides utilities for loading and preprocessing face datasets.
"""

import os
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# Import settings
from bp_face_recognition.config.settings import settings

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
NUM_CLASSES = 15  # Support up to 15 classes for safety


def load_image_and_label(
    image_path: tf.Tensor,
    target_image_size: Tuple[int, int] = IMAGE_SIZE,
    num_classes: int = NUM_CLASSES,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Load and preprocess image with label extraction."""
    # Read and decode image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize to target size
    image = tf.image.resize(image, target_image_size)

    # Extract label from filename (format: uuid.personID.faceID.jpg)
    filename = tf.strings.split(image_path, os.sep)[-1]
    label_parts = tf.strings.split(filename, ".")
    # The person ID is the second part: uuid.personID.faceID.jpg
    label_str = label_parts[-3] if len(label_parts) >= 4 else label_parts[-2]

    # Convert label to integer
    label = tf.strings.to_number(label_str, out_type=tf.int32)

    # Convert to one-hot encoding (adjust for 0-indexing)
    label_zero_indexed = label - 1  # Convert 1-12 to 0-11
    label_onehot = tf.one_hot(label_zero_indexed, depth=num_classes)

    return image, label_onehot


def create_dataset_from_directory(
    directory: Path,
    dataset_name: str,
    batch_size: int,
    image_size: Tuple[int, int],
    num_classes: int,
    shuffle_buffer_size: int = 1000,
) -> tf.data.Dataset:
    """Create dataset from a directory containing image files."""
    if not directory.exists():
        print(
            f"[PREPROCESSING] Warning: {dataset_name} directory not found at {directory}"
        )
        # Return empty dataset
        return tf.data.Dataset.from_tensor_slices(
            (np.empty((0, *image_size, 3)), np.empty((0, num_classes)))
        )

    # Get all image files
    image_files = list(directory.glob("*.jpg"))
    if not image_files:
        print(f"[PREPROCESSING] Warning: No images found in {directory}")
        return tf.data.Dataset.from_tensor_slices(
            (np.empty((0, *image_size, 3)), np.empty((0, num_classes)))
        )

    print(f"[PREPROCESSING] Found {len(image_files)} images in {dataset_name}")

    # Create dataset from file paths
    file_paths = [str(f) for f in image_files]
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)

    # Create a wrapper function with fixed parameters for TensorFlow
    def _load_wrapper(image_path):
        return load_image_and_label(
            image_path, target_image_size=image_size, num_classes=num_classes
        )

    # Map to load images and labels
    dataset = dataset.map(_load_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(shuffle_buffer_size if dataset_name == "train" else 100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def load_face_dataset(
    data_path: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    validation_split: float = 0.2,
    test_split: float = 0.2,
    shuffle_buffer_size: int = 1000,
    num_classes: int = NUM_CLASSES,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Load face dataset from cropped face images directory.

    Args:
        data_path: Path to the dataset directory (e.g., "data/datasets/cropped/seccam_2")
        batch_size: Batch size for loading data
        image_size: Target image size (height, width)
        validation_split: Fraction of data to use for validation (not used with pre-split data)
        test_split: Fraction of data to use for testing (not used with pre-split data)
        shuffle_buffer_size: Buffer size for shuffling
        num_classes: Number of classes for one-hot encoding

    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset)
    """
    print(f"[PREPROCESSING] Loading dataset from: {data_path}")

    # Convert to Path object
    data_dir = Path(data_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    # Get the dataset splits
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    # Create datasets for each split
    train_dataset = create_dataset_from_directory(
        train_dir, "train", batch_size, image_size, num_classes, shuffle_buffer_size
    )
    val_dataset = create_dataset_from_directory(
        val_dir, "validation", batch_size, image_size, num_classes, 100
    )
    test_dataset = create_dataset_from_directory(
        test_dir, "test", batch_size, image_size, num_classes, 100
    )

    return train_dataset, val_dataset, test_dataset


def load_cropped_seccam2_dataset(
    batch_size: int = 32,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Convenience function to load the cropped seccam_2 dataset.

    Args:
        batch_size: Batch size for loading data

    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset)
    """
    data_path = str(settings.CROPPED_DIR / "seccam_2")
    return load_face_dataset(data_path, batch_size=batch_size)


def get_dataset_info(data_path: str) -> dict:
    """
    Get information about the dataset structure.

    Args:
        data_path: Path to the dataset directory

    Returns:
        Dictionary containing dataset information
    """
    data_dir = Path(data_path)
    info = {
        "path": data_path,
        "exists": data_dir.exists(),
        "splits": {},
        "total_images": 0,
    }

    if not data_dir.exists():
        return info

    for split_name in ["train", "val", "test"]:
        split_dir = data_dir / split_name
        if split_dir.exists():
            image_files = list(split_dir.glob("*.jpg"))
            info["splits"][split_name] = len(image_files)
            info["total_images"] += len(image_files)

    return info
