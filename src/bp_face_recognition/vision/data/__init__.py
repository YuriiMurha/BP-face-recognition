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


def load_face_dataset(
    data_path: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    validation_split: float = 0.2,
    test_split: float = 0.2,
    shuffle_buffer_size: int = 1000,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Load face dataset from cropped face images directory.

    Args:
        data_path: Path to the dataset directory (e.g., "data/datasets/cropped/seccam_2")
        batch_size: Batch size for loading data
        image_size: Target image size (height, width)
        validation_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing

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

    datasets = {}

    def load_image_and_label(image_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Load and preprocess image with label extraction."""
        # Read and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Resize to target size
        image = tf.image.resize(image, image_size)

        # Extract label from filename (format: imagename.person.jpg)
        filename = tf.strings.split(image_path, os.sep)[-1]
        label_parts = tf.strings.split(filename, ".")
        # The label is typically the second-to-last part before .jpg
        label_str = label_parts[-2]

        # Convert label to integer
        label = tf.strings.to_number(label_str, out_type=tf.int32)

        # Convert to one-hot encoding (assuming max 15 classes based on dummy data)
        label_onehot = tf.one_hot(label, depth=15)

        return image, label_onehot

    def create_dataset_from_directory(
        directory: Path, dataset_name: str
    ) -> tf.data.Dataset:
        """Create dataset from a directory containing image files."""
        if not directory.exists():
            print(
                f"[PREPROCESSING] Warning: {dataset_name} directory not found at {directory}"
            )
            # Return empty dataset
            return tf.data.Dataset.from_tensor_slices(
                (np.empty((0, *image_size, 3)), np.empty((0, 15)))
            )

        # Get all image files
        image_files = list(directory.glob("*.jpg"))
        if not image_files:
            print(f"[PREPROCESSING] Warning: No images found in {directory}")
            return tf.data.Dataset.from_tensor_slices(
                (np.empty((0, *image_size, 3)), np.empty((0, 15)))
            )

        print(f"[PREPROCESSING] Found {len(image_files)} images in {dataset_name}")

        # Create dataset from file paths
        file_paths = [str(f) for f in image_files]
        dataset = tf.data.Dataset.from_tensor_slices(file_paths)

        # Map to load images and labels
        dataset = dataset.map(load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)

        # Shuffle and batch
        dataset = dataset.shuffle(
            shuffle_buffer_size if dataset_name == "train" else 100
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    # Create datasets for each split
    datasets["train"] = create_dataset_from_directory(train_dir, "train")
    datasets["val"] = create_dataset_from_directory(val_dir, "validation")
    datasets["test"] = create_dataset_from_directory(test_dir, "test")

    return datasets["train"], datasets["val"], datasets["test"]


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
