"""Unified dataset loader for FaceNet fine-tuning

Handles custom datasets (webcam, seccam, seccam_2) with flat file structure.
Creates train/val/test splits and returns TensorFlow datasets.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class FaceNetDatasetLoader:
    """
    Unified dataset loader for custom face recognition datasets.

    Handles flat file structures where images are named like:
    - PersonName_UUID.index.jpg
    - Stranger_1_UUID.index.jpg

    Creates train/val/test splits with stratification.
    """

    def __init__(
        self,
        data_dirs: List[str],
        img_size: Tuple[int, int] = (160, 160),
        batch_size: int = 32,
        validation_split: float = 0.15,
        test_split: float = 0.15,
        augmentation: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize dataset loader.

        Args:
            data_dirs: List of directories containing images
            img_size: Target image size (height, width)
            batch_size: Batch size for training
            validation_split: Fraction for validation (0.0 to 1.0)
            test_split: Fraction for test (0.0 to 1.0)
            augmentation: Whether to apply data augmentation
            cache_dir: Directory to cache preprocessed data
        """
        self.data_dirs = [Path(d) for d in data_dirs]
        self.img_size = img_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.augmentation = augmentation
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.class_names = []
        self.class_to_idx = {}
        self.num_classes = 0

        logger.info(f"DatasetLoader initialized with dirs: {data_dirs}")

    def _parse_filename(self, filename: str) -> Tuple[str, str]:
        """
        Parse person ID from filename.

        Expected formats:
        - PersonName_UUID.index.jpg
        - Stranger_1_UUID.index.jpg

        Returns:
            (person_id, full_filename)
        """
        # Remove extension
        name = Path(filename).stem

        # Try to extract person name (everything before UUID or first number sequence)
        # Pattern: Name_part1_part2_UUID.index
        match = re.match(r"^([A-Za-z_]+(?:_[0-9]+)?)_[0-9a-f]{8}", name)

        if match:
            person_id = match.group(1)
        else:
            # Fallback: use first part before underscore
            parts = name.split("_")
            if len(parts) >= 2:
                person_id = parts[0]
            else:
                person_id = name

        return person_id, filename

    def _load_image_paths(self) -> Tuple[List[str], List[int]]:
        """
        Load all image paths and create class mappings.

        Returns:
            (image_paths, labels)
        """
        image_paths = []
        labels = []
        person_to_images = {}

        for data_dir in self.data_dirs:
            if not data_dir.exists():
                logger.warning(f"Directory not found: {data_dir}")
                continue

            # Look for images in train/val/test subdirs or root
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

            for split in ["train", "val", "test"]:
                split_dir = data_dir / split
                if split_dir.exists():
                    for ext in image_extensions:
                        for img_path in split_dir.glob(f"*{ext}"):
                            person_id, _ = self._parse_filename(img_path.name)

                            if person_id not in person_to_images:
                                person_to_images[person_id] = []
                            person_to_images[person_id].append(str(img_path))

            # Also check root directory
            for ext in image_extensions:
                for img_path in data_dir.glob(f"*{ext}"):
                    person_id, _ = self._parse_filename(img_path.name)

                    if person_id not in person_to_images:
                        person_to_images[person_id] = []
                    person_to_images[person_id].append(str(img_path))

        # Create class mappings
        self.class_names = sorted(person_to_images.keys())
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        logger.info(f"Found {self.num_classes} unique identities:")
        for idx, name in enumerate(self.class_names):
            count = len(person_to_images[name])
            logger.info(f"  {idx}: {name} ({count} images)")

        # Create flat lists
        for person_id, paths in person_to_images.items():
            label = self.class_to_idx[person_id]
            for path in paths:
                image_paths.append(path)
                labels.append(label)

        logger.info(f"Total images loaded: {len(image_paths)}")
        return image_paths, labels

    def _preprocess_image(
        self, image_path: str, label: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Load and preprocess a single image.

        Args:
            image_path: Path to image file
            label: Class label

        Returns:
            (image_tensor, label_tensor)
        """
        # Read image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.cast(img, tf.float32)

        # Resize
        img = tf.image.resize(img, self.img_size, method="bilinear")

        # Normalize to [0, 1]
        img = img / 255.0

        # FaceNet preprocessing: normalize to [-1, 1]
        img = (img - 0.5) * 2.0

        return img, tf.one_hot(label, self.num_classes)

    def _augment_image(
        self, image: tf.Tensor, label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply data augmentation.

        Args:
            image: Image tensor
            label: Label tensor

        Returns:
            (augmented_image, label)
        """
        if not self.augmentation:
            return image, label

        # Random flip
        image = tf.image.random_flip_left_right(image)

        # Random rotation (small angles for faces)
        angle = tf.random.uniform([], -0.1, 0.1)  # ~5.7 degrees
        image = tf.keras.preprocessing.image.apply_affine_transform(
            image.numpy(), theta=angle * 180 / 3.14159
        )
        image = tf.convert_to_tensor(image, dtype=tf.float32)

        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.1)

        # Random contrast
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

        # Clip values
        image = tf.clip_by_value(image, -1.0, 1.0)

        return image, label

    def load_dataset(
        self,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict]:
        """
        Load and prepare datasets for training.

        Returns:
            (train_dataset, val_dataset, test_dataset, dataset_info)
        """
        logger.info("Loading dataset...")

        # Load image paths and labels
        image_paths, labels = self._load_image_paths()

        if len(image_paths) == 0:
            raise ValueError("No images found in specified directories!")

        # Create train/val/test splits with stratification
        # First split: train vs (val+test)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths,
            labels,
            test_size=self.validation_split + self.test_split,
            stratify=labels,
            random_state=42,
        )

        # Second split: val vs test
        val_size = self.validation_split / (self.validation_split + self.test_split)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths,
            temp_labels,
            test_size=1 - val_size,
            stratify=temp_labels,
            random_state=42,
        )

        logger.info(f"Data splits:")
        logger.info(f"  Train: {len(train_paths)} images")
        logger.info(f"  Validation: {len(val_paths)} images")
        logger.info(f"  Test: {len(test_paths)} images")

        # Create TensorFlow datasets
        train_ds = self._create_tf_dataset(
            train_paths, train_labels, shuffle=True, augment=True
        )
        val_ds = self._create_tf_dataset(
            val_paths, val_labels, shuffle=False, augment=False
        )
        test_ds = self._create_tf_dataset(
            test_paths, test_labels, shuffle=False, augment=False
        )

        # Dataset info
        dataset_info = {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "num_train": len(train_paths),
            "num_val": len(val_paths),
            "num_test": len(test_paths),
            "img_size": self.img_size,
            "batch_size": self.batch_size,
        }

        return train_ds, val_ds, test_ds, dataset_info

    def _create_tf_dataset(
        self,
        image_paths: List[str],
        labels: List[int],
        shuffle: bool = True,
        augment: bool = False,
    ) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from paths and labels.

        Args:
            image_paths: List of image paths
            labels: List of labels
            shuffle: Whether to shuffle
            augment: Whether to apply augmentation

        Returns:
            TensorFlow dataset
        """
        # Create dataset
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        # Shuffle if needed
        if shuffle:
            ds = ds.shuffle(buffer_size=len(image_paths))

        # Load and preprocess images
        ds = ds.map(
            lambda x, y: tf.py_function(
                func=self._load_and_preprocess_wrapper,
                inp=[x, y],
                Tout=[tf.float32, tf.float32],
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Set shapes (required after py_function)
        ds = ds.map(
            lambda x, y: (
                tf.reshape(x, (*self.img_size, 3)),
                tf.reshape(y, (self.num_classes,)),
            )
        )

        # Apply augmentation if needed
        if augment and self.augmentation:
            ds = ds.map(self._augment_tf, num_parallel_calls=tf.data.AUTOTUNE)

        # Batch
        ds = ds.batch(self.batch_size)

        # Prefetch
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    def _load_and_preprocess_wrapper(self, image_path, label):
        """Wrapper for tf.py_function"""
        image_path_str = image_path.numpy().decode("utf-8")
        label_int = int(label.numpy())

        import cv2

        img = cv2.imread(image_path_str)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) * 2.0  # Normalize to [-1, 1]

        label_onehot = np.zeros(self.num_classes, dtype=np.float32)
        label_onehot[label_int] = 1.0

        return img, label_onehot

    def _augment_tf(self, image, label):
        """TensorFlow augmentation wrapper"""
        # Random flip
        image = tf.image.random_flip_left_right(image)

        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.1)

        # Random contrast
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

        # Clip
        image = tf.clip_by_value(image, -1.0, 1.0)

        return image, label

    def save_metadata(self, output_path: str):
        """
        Save dataset metadata to JSON.

        Args:
            output_path: Path to save metadata
        """
        metadata = {
            "class_names": self.class_names,
            "class_to_idx": self.class_to_idx,
            "num_classes": self.num_classes,
            "img_size": self.img_size,
            "batch_size": self.batch_size,
        }

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {output_path}")


def create_combined_dataset(
    webcam_dir: str = "data/datasets/augmented/webcam",
    seccam_dir: str = "data/datasets/augmented/seccam_2",
    **kwargs,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict]:
    """
    Convenience function to create combined dataset from webcam and seccam_2.

    Args:
        webcam_dir: Path to webcam dataset
        seccam_dir: Path to seccam_2 dataset
        **kwargs: Additional arguments for FaceNetDatasetLoader

    Returns:
        (train_dataset, val_dataset, test_dataset, dataset_info)
    """
    loader = FaceNetDatasetLoader(data_dirs=[webcam_dir, seccam_dir], **kwargs)

    return loader.load_dataset()


if __name__ == "__main__":
    # Test the dataset loader
    logging.basicConfig(level=logging.INFO)

    train_ds, val_ds, test_ds, info = create_combined_dataset(
        batch_size=32, augmentation=True
    )

    print(f"\nDataset loaded successfully!")
    print(f"Classes: {info['num_classes']}")
    print(f"Class names: {info['class_names']}")
    print(
        f"Train: {info['num_train']}, Val: {info['num_val']}, Test: {info['num_test']}"
    )

    # Test batch
    for images, labels in train_ds.take(1):
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.numpy().min():.2f}, {images.numpy().max():.2f}]")
