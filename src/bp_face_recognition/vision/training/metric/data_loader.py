"""
Triplet Data Loader for Metric Learning.

Supports:
- Single dataset path or multiple dataset paths (list)
- Two formats:
  1. Folder-based: {dataset_path}/{identity_name}/image.jpg
  2. Filename-prefix: {dataset_path}/{identity}_{index}.jpg

Usage:
    # Single dataset
    loader = TripletDataLoader("data/datasets/augmented/lfw")

    # Multiple datasets
    loader = TripletDataLoader([
        "data/datasets/augmented/lfw",
        "data/datasets/augmented/webcam",
        "data/datasets/augmented/seccam_2",
    ])
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import random
from pathlib import Path


class TripletDataLoader:
    """
    Loads images from multiple datasets and generates triplets (Anchor, Positive, Negative).
    """

    def __init__(self, dataset_paths, img_size=(224, 224), subset="train"):
        """
        Initialize the data loader.

        Args:
            dataset_paths: Single path (str) or list of paths to augmented datasets
            img_size: Target image size (height, width)
            subset: Which subset to use (train, val, test)
        """
        # Convert single path to list
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]

        self.dataset_paths = dataset_paths
        self.img_size = img_size
        self.subset = subset

        # Collect all identities and files from all datasets
        self.identities = []
        self.identity_to_files = {}

        for dataset_path in dataset_paths:
            # Handle both absolute and relative paths
            dataset_path = Path(dataset_path)

            # Check if path exists as is, or try relative to data/datasets
            if not dataset_path.exists():
                # Try data/datasets/augmented/{name}
                alt_path = Path("data/datasets/augmented") / dataset_path.name
                if alt_path.exists():
                    dataset_path = alt_path

            # Handle subset
            if not (dataset_path / subset).exists():
                # Try without subset (folder-based format)
                subset_path = dataset_path
            else:
                subset_path = dataset_path / subset

            if not subset_path.exists():
                print(f"Warning: Dataset path not found: {subset_path}")
                continue

            # Discover identities and files
            self._discover_identities(subset_path, dataset_path.name)

        # Check we have enough identities
        if len(self.identities) < 2:
            raise ValueError(
                f"Need at least 2 identities, found {len(self.identities)}"
            )

        print(
            f"Loaded {len(self.identities)} identities from {len(dataset_paths)} datasets"
        )

    def _discover_identities(self, base_path, dataset_name):
        """
        Discover identities from a dataset directory.
        Supports both folder-based and filename-prefix formats.
        """
        entries = os.listdir(base_path)

        # Check if it's folder-based (entries are directories)
        dirs = [e for e in entries if os.path.isdir(os.path.join(base_path, e))]

        if dirs:
            # Folder-based format: {identity}/image.jpg
            for identity in dirs:
                identity_dir = os.path.join(base_path, identity)
                files = [
                    os.path.join(identity_dir, f)
                    for f in os.listdir(identity_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                if files:
                    full_identity = (
                        f"{dataset_name}_{identity}"
                        if dataset_name != "lfw"
                        else identity
                    )
                    self.identities.append(full_identity)
                    self.identity_to_files[full_identity] = files
        else:
            # Filename-prefix format: {identity}_{index}.jpg
            # Group files by identity prefix
            identity_groups = {}
            for f in entries:
                if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                # Extract identity from filename prefix
                # Format: {identity}_{index}.jpg or {identity}_{original}_{idx}.jpg
                parts = f.rsplit("_", 1)[0]  # Remove last _index part
                # Find the identity by removing _aug{N} suffix if present
                if "_aug" in parts:
                    parts = parts.rsplit("_aug", 1)[0]

                identity = parts

                if identity not in identity_groups:
                    identity_groups[identity] = []
                identity_groups[identity].append(os.path.join(base_path, f))

            # Add to collections
            for identity, files in identity_groups.items():
                full_identity = (
                    f"{dataset_name}_{identity}" if dataset_name != "lfw" else identity
                )
                self.identities.append(full_identity)
                self.identity_to_files[full_identity] = files

    def _load_img(self, path):
        """Load and preprocess a single image."""
        img = Image.open(path).convert("RGB")
        img = img.resize(self.img_size)
        img = np.array(img) / 255.0
        return img.astype(np.float32)

    def generate_triplets(self, batch_size=32):
        """
        Generator for triplets.
        Returns ((anchors, positives, negatives), labels)
        """
        while True:
            anchors, positives, negatives = [], [], []

            for _ in range(batch_size):
                # 1. Pick an identity for Anchor and Positive
                identity = random.choice(self.identities)

                # Need at least 2 images for Anchor/Positive
                while len(self.identity_to_files[identity]) < 2:
                    identity = random.choice(self.identities)

                # 2. Pick two different images of this identity
                a_path, p_path = random.sample(self.identity_to_files[identity], 2)

                # 3. Pick a different identity for Negative
                neg_identity = random.choice(
                    [i for i in self.identities if i != identity]
                )
                n_path = random.choice(self.identity_to_files[neg_identity])

                anchors.append(self._load_img(a_path))
                positives.append(self._load_img(p_path))
                negatives.append(self._load_img(n_path))

            yield (
                (np.array(anchors), np.array(positives), np.array(negatives)),
                np.zeros(batch_size),
            )

    def get_dataset(self, batch_size=32):
        """
        Returns a tf.data.Dataset version of the generator.
        """
        return tf.data.Dataset.from_generator(
            lambda: self.generate_triplets(batch_size),
            output_signature=(
                (
                    tf.TensorSpec(shape=(None, *self.img_size, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, *self.img_size, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, *self.img_size, 3), dtype=tf.float32),
                ),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            ),
        )


# Backward compatibility alias
def create_triplet_dataloader(dataset_path, img_size=(224, 224), subset="train"):
    """Create a TripletDataLoader (backward compatibility)."""
    return TripletDataLoader(dataset_path, img_size, subset)
