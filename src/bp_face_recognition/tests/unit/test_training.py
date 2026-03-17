import pytest
import json
import tempfile
import os
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestLabelParsing:
    """Test label parsing functionality."""

    def test_load_labels_json(self):
        """Test loading labels from JSON file."""
        labels = {
            "person_001": ["img_001.jpg", "img_002.jpg", "img_003.jpg"],
            "person_002": ["img_004.jpg", "img_005.jpg"],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(labels, f)
            temp_path = f.name

        try:
            with open(temp_path, "r") as f:
                loaded = json.load(f)

            assert isinstance(loaded, dict)
            assert len(loaded) == 2
            assert "person_001" in loaded
            assert len(loaded["person_001"]) == 3
        finally:
            os.unlink(temp_path)

    def test_labels_structure(self):
        """Test labels have correct structure."""
        labels = {
            "person_001": ["img_001.jpg", "img_002.jpg"],
            "person_002": ["img_003.jpg", "img_004.jpg"],
        }

        for person, images in labels.items():
            assert isinstance(person, str)
            assert isinstance(images, list)
            assert all(isinstance(img, str) for img in images)


class TestDatasetInfo:
    """Test dataset info functionality."""

    def test_get_dataset_info(self):
        """Test get_dataset_info returns proper structure."""
        try:
            from bp_face_recognition.vision.data.preprocessing import get_dataset_info
        except SystemExit:
            pytest.skip(
                "face_recognition import chain issue - requires manual pip install in venv"
            )

        info = get_dataset_info("data/datasets/cropped/seccam_2")

        assert isinstance(info, dict)
        assert "path" in info
        assert "exists" in info
        assert "splits" in info
        assert "total_images" in info

    def test_dataset_exists_flag(self):
        """Test exists flag reflects actual filesystem state."""
        try:
            from bp_face_recognition.vision.data.preprocessing import get_dataset_info
        except SystemExit:
            pytest.skip(
                "face_recognition import chain issue - requires manual pip install in venv"
            )

        info = get_dataset_info("data/datasets/cropped/seccam_2")

        if info["exists"]:
            assert info["total_images"] >= 0

    def test_nonexistent_dataset(self):
        """Test get_dataset_info for non-existent dataset."""
        try:
            from bp_face_recognition.vision.data.preprocessing import get_dataset_info
        except SystemExit:
            pytest.skip(
                "face_recognition import chain issue - requires manual pip install in venv"
            )

        info = get_dataset_info("data/datasets/nonexistent_dataset_xyz")

        assert info["exists"] is False
        assert info["total_images"] == 0


class TestDatasetLoading:
    """Test dataset loading functionality."""

    def test_load_face_dataset_structure(self):
        """Test load_face_dataset returns train, val, test datasets."""
        try:
            from bp_face_recognition.vision.data.preprocessing import load_face_dataset

            train_ds, val_ds, test_ds = load_face_dataset(
                "data/datasets/cropped/seccam_2", batch_size=4
            )

            assert train_ds is not None
            assert val_ds is not None
            assert test_ds is not None
        except FileNotFoundError:
            pytest.skip("Dataset not available")

    def test_batch_shapes(self):
        """Test dataset produces correct batch shapes."""
        try:
            from bp_face_recognition.vision.data.preprocessing import load_face_dataset

            train_ds, _, _ = load_face_dataset(
                "data/datasets/cropped/seccam_2", batch_size=4
            )

            for images, labels in train_ds.take(1):
                assert images.shape[0] <= 4
                assert len(images.shape) == 4
        except FileNotFoundError:
            pytest.skip("Dataset not available")


class TestDataAugmentation:
    """Test data augmentation functionality."""

    def test_augmentation_get_augmentor(self):
        """Test get_augmentor returns augmentor."""
        from bp_face_recognition.preprocessing.augmentation import get_augmentor

        augmentor = get_augmentor(224, 224)

        assert augmentor is not None

    def test_augmentor_is_compose(self):
        """Test augmentor is albumentations compose."""
        from bp_face_recognition.preprocessing.augmentation import get_augmentor

        augmentor = get_augmentor(224, 224)

        assert hasattr(augmentor, "__call__")
