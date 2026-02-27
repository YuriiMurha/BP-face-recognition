import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from bp_face_recognition.vision.data.preprocessing import (
    load_face_dataset,
    get_dataset_info,
    load_cropped_seccam2_dataset,
)


class TestDatasetInfo:
    """Test dataset info functionality."""

    def test_get_dataset_info_exists(self):
        """Test get_dataset_info returns proper structure."""
        info = get_dataset_info("data/datasets/cropped/seccam_2")

        assert isinstance(info, dict)
        assert "path" in info
        assert "exists" in info
        assert "splits" in info
        assert "total_images" in info

    def test_dataset_exists_flag(self):
        """Test exists flag reflects actual filesystem state."""
        info = get_dataset_info("data/datasets/cropped/seccam_2")

        if info["exists"]:
            assert info["total_images"] >= 0
            assert isinstance(info["splits"], (list, dict))


class TestDatasetLoading:
    """Test dataset loading functionality."""

    def test_load_face_dataset_structure(self):
        """Test load_face_dataset returns train, val, test datasets."""
        try:
            train_ds, val_ds, test_ds = load_face_dataset(
                "data/datasets/cropped/seccam_2", batch_size=4
            )

            assert train_ds is not None
            assert val_ds is not None
            assert test_ds is not None
        except FileNotFoundError:
            pytest.skip("Dataset not available")

    def test_train_dataset_batch(self):
        """Test train dataset produces correct batch shapes."""
        try:
            train_ds, _, _ = load_face_dataset(
                "data/datasets/cropped/seccam_2", batch_size=4
            )

            for images, labels in train_ds.take(1):
                assert images.shape[0] <= 4
                assert len(images.shape) == 4
                assert labels.shape[0] <= 4
        except FileNotFoundError:
            pytest.skip("Dataset not available")

    def test_val_dataset_batch(self):
        """Test validation dataset produces correct batch shapes."""
        try:
            _, val_ds, _ = load_face_dataset(
                "data/datasets/cropped/seccam_2", batch_size=4
            )

            for images, labels in val_ds.take(1):
                assert images.shape[0] <= 4
                assert len(images.shape) == 4
        except FileNotFoundError:
            pytest.skip("Dataset not available")

    def test_test_dataset_batch(self):
        """Test test dataset produces correct batch shapes."""
        try:
            _, _, test_ds = load_face_dataset(
                "data/datasets/cropped/seccam_2", batch_size=4
            )

            for images, labels in test_ds.take(1):
                assert images.shape[0] <= 4
                assert len(images.shape) == 4
        except FileNotFoundError:
            pytest.skip("Dataset not available")

    def test_batch_size_parameter(self):
        """Test different batch sizes work correctly."""
        try:
            for batch_size in [2, 4, 8]:
                train_ds, _, _ = load_face_dataset(
                    "data/datasets/cropped/seccam_2", batch_size=batch_size
                )

                for images, _ in train_ds.take(1):
                    assert images.shape[0] <= batch_size
        except FileNotFoundError:
            pytest.skip("Dataset not available")


class TestConvenienceFunction:
    """Test convenience function for loading seccam2 dataset."""

    def test_load_cropped_seccam2_dataset(self):
        """Test convenience function loads dataset correctly."""
        try:
            train_ds, val_ds, test_ds = load_cropped_seccam2_dataset(batch_size=2)

            assert train_ds is not None
            assert val_ds is not None
            assert test_ds is not None
        except FileNotFoundError:
            pytest.skip("Dataset not available")

    def test_convenience_batch_shape(self):
        """Test convenience function produces correct batch shapes."""
        try:
            train_ds, _, _ = load_cropped_seccam2_dataset(batch_size=2)

            for images, labels in train_ds.take(1):
                assert images.shape[0] <= 2
                assert len(images.shape) == 4
                assert labels.shape[0] <= 2
        except FileNotFoundError:
            pytest.skip("Dataset not available")


class TestDatasetNotFound:
    """Test error handling for missing datasets."""

    def test_nonexistent_dataset(self):
        """Test loading non-existent dataset raises error."""
        with pytest.raises(FileNotFoundError):
            load_face_dataset("data/datasets/nonexistent_dataset_xyz", batch_size=4)

    def test_info_nonexistent_dataset(self):
        """Test get_dataset_info for non-existent dataset."""
        info = get_dataset_info("data/datasets/nonexistent_dataset_xyz")

        assert info["exists"] is False
        assert info["total_images"] == 0
