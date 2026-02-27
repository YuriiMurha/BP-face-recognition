import pytest
import numpy as np
import tempfile
import os
import json
import subprocess
import sys
from pathlib import Path


def pytest_configure(config):
    """Ensure setuptools is installed before any tests run."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "setuptools"], capture_output=True
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def dummy_image():
    """Generate a dummy image array."""
    return np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)


@pytest.fixture
def dummy_batch_images():
    """Generate a batch of dummy images."""
    return np.random.randint(0, 255, (4, 160, 160, 3), dtype=np.uint8)


@pytest.fixture
def mock_model_path(temp_dir):
    """Create a temporary model file path."""
    model_path = temp_dir / "test_model.keras"
    return str(model_path)


@pytest.fixture
def mock_config_path(temp_dir):
    """Create a temporary config file path."""
    config_path = temp_dir / "test_config.yaml"
    return str(config_path)


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return {
        "person_001": ["image_001.jpg", "image_002.jpg"],
        "person_002": ["image_003.jpg", "image_004.jpg"],
    }


@pytest.fixture
def temp_dataset_dir(temp_dir, sample_labels):
    """Create a temporary dataset directory structure."""
    dataset_dir = temp_dir / "dataset"
    dataset_dir.mkdir()

    for person, images in sample_labels.items():
        person_dir = dataset_dir / person
        person_dir.mkdir()
        for img in images:
            img_path = person_dir / img
            dummy_img = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
            from PIL import Image

            Image.fromarray(dummy_img).save(img_path)

    return str(dataset_dir)
