"""
Test script to validate the preprocessing module functionality.
"""

import sys
import os

sys.path.insert(0, "src")

from bp_face_recognition.vision.data.preprocessing import (
    load_face_dataset,
    get_dataset_info,
    load_cropped_seccam2_dataset,
)


def test_dataset_info():
    """Test dataset info functionality."""
    print("=== Testing Dataset Info ===")

    # Test seccam_2 dataset
    info = get_dataset_info("data/datasets/cropped/seccam_2")
    print(f"Dataset path: {info['path']}")
    print(f"Exists: {info['exists']}")
    print(f"Splits: {info['splits']}")
    print(f"Total images: {info['total_images']}")

    return info


def test_dataset_loading():
    """Test dataset loading functionality."""
    print("\n=== Testing Dataset Loading ===")

    # Test with seccam_2 dataset
    try:
        train_ds, val_ds, test_ds = load_face_dataset(
            "data/datasets/cropped/seccam_2", batch_size=4
        )
        print("✅ Dataset loaded successfully!")

        # Test a single batch from train dataset
        for images, labels in train_ds.take(1):
            print(f"Train batch - Images shape: {images.shape}")
            print(f"Train batch - Labels shape: {labels.shape}")
            print(f"Train batch - Sample labels: {labels[:2].numpy()}")
            break

        # Test validation dataset
        for images, labels in val_ds.take(1):
            print(f"Val batch - Images shape: {images.shape}")
            print(f"Val batch - Labels shape: {labels.shape}")
            break

        # Test test dataset
        for images, labels in test_ds.take(1):
            print(f"Test batch - Images shape: {images.shape}")
            print(f"Test batch - Labels shape: {labels.shape}")
            break

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

    return True


def test_convenience_function():
    """Test the convenience function."""
    print("\n=== Testing Convenience Function ===")

    try:
        train_ds, val_ds, test_ds = load_cropped_seccam2_dataset(batch_size=2)
        print("✅ Convenience function worked!")

        for images, labels in train_ds.take(1):
            print(f"Convenience batch - Images shape: {images.shape}")
            print(f"Convenience batch - Labels shape: {labels.shape}")
            break

    except Exception as e:
        print(f"❌ Error with convenience function: {e}")
        return False

    return True


if __name__ == "__main__":
    print("🧪 Testing BP Face Recognition Preprocessing Module\n")

    # Run tests
    info = test_dataset_info()
    loading_success = test_dataset_loading()
    convenience_success = test_convenience_function()

    # Summary
    print("\n=== Test Summary ===")
    print(f"Dataset info: ✅")
    print(f"Dataset loading: {'✅' if loading_success else '❌'}")
    print(f"Convenience function: {'✅' if convenience_success else '❌'}")

    if loading_success and convenience_success:
        print("\n🎉 All tests passed! The preprocessing module is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
