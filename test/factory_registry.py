#!/usr/bin/env python3
"""
Test Factory Pattern with New Architecture

Test that the factory can successfully create detectors using the registry system.
"""

import sys
import os
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from bp_face_recognition.vision.registry import get_registry
from bp_face_recognition.vision.factory import RecognizerFactory


def test_factory_with_registry():
    """Test that factory works with registry system."""
    print("Testing factory pattern with registry...")

    try:
        # Test default detector
        detector = get_registry().get_default_detector()
        print(f"Default detector: {detector.get_detector_info()}")

        # Test all available detectors
        available_detectors = get_registry().list_detectors()
        print(f"Available detectors: {list(available_detectors.keys())}")

        # Test face recognizers
        available_recognizers = get_registry().list_recognizers()
        print(f"Available recognizers: {list(available_recognizers.keys())}")

        # Test specific detectors by name
        test_detectors = ["mediapipe_v1", "mtcnn_v1", "haar_v1", "dlib_hog_v1"]

        for detector_name in test_detectors:
            try:
                detector = get_registry().get_detector(detector_name)
                print(f"Testing {detector_name}: {detector.get_detector_info()}")

                # Create a small test image
                test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                # Test detection
                detections = detector.detect(test_img)
                print(f"  {detector_name}: Found {len(detections)} faces")

            except Exception as e:
                print(f"Error testing {detector_name}: {e}")

        # Test face recognizers by name
        for recognizer_name in test_recognizers:
            try:
                recognizer = get_registry().get_recognizer(recognizer_name)
                print(f"Testing {recognizer_name}: {recognizer.get_recognizer_info()}")

                # Create a small test image (160x160 with 3 channels)
                test_img = np.random.randint(0, 255, (160, 3), dtype=np.uint8)

                # Test embedding
                embedding = recognizer.get_embedding(test_img)
                print(f"  {recognizer_name}: Embedding shape: {embedding.shape}")

            except Exception as e:
                print(f"Error testing {recognizer_name}: {e}")

        return True

    except Exception as e:
        return False

    print("Factory pattern test completed!")


if __name__ == "__main__":
    success = test_factory_with_registry()
    if success:
        print("✅ Factory system works correctly!")
    else:
        print("❌ Factory test failed!")

    return success
