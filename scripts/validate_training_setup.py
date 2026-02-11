#!/usr/bin/env python3
"""
Quick Training Validation Script
Tests training pipeline with minimal epochs to verify everything works
"""

import subprocess
import sys
import time
from pathlib import Path
from bp_face_recognition.config.settings import settings


def quick_validation_test():
    """
    Run a quick 2-epoch validation test for both backbones
    """
    print("Quick Training Validation Test")
    print("This will run 2 epochs for each backbone to verify setup")

    backbones = ["EfficientNetB0", "MobileNetV3Small"]

    for backbone in backbones:
        print("\n" + "=" * 50)
        print(f"Testing: {backbone}")
        print("=" * 50)

        cmd = [
            sys.executable,
            "experiments/train.py",
            "--dataset",
            "seccam_2",
            "--epochs",
            "2",  # Quick test
            "--batch_size",
            "16",  # Smaller batch for validation
            "--backbone",
            backbone,
            "--fine-tune",  # Test fine-tuning too
        ]

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd, cwd=settings.ROOT_DIR, capture_output=True, text=True
            )

            execution_time = time.time() - start_time

            if result.returncode == 0:
                print(f"SUCCESS: {backbone} validation successful!")
                print(f"Time: {execution_time:.2f}s")

                # Check for key success indicators
                if "Training complete" in result.stdout:
                    print("Training pipeline working")
                if "Model saved to:" in result.stdout:
                    print("Model saving working")

            else:
                print(f"FAILED: {backbone} validation failed!")
                print(f"Return code: {result.returncode}")
                if result.stderr:
                    print(f"Error preview: {result.stderr[:500]}...")
                return False

        except Exception as e:
            print(f"EXCEPTION: {backbone} validation exception: {e}")
            return False

    print("\n" + "=" * 50)
    print("All validation tests passed!")
    print("Ready for full training matrix")
    print("=" * 50)

    return True


if __name__ == "__main__":
    import os

    os.chdir(settings.ROOT_DIR)

    success = quick_validation_test()

    if success:
        print("\nReady to launch full training matrix:")
        print("   make train-comparison")
        print("   or")
        print("   python scripts/run_training_comparison.py")
    else:
        print("\nFix validation issues before proceeding")
