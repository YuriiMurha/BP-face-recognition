"""
Split LFW dataset into train/val/test subsets.

Input:  raw/lfw/{identity_name}/ (34 identities, already cropped)
Output: cropped/lfw/{train,val,test}/{identity}_{index}.jpg (FLAT - no identity subfolders)

Split ratio: 65% train, 20% val, 15% test (per identity)

Usage:
    python split_lfw.py
"""

import os
import random
import shutil
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
RAW_LFW_DIR = DATASETS_DIR / "raw" / "lfw"
CROPPED_LFW_DIR = DATASETS_DIR / "cropped" / "lfw"

# Split ratios
TRAIN_RATIO = 0.65
VAL_RATIO = 0.20
TEST_RATIO = 0.15

# Set random seed for reproducibility
random.seed(42)


def split_lfw():
    """Split LFW dataset into train/val/test (FLAT structure)."""
    print("\n" + "=" * 60)
    print("Splitting LFW Dataset")
    print("=" * 60)
    print(
        f"Split ratios: train={TRAIN_RATIO*100:.0f}%, val={VAL_RATIO*100:.0f}%, test={TEST_RATIO*100:.0f}%"
    )

    if not RAW_LFW_DIR.exists():
        print(f"Error: LFW directory not found at {RAW_LFW_DIR}")
        return

    # Get all identity folders
    identities = [d for d in os.listdir(RAW_LFW_DIR) if os.path.isdir(RAW_LFW_DIR / d)]

    print(f"Found {len(identities)} identities")

    # Create flat output directories
    train_dir = CROPPED_LFW_DIR / "train"
    val_dir = CROPPED_LFW_DIR / "val"
    test_dir = CROPPED_LFW_DIR / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    total_train = 0
    total_val = 0
    total_test = 0

    for identity in sorted(identities):
        identity_dir = RAW_LFW_DIR / identity

        # Get all images for this identity
        image_files = [
            f
            for f in os.listdir(identity_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            print(f"  {identity}: no images found, skipping")
            continue

        # Shuffle
        random.shuffle(image_files)

        # Calculate split indices
        n = len(image_files)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_files = image_files[:n_train]
        val_files = image_files[n_train : n_train + n_val]
        test_files = image_files[n_train + n_val :]

        # Copy files with new names: {identity}_{index}.jpg (FLAT)
        # Global counter per identity to avoid collisions
        train_idx = 0
        for filename in train_files:
            src = identity_dir / filename
            dst = train_dir / f"{identity}_{train_idx:04d}.jpg"
            shutil.copy2(src, dst)
            train_idx += 1

        val_idx = 0
        for filename in val_files:
            src = identity_dir / filename
            dst = val_dir / f"{identity}_{val_idx:04d}.jpg"
            shutil.copy2(src, dst)
            val_idx += 1

        test_idx = 0
        for filename in test_files:
            src = identity_dir / filename
            dst = test_dir / f"{identity}_{test_idx:04d}.jpg"
            shutil.copy2(src, dst)
            test_idx += 1

        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)

        print(
            f"  {identity}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test"
        )

    print(f"\nTotal: {total_train} train, {total_val} val, {total_test} test")
    print(f"Output: {CROPPED_LFW_DIR}")
    print("\nDone!")


def main():
    print("LFW Dataset Splitter")
    print(f"Input:  {RAW_LFW_DIR}")
    print(f"Output: {CROPPED_LFW_DIR} (flat structure)")

    split_lfw()


if __name__ == "__main__":
    main()
