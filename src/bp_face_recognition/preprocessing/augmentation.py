"""
Augmentation pipeline for cropped face images.

Input:  cropped/{dataset}/{train,val,test}/{label}_{original}.jpg
Output: augmented/{dataset}/{train,val,test}/{label}_{original}.{N}.jpg

Universal flat structure for all datasets.

Usage:
    python augmentation.py --dataset all
    python augmentation.py --dataset webcam
"""

import os
import cv2
import argparse
import numpy as np
import albumentations as alb
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
CROPPED_DIR = DATASETS_DIR / "cropped"
AUGMENTED_DIR = DATASETS_DIR / "augmented"

# Image size for augmentation
TARGET_SIZE = (224, 224)


# Dynamically discover datasets from cropped folder
def get_available_datasets():
    """Discover available datasets from cropped folder."""
    if not CROPPED_DIR.exists():
        return []
    return [d.name for d in CROPPED_DIR.iterdir() if d.is_dir()]


DATASETS = get_available_datasets()
SUBSETS = ["train", "val", "test"]

# Number of augmentations per image
DEFAULT_AUGMENTATIONS = 60


def get_augmentor(height, width):
    """
    Creates an albumentations augmentation pipeline for cropped faces.
    """
    return alb.Compose(
        [
            alb.Resize(height=height, width=width),
            alb.HorizontalFlip(p=0.5),
            alb.RandomBrightnessContrast(p=0.3),
            alb.RandomGamma(p=0.3),
            alb.RGBShift(p=0.2),
            alb.GaussNoise(p=0.2),
            alb.Blur(blur_limit=3, p=0.2),
        ]
    )


def augment_dataset(dataset_name, num_augmentations=DEFAULT_AUGMENTATIONS):
    """
    Applies augmentation to a dataset.
    Universal flat structure: input and output are flat (no subfolders for images/labels).
    """
    height, width = TARGET_SIZE
    augmentor = get_augmentor(height, width)

    source_dir = CROPPED_DIR / dataset_name
    target_dir = AUGMENTED_DIR / dataset_name

    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        return

    # Delete existing augmented dataset if exists
    if target_dir.exists():
        import shutil

        print(f"  Removing existing augmented dataset: {target_dir}")
        shutil.rmtree(target_dir)
        print(f"  Existing augmented dataset removed.")

    # Create fresh output directory
    target_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_augmented = 0

    for subset in SUBSETS:
        source_subset_dir = source_dir / subset
        target_subset_dir = target_dir / subset

        if not source_subset_dir.exists():
            print(f"  {subset}: source directory not found, skipping")
            continue

        # Get all image files (flat structure - no subdirs)
        image_files = [
            f
            for f in os.listdir(source_subset_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            print(f"  {subset}: no images found, skipping")
            continue

        target_subset_dir.mkdir(parents=True, exist_ok=True)

        subset_images = len(image_files)
        subset_augmented = 0

        print(f"Augmenting {dataset_name} - {subset} ({subset_images} images)...")

        for idx, image_file in enumerate(image_files):
            img_path = source_subset_dir / image_file
            img = cv2.imread(str(img_path))

            if img is None:
                continue

            # Get base name without extension
            base_name = Path(image_file).stem

            # Apply augmentations
            for aug_idx in range(num_augmentations):
                try:
                    augmented = augmentor(image=img)
                    aug_image = augmented["image"]

                    # Output filename: {base_name}.{N}.jpg
                    # Example: Yurii_f3d9f09a-8335.0.jpg, Yurii_f3d9f09a-8335.1.jpg
                    output_filename = f"{base_name}.{aug_idx}.jpg"
                    output_path = target_subset_dir / output_filename

                    cv2.imwrite(str(output_path), aug_image)
                    subset_augmented += 1
                    total_augmented += 1

                except Exception as e:
                    print(f"    Error augmenting {image_file}: {e}")

            total_images += 1

            # Progress update every 100 images
            if (idx + 1) % 100 == 0:
                print(
                    f"  Processed {idx + 1}/{subset_images} images, generated {subset_augmented} augmented"
                )

        print(
            f"  {subset}: {subset_images} images -> {subset_augmented} augmented images"
        )

    print(f"\nTotal: {total_images} images -> {total_augmented} augmented images")
    print(f"Output: {target_dir}")


def main():
    parser = argparse.ArgumentParser(description="Augment cropped face datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help=f"Dataset to augment (or 'all'): {', '.join(DATASETS) if DATASETS else 'none found'}",
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=DEFAULT_AUGMENTATIONS,
        help=f"Number of augmentations per image (default: {DEFAULT_AUGMENTATIONS})",
    )

    args = parser.parse_args()

    # Handle "all" parameter
    datasets_to_process = DATASETS if args.dataset == "all" else [args.dataset]

    if not datasets_to_process:
        print("No datasets found to augment!")
        return

    print("=" * 60)
    print("Face Augmentation Pipeline")
    print("=" * 60)
    print(f"Datasets to process: {datasets_to_process}")
    print(f"Augmentations per image: {args.num_augmentations}")

    for dataset in datasets_to_process:
        print(f"\n{'=' * 60}")
        print(f"Processing dataset: {dataset}")
        print(f"Input:  cropped/{dataset}/")
        print(f"Output: augmented/{dataset}/")

        augment_dataset(dataset, args.num_augmentations)

    print("\nAll datasets augmented!")


if __name__ == "__main__":
    main()
