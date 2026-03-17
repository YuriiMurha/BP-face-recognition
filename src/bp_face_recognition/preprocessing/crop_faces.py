"""
Crop faces from raw dataset images using bounding boxes from JSON labels.

Input:  raw/{dataset}/{train,val,test}/images/ + labels/
Output: cropped/{dataset}/{train,val,test}/{identity}_{index}.jpg

Usage:
    python crop_faces.py --dataset webcam
    python crop_faces.py --dataset seccam
    python crop_faces.py --dataset seccam_2
"""

import os
import cv2
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
CROPPED_DIR = DATASETS_DIR / "cropped"

# Datasets to exclude (lfw doesn't need cropping - already cropped)
EXCLUDE_DATASETS = {"lfw"}


# Dynamically discover datasets from raw folder
def get_available_datasets():
    """Discover available datasets from raw folder, excluding lfw."""
    if not RAW_DIR.exists():
        return []
    return [
        d.name
        for d in RAW_DIR.iterdir()
        if d.is_dir() and d.name not in EXCLUDE_DATASETS
    ]


# Default image sizes (can be overridden by dataset)
DEFAULT_IMAGE_SIZE = (800, 1280)

DATASETS = get_available_datasets()
SUBSETS = ["train", "val", "test"]


def crop_faces_from_dataset(dataset_name: str):
    """Crop faces from all images in a dataset."""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")

    source_dir = RAW_DIR / dataset_name
    target_dir = CROPPED_DIR / dataset_name

    total_images = 0
    total_faces = 0

    for subset in SUBSETS:
        image_dir = source_dir / subset / "images"
        label_dir = source_dir / subset / "labels"
        target_subset_dir = target_dir / subset

        if not image_dir.exists():
            print(f"  {subset}: source directory not found, skipping")
            continue

        target_subset_dir.mkdir(parents=True, exist_ok=True)

        # Count files
        image_files = [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        subset_images = 0
        subset_faces = 0

        for idx, image_file in enumerate(image_files):
            img_path = image_dir / image_file
            img = cv2.imread(str(img_path))

            if img is None:
                print(f"    Warning: Could not read {image_file}")
                continue

            # Get label file
            label_filename = Path(image_file).stem + ".json"
            label_path = label_dir / label_filename

            faces_cropped = 0

            if label_path.exists():
                try:
                    with open(label_path, "r") as f:
                        label_data = json.load(f)

                    for shape_idx, shape in enumerate(label_data.get("shapes", [])):
                        label = shape.get("label", "unknown")
                        points = shape.get("points", [])

                        if len(points) != 2:
                            continue

                        # Get bounding box
                        x_min = min(points[0][0], points[1][0])
                        x_max = max(points[0][0], points[1][0])
                        y_min = min(points[0][1], points[1][1])
                        y_max = max(points[0][1], points[1][1])

                        # Convert to integer
                        x_min, x_max = int(x_min), int(x_max)
                        y_min, y_max = int(y_min), int(y_max)

                        # Add margin (10%)
                        h, w = img.shape[:2]
                        margin_x = int((x_max - x_min) * 0.1)
                        margin_y = int((y_max - y_min) * 0.1)

                        x_min = max(0, x_min - margin_x)
                        x_max = min(w, x_max + margin_x)
                        y_min = max(0, y_min - margin_y)
                        y_max = min(h, y_max + margin_y)

                        # Crop face
                        face_crop = img[y_min:y_max, x_min:x_max]

                        if face_crop.size == 0:
                            continue

                        # Generate output filename: {label}_{original_name}.jpg
                        # Example: Yurii_f3d9f09a-8335-11ee-9a89-9cfeff47d2fa.jpg
                        output_filename = f"{label}_{Path(image_file).stem}.jpg"
                        output_path = target_subset_dir / output_filename

                        # Save cropped face
                        cv2.imwrite(str(output_path), face_crop)
                        faces_cropped += 1
                        total_faces += 1

                except Exception as e:
                    print(f"    Error processing {label_filename}: {e}")

            subset_images += 1
            subset_faces += faces_cropped

            # Progress update every 50 images
            if (idx + 1) % 50 == 0:
                print(
                    f"  {subset}: processed {idx + 1}/{len(image_files)} images, {subset_faces} faces"
                )

        total_images += subset_images
        print(f"  {subset}: {subset_images} images -> {subset_faces} faces")

    print(f"\nTotal: {total_images} images -> {total_faces} faces cropped")
    print(f"Output: {target_dir}")


def main():
    parser = argparse.ArgumentParser(description="Crop faces from raw datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help=f"Dataset to process (or 'all'): {', '.join(DATASETS) if DATASETS else 'none found'}",
    )

    args = parser.parse_args()

    # Handle "all" parameter
    datasets_to_process = DATASETS if args.dataset == "all" else [args.dataset]

    if not datasets_to_process:
        print("No datasets found to process!")
        return

    print("Face Cropping Pipeline")
    print(f"Datasets to process: {datasets_to_process}")

    for dataset in datasets_to_process:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset}")
        print(f"Input:  raw/{dataset}/")
        print(f"Output: cropped/{dataset}/")

        crop_faces_from_dataset(dataset)

    print("\nAll datasets processed!")


if __name__ == "__main__":
    main()
