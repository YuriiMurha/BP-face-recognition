import os
import argparse
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))
from bp_face_recognition.config.settings import settings


def init_dataset(name, source_dir=None, train_split=0.7, test_split=0.15):
    """
    Scaffolds a new dataset and optionally populates it with images from a source directory.
    """
    # Root for raw data (where we put things before labeling)
    # Based on PROGRESS.md, the structure is data/datasets/[name]
    dataset_path = settings.DATASETS_DIR / name

    subsets = ["train", "test", "val"]
    print(f"📁 Initializing dataset structure at: {dataset_path}")

    for subset in subsets:
        (dataset_path / subset / "images").mkdir(parents=True, exist_ok=True)
        (dataset_path / subset / "labels").mkdir(parents=True, exist_ok=True)

    if source_dir:
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"❌ Error: Source directory {source_dir} does not exist.")
            return

        images = [
            f
            for f in os.listdir(source_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not images:
            print(f"⚠️ Warning: No images found in {source_dir}")
            return

        print(f"🔍 Found {len(images)} images. Splitting into train/test/val...")
        np.random.shuffle(images)

        total = len(images)
        train_end = int(total * train_split)
        test_end = train_end + int(total * test_split)

        splits = {
            "train": images[:train_end],
            "test": images[train_end:test_end],
            "val": images[test_end:],
        }

        for subset, files in splits.items():
            for f in tqdm(files, desc=f"Moving images to {subset}", unit="file"):
                shutil.copy(source_path / f, dataset_path / subset / "images" / f)

        print(f"✨ Successfully populated {name} with {len(images)} images.")
    else:
        print(
            f"✅ Empty dataset structure created for {name}. Ready for manual data addition."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initialize a new dataset for labeling"
    )
    parser.add_argument("name", type=str, help="Name of the new dataset")
    parser.add_argument(
        "--source",
        type=str,
        help="Source directory of raw images to split",
        default=None,
    )
    parser.add_argument("--train", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--test", type=float, default=0.15, help="Test split ratio")

    args = parser.parse_args()
    init_dataset(args.name, args.source, args.train, args.test)
