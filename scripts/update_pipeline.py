import sys
import os
import subprocess
from pathlib import Path

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))
from bp_face_recognition.config.settings import settings


def run_script(script_path, description):
    print(f"--- Running {description} ---")
    # Use uv run to ensure environment consistency
    result = subprocess.run(
        ["uv", "run", "python", str(script_path)], capture_output=False
    )
    if result.returncode != 0:
        print(f"Error during {description}. Exiting.")
        sys.exit(1)
    print(f"--- {description} Completed Successfully ---\n")


def main():
    print("🚀 Starting Data Pipeline Update\n")

    # Define paths to modular scripts
    augmentation_script = (
        settings.SRC_DIR / "bp_face_recognition" / "data" / "augmentation.py"
    )
    crop_script = settings.SRC_DIR / "bp_face_recognition" / "utils" / "crop_faces.py"

    # 1. Augmentation (Creates variations of training data)
    # This script reads from data/datasets/raw and writes to data/datasets/augmented
    run_script(augmentation_script, "Data Augmentation")

    # 2. Cropping (Extracts 224x224 faces)
    # This script reads from data/datasets/augmented and writes to data/datasets/cropped
    run_script(crop_script, "Face Cropping")

    print(
        "✅ Pipeline update finished. You can now run 'make train' or 'run_training' tool."
    )


if __name__ == "__main__":
    main()
