"""
Model Switcher Utility

Switches between FaceNet models by updating the configuration.
Also handles database backup and management.

Usage:
    python switch_model.py pu      # Switch to FaceNet PU
    python switch_model.py tl      # Switch to FaceNet TL
    python switch_model.py tloss   # Switch to FaceNet TLoss
    python switch_model.py clear   # Clear database
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml


def backup_database():
    """Backup current database with timestamp."""
    db_path = Path("data/faces.csv")
    if not db_path.exists():
        print("  No existing database to backup")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(f"data/faces_backup_{timestamp}.csv")

    shutil.copy(db_path, backup_path)
    print(f"  [OK] Database backed up to: {backup_path}")
    return backup_path


def clear_database():
    """Clear the face database."""
    db_path = Path("data/faces.csv")

    if db_path.exists():
        backup_database()
        db_path.unlink()
        print("  [OK] Database cleared")
    else:
        print("  [INFO] No database to clear")

    # Create empty database
    import pandas as pd

    pd.DataFrame({"id": [], "name": [], "embedding": []}).to_csv(db_path, index=False)
    print("  [OK] Fresh database created")


def update_config(model_name):
    """Update config/models.yaml to use specified model."""
    config_path = Path("config/models.yaml")

    if not config_path.exists():
        print(f"  [ERROR] Config file not found: {config_path}")
        return False

    with open(config_path, "r") as f:
        content = f.read()

    # Update the default_recognizer line
    import re

    pattern = r'(default_recognizer:\s*")([^"]+)(")'

    if re.search(pattern, content):
        new_content = re.sub(pattern, f"\\g<1>{model_name}\\g<3>", content)

        with open(config_path, "w") as f:
            f.write(new_content)

        print(f"  [OK] Config updated: default_recognizer = {model_name}")
        return True
    else:
        print(f"  [ERROR] Could not find default_recognizer in config")
        return False


def switch_model(model_name):
    """Switch to specified model."""
    models = {
        "pu": {
            "name": "FaceNet PU (Progressive Unfreezing)",
            "accuracy": 99.15,
            "file": "facenet_progressive_v1.0.keras",
            "config_name": "facenet_pu",
        },
        "tl": {
            "name": "FaceNet TL (Transfer Learning)",
            "accuracy": 92.84,
            "file": "facenet_transfer_v1.0.keras",
            "config_name": "facenet_tl",
        },
        "tloss": {
            "name": "FaceNet TLoss (Triplet Loss)",
            "accuracy": 94.63,
            "file": "facenet_triplet_best.keras",
            "config_name": "facenet_tloss",
        },
    }

    if model_name not in models:
        print(f"[ERROR] Unknown model: {model_name}")
        print(f"  Available: {', '.join(models.keys())}")
        return False

    model_info = models[model_name]

    print("=" * 60)
    print(f"Switching to: {model_info['name']}")
    print("=" * 60)
    print(f"  Expected accuracy: {model_info['accuracy']}%")
    print(f"  Model file: {model_info['file']}")
    print()

    # Backup database
    print("Step 1: Backup current database")
    backup_database()
    print()

    # Update config
    print("Step 2: Update configuration")
    if not update_config(model_info["config_name"]):
        return False
    print()

    print("=" * 60)
    print("[OK] Model switch complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Clear database: python switch_model.py clear")
    print("  2. Run app: make run")
    print("  3. Register your face and test")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Switch between FaceNet models for testing"
    )
    parser.add_argument(
        "command",
        choices=["pu", "tl", "tloss", "clear"],
        help="Model to switch to (pu/tl/tloss) or clear database",
    )

    args = parser.parse_args()

    if args.command == "clear":
        print("=" * 60)
        print("Clearing Database")
        print("=" * 60)
        clear_database()
        print()
        print("[OK] Database cleared and ready for new registrations")
    else:
        success = switch_model(args.command)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
