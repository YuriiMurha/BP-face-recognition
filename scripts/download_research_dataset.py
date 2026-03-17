import os
import shutil
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np


def prepare_research_dataset(
    min_faces_per_person=30,
    output_dir="data/datasets/research/triplet_gallery",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_state=42,
):
    """
    Download and prepare LFW dataset with flat structure and train/val/test splits.

    Args:
        min_faces_per_person: Minimum number of faces per person to include
        output_dir: Output directory for the dataset
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_state: Random seed for reproducible splits
    """
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1.0"

    print(
        f"[DATASET] Fetching LFW dataset (min {min_faces_per_person} faces per person)..."
    )

    # Fetch data
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, color=True)

    n_samples, h, w, c = lfw_people.images.shape
    target_names = lfw_people.target_names
    targets = lfw_people.target

    print(f"[DATASET] Found {len(target_names)} identities meeting criteria.")
    print(f"[DATASET] Total images: {n_samples}")
    print(f"[DATASET] Image size: {h}x{w}x{c}")

    # Create output directories
    splits = ["train", "val", "test"]
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

    # Split data by identity to ensure all splits have all identities
    # This is important for few-shot learning / metric learning
    identity_to_indices = {}
    for idx, target in enumerate(targets):
        identity = target_names[target]
        if identity not in identity_to_indices:
            identity_to_indices[identity] = []
        identity_to_indices[identity].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    for identity, indices in identity_to_indices.items():
        n_identity = len(indices)

        # Ensure we have at least 3 images per identity for splitting
        if n_identity < 3:
            print(
                f"[WARN] Identity '{identity}' has only {n_identity} images, skipping split"
            )
            # Put all in train
            train_indices.extend(indices)
            continue

        # Split this identity's images
        # First split: separate test set
        remaining_idx, test_idx = train_test_split(
            indices, test_size=test_ratio, random_state=random_state
        )

        # Second split: separate train and val from remaining
        if len(remaining_idx) > 0:
            val_size = val_ratio / (train_ratio + val_ratio)
            train_idx, val_idx = train_test_split(
                remaining_idx, test_size=val_size, random_state=random_state
            )
        else:
            train_idx, val_idx = [], []

        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)

    print(
        f"[SPLIT] Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
    )

    # Save images to respective directories (flat structure)
    split_map = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }

    for split_name, indices in split_map.items():
        split_dir = os.path.join(output_dir, split_name)
        print(f"[SAVING] Saving {len(indices)} images to {split_name}/")

        for i in indices:
            person_name = target_names[targets[i]].replace(" ", "_")

            # Convert image data
            img_data = lfw_people.images[i]
            if img_data.max() <= 1.0:
                img_data = (img_data * 255).astype(np.uint8)
            else:
                img_data = img_data.astype(np.uint8)

            img = Image.fromarray(img_data)

            # Flat structure: {identity}_{index:04d}.jpg directly in split folder
            filename = f"{person_name}_{i:04d}.jpg"
            img.save(os.path.join(split_dir, filename))

    print(f"[DATASET] Dataset prepared at: {output_dir}")
    print(f"[DATASET] Structure:")
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        n_images = len([f for f in os.listdir(split_dir) if f.endswith(".jpg")])
        print(f"  {split}/: {n_images} images")

    return output_dir


if __name__ == "__main__":
    prepare_research_dataset()
