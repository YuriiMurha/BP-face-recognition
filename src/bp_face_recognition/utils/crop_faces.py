import cv2  # pip install opencv-python
import os
import numpy as np
import json
from tqdm import tqdm

# --- Configuration for Cropping ---
from bp_face_recognition.config.settings import settings

INPUT_IMAGES_DIR = settings.AUGMENTED_DIR
OUTPUT_CROPPED_FACES_DIR = settings.CROPPED_DIR

# Dataset types and dataset sources
DATASET_TYPES = ["train", "test", "val"]
DATASET_SOURCES = ["webcam", "seccam", "seccam_2"]

# Define image sizes for each dataset source
IMAGE_SIZES = {
    "webcam": (480, 640),
    "seccam": (1280, 800),
    "seccam_2": (1280, 800)
}

IMG_HEIGHT = 224  # Target size for cropped faces
IMG_WIDTH = 224

# Function to generate dataset paths dynamically
def get_dataset_paths(dataset_source, dataset_type):
    base_path = os.path.join(INPUT_IMAGES_DIR, dataset_source, dataset_type)
    return {
        "images": os.path.join(base_path, "images"),
        "labels": os.path.join(base_path, "labels")
    }

def load_labels(label_path):
    """
    Load labels from a JSON file and extract bounding boxes and class IDs.
    """
    with open(label_path, 'r', encoding="utf-8") as f:
        label_data = json.load(f)
    shapes = label_data.get("shapes", [])
    
    labels = []
    bboxes = []
    for shape in shapes:
        label = shape["label"]  # Keep the label as-is (string or int)
        labels.append(label)
        
        points = np.array(shape["points"], dtype=np.float32)
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        bboxes.append([x1, y1, x2, y2])
    
    return labels, bboxes

processed_faces_count = 0

# Process each dataset source and type
for dataset_source in tqdm(DATASET_SOURCES, desc="Processing Sources"):
    for dataset_type in tqdm(DATASET_TYPES, desc=f"Subsets for {dataset_source}", leave=False):
        paths = get_dataset_paths(dataset_source, dataset_type)
        image_dir = paths["images"]
        label_dir = paths["labels"]

        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            continue

        # Get all image and label file paths
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        label_files = {os.path.splitext(f)[0]: os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.lower().endswith('.json')}

        for img_path in tqdm(image_files, desc=f"Cropping {dataset_source}/{dataset_type}", unit="img", leave=False):
            try:
                # Load image
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    continue

                # Find matching label
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                if img_name not in label_files:
                    continue
                label_path = label_files[img_name]

                # Load labels
                labels, bboxes = load_labels(label_path)

                for label, bbox in zip(labels, bboxes):
                    xmin, ymin, xmax, ymax = map(int, bbox)

                    # Ensure coordinates are within image bounds
                    xmin = max(0, int(bbox[0]))
                    ymin = max(0, int(bbox[1]))
                    xmax = min(img_bgr.shape[1], int(bbox[2]))
                    ymax = min(img_bgr.shape[0], int(bbox[3]))

                    # Check if the bounding box is valid (non-zero width and height)
                    if xmax <= xmin or ymax <= ymin:
                        print(f"Warning: Invalid bounding box {bbox} for image {img_path}. Skipping.")
                        continue

                    # # Add a small margin around the face (optional)
                    # margin_x = int(0.1 * (xmax - xmin))
                    # margin_y = int(0.1 * (ymax - ymin))

                    # # Dynamically adjust coordinates with margins while staying within bounds
                    # xmin = max(0, xmin - margin_x)
                    # ymin = max(0, ymin - margin_y)
                    # xmax = min(img_bgr.shape[1], xmax + margin_x)
                    # ymax = min(img_bgr.shape[0], ymax + margin_y)

                    # # Check again after adding margins
                    # if xmax <= xmin or ymax <= ymin:
                    #     print(f"Warning: Invalid bounding box after adding margins {bbox} for image {img_path}. Skipping.")
                    #     continue

                    # Crop the face
                    cropped_face = img_bgr[ymin:ymax, xmin:xmax]

                    # Check if the cropped face is valid
                    if cropped_face.size == 0 or cropped_face.shape[0] < 1 or cropped_face.shape[1] < 1:
                        print(f"Warning: Empty or invalid crop from {img_path} for bbox {bbox}. Skipping.")
                        continue

                    # Resize cropped face to the target size
                    cropped_face_resized = cv2.resize(cropped_face, (IMG_WIDTH, IMG_HEIGHT))

                    # Create output directories for images and labels
                    output_dir = os.path.join(OUTPUT_CROPPED_FACES_DIR, dataset_source, dataset_type)
                    os.makedirs(output_dir, exist_ok=True)

                    # Save the cropped face using the original image name
                    cropped_image_name = f"{os.path.splitext(os.path.basename(img_path))[0]}.{label}.jpg"
                    output_filepath = os.path.join(output_dir, cropped_image_name)
                    cv2.imwrite(output_filepath, cropped_face_resized)
                    
                    processed_faces_count += 1

            except Exception as e:
                print(f"Error processing {img_path} with label {label_path}: {e}. Skipping.")

print(f"Dataset preparation complete. Total cropped faces saved: {processed_faces_count}")