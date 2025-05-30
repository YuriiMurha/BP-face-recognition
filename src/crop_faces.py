import cv2  # pip install opencv-python
import os
import numpy as np
import json

# --- Configuration for Cropping ---
INPUT_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/datasets/augmented')
OUTPUT_CROPPED_FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/datasets/cropped_faces')

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
for dataset_source in DATASET_SOURCES:
    for dataset_type in DATASET_TYPES:
        paths = get_dataset_paths(dataset_source, dataset_type)
        image_dir = paths["images"]
        label_dir = paths["labels"]

        # Get all image and label file paths
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        label_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.lower().endswith('.json')]

        for img_path, label_path in zip(image_files, label_files):
            try:
                # Load image
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    print(f"Warning: Could not read image {img_path}. Skipping.")
                    continue

                # Load labels
                labels, bboxes = load_labels(label_path)

                for label, bbox in zip(labels, bboxes):
                    xmin, ymin, xmax, ymax = map(int, bbox)

                    # Ensure coordinates are within image bounds
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img_bgr.shape[1], xmax)
                    ymax = min(img_bgr.shape[0], ymax)

                    # Add a small margin around the face (optional)
                    margin_x = int(0.1 * (xmax - xmin))
                    margin_y = int(0.1 * (ymax - ymin))
                    xmin = max(0, xmin - margin_x)
                    ymin = max(0, ymin - margin_y)
                    xmax = min(img_bgr.shape[1], xmax + margin_x)
                    ymax = min(img_bgr.shape[0], ymax + margin_y)

                    cropped_face = img_bgr[ymin:ymax, xmin:xmax]

                    if cropped_face.size == 0 or cropped_face.shape[0] < 1 or cropped_face.shape[1] < 1:
                        print(f"Warning: Empty or invalid crop from {img_path} for bbox {bbox}. Skipping.")
                        continue

                    # Resize cropped face to the target size
                    cropped_face_resized = cv2.resize(cropped_face, (IMG_WIDTH, IMG_HEIGHT))

                    # Create output directory for the label within the dataset structure
                    output_subdir = os.path.join(OUTPUT_CROPPED_FACES_DIR, dataset_source, dataset_type, str(label))
                    os.makedirs(output_subdir, exist_ok=True)

                    # Save the cropped face using the original image name
                    original_image_name = os.path.basename(img_path)
                    output_filepath = os.path.join(output_subdir, original_image_name)
                    cv2.imwrite(output_filepath, cropped_face_resized)
                    processed_faces_count += 1

                    if processed_faces_count % 200 == 0:
                        print(f"Processed {processed_faces_count} faces...")

            except Exception as e:
                print(f"Error processing {img_path} with label {label_path}: {e}. Skipping.")

print(f"Dataset preparation complete. Total cropped faces saved: {processed_faces_count}")