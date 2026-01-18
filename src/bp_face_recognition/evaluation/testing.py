import os
import json
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0

# Define constants
BASE_DATASET_PATH = os.path.join(os.path.dirname(__file__), "../../data/datasets/augmented")

# Dataset types and dataset sources
DATASET_TYPES = ["train", "test", "val"]
DATASET_SOURCES = ["webcam", "seccam", "seccam_2"]

# Define image sizes for each dataset source
IMAGE_SIZES = {
    "webcam": (480, 640),
    "seccam": (1280, 800),
    "seccam_2": (1280, 800)
}

def get_dataset_paths(dataset_source, dataset_type):
    base_path = os.path.join(BASE_DATASET_PATH, dataset_source, dataset_type)
    return {
        "images": os.path.join(base_path, "images", "*.jpg"),
        "labels": os.path.join(base_path, "labels", "*.json")
    }

def load_image_with_size(image_path, source):
    """
    Reads and decodes an image from a given file path and resizes it based on the source.
    Args:
        image_path (tf.Tensor): The file path of the image.
        source (str): The dataset source to determine the target size.
    Returns:
        tf.Tensor: A decoded, resized, and normalized image tensor.
    """
    try:
        tf.debugging.assert_type(image_path, tf.string, "Image path must be a string tensor.")
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        target_size = IMAGE_SIZES[source]
        image = tf.image.resize(image, target_size)
        image = tf.image.convert_image_dtype(image, tf.float32)
        tf.debugging.assert_shapes([(image, (*target_size, 3))], "Image shape mismatch after resizing.")
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return tf.zeros((*IMAGE_SIZES[source], 3), dtype=tf.float32)

def load_labels_with_flag(label_path):
    try:
        if isinstance(label_path, tf.Tensor):
            label_path = label_path.numpy().decode('utf-8')
        with open(label_path, 'r', encoding="utf-8") as f:
            label_data = json.load(f)
        shapes = label_data.get("shapes")
        if shapes is not None and isinstance(shapes, list) and len(shapes) > 0:
            labels = [int(shape["label"]) for shape in shapes]
            points = np.array([shape["points"] for shape in shapes], dtype=np.float32)
            return labels, points, True
        else:
            return [], np.array([]), False
    except Exception as e:
        print(f"Error loading label file {label_path}: {e}")
        return [], np.array([]), False

def tf_load_labels(label_path):
    labels, points, valid = tf.py_function(
        load_labels_with_flag, [label_path], [tf.float32, tf.float32, tf.bool]
    )
    tf.debugging.assert_type(valid, tf.bool, "Valid flag must be a boolean tensor.")
    return labels, points, valid

def create_datasets_with_size(dataset_source):
    """
    Creates TensorFlow datasets for a given dataset source.

    Args:
        dataset_source (str): The source of the dataset (e.g., 'webcam').

    Returns:
        dict: A dictionary containing train, test, and validation datasets.
    """
    datasets = {}

    for dataset_type in DATASET_TYPES:
        img_size = IMAGE_SIZES[dataset_source]
        paths = get_dataset_paths(dataset_source, dataset_type)

        # Validate paths
        if not tf.io.gfile.exists(os.path.dirname(paths["images"])):
            print(f"Warning: Image path does not exist: {paths['images']}")
        if not tf.io.gfile.exists(os.path.dirname(paths["labels"])):
            print(f"Warning: Label path does not exist: {paths['labels']}")

        # Load image dataset
        images = tf.data.Dataset.list_files(paths["images"], shuffle=False)
        images = images.map(lambda x: load_image_with_size(x, dataset_source), num_parallel_calls=tf.data.AUTOTUNE)

        # Load label dataset
        labels = tf.data.Dataset.list_files(paths["labels"], shuffle=False)
        labels = labels.map(tf_load_labels, num_parallel_calls=tf.data.AUTOTUNE)
        labels = labels.filter(lambda labels, points, valid: valid)
        labels = labels.map(lambda labels, points, valid: (labels, points), num_parallel_calls=tf.data.AUTOTUNE)

        # Validate dataset consistency
        num_images = len(tf.io.gfile.glob(paths["images"]))
        num_labels = len(tf.io.gfile.glob(paths["labels"]))
        print(f"[{dataset_source}][{dataset_type}] {num_images} images, {num_labels} labels, image size: {img_size}")

        # Validate dataset consistency
        if num_images != num_labels:
            print(f"Warning: Mismatch between number of images and labels in {dataset_source}/{dataset_type}")

        dataset = tf.data.Dataset.zip((images, labels))
        dataset = dataset.shuffle(5000 if dataset_type == "train" else 1000).batch(8).prefetch(tf.data.AUTOTUNE)
        datasets[dataset_type] = dataset

    return datasets

def create_datasets_with_opencv(dataset_source):
    """
    Creates datasets using OpenCV and basic Python for a given dataset source.

    Args:
        dataset_source (str): The source of the dataset (e.g., 'webcam').

    Returns:
        dict: A dictionary containing train, test, and validation datasets.
    """
    datasets = {}

    for dataset_type in DATASET_TYPES:
        base_path = os.path.join(BASE_DATASET_PATH, dataset_source, dataset_type)
        image_dir = os.path.join(base_path, "images")
        label_dir = os.path.join(base_path, "labels")

        # List image and label files
        image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")])
        label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".json")])

        if len(image_files) != len(label_files):
            print(f"Warning: Mismatch between number of images and labels in {dataset_source}/{dataset_type}")
            continue

        images = []
        labels = []

        for img_path, lbl_path in zip(image_files, label_files):
            # Load and preprocess image
            image = cv2.imread(img_path)
            target_size = IMAGE_SIZES[dataset_source]
            image = cv2.resize(image, (target_size[1], target_size[0]))
            image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
            images.append(image)

            # Load and preprocess label
            with open(lbl_path, 'r', encoding="utf-8") as f:
                label_data = json.load(f)
            shapes = label_data.get("shapes", [])
            if shapes:
                labels.append({
                    "class": [int(shape["label"]) for shape in shapes],
                    "points": np.array([shape["points"] for shape in shapes], dtype=np.float32)
                })
            else:
                labels.append({"class": [], "points": np.array([])})

        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)

        datasets[dataset_type] = (images, labels)

    return datasets

def build_model():
    """
    Builds a face recognition model using EfficientNetB0.

    Returns:
        Model: A Keras model instance.
    """
    max_height = max([s[0] for s in IMAGE_SIZES.values()])
    max_width = max([s[1] for s in IMAGE_SIZES.values()])
    input_layer = Input(shape=(max_height, max_width, 3))

    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=input_layer)
    for layer in base_model.layers[:50]:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    embedding = Dense(128, activation='linear', name='embedding')(x)

    return Model(inputs=input_layer, outputs=embedding)

def validate_dataset(dataset, dataset_type):
    """Validate dataset structure and data types."""
    for X_batch, y_batch in dataset.take(1):
        # Check image tensor shape and type
        assert X_batch.shape[1:] == IMAGE_SIZES["webcam"], f"Image shape mismatch for {dataset_type}"
        assert X_batch.dtype == tf.float32, f"Image dtype mismatch for {dataset_type}"
        
        # Check label tensor shapes
        assert y_batch[0].dtype == tf.int32, f"Label dtype mismatch for {dataset_type}"
        assert y_batch[1].shape[1] == 4, f"Bounding box shape mismatch for {dataset_type}"
        
        print(f"[Validation] {dataset_type} dataset validated successfully")

def test_full_pipeline():
    """Test complete pipeline from dataset creation to model input validation."""
    webcam_sets = create_datasets_with_opencv("webcam")
    
    # Validate all datasets
    for dataset_type in DATASET_TYPES:
        validate_dataset(webcam_sets[dataset_type], dataset_type)
    
    # Test model input compatibility
    facetracker = build_model()
    train_dataset = webcam_sets["train"]
    for X_batch, _ in train_dataset.take(1):
        facetracker(X_batch)  # Test model can process batch without error
    print("[Test] Full pipeline validation completed successfully")

def main():
    """
    Main function to execute the workflow step-by-step.
    """
    # Step 1: Create datasets
    print("Creating datasets...")
    webcam_sets = create_datasets_with_size("webcam")

    # Step 2: Build the model
    print("Building the model...")
    build_model()

    # Step 3: Inspect a sample batch
    print("Inspecting a sample batch...")
    train_dataset = webcam_sets["train"]
    for X_batch, y_batch in train_dataset.take(1):
        print("Image batch shape:", X_batch.shape)
        print("Class label batch shape:", y_batch[0].shape)
        print("Bounding box batch shape:", y_batch[1].shape)

    # Step 4: Train the model (placeholder for training logic)
    print("Training logic can be added here.")

if __name__ == "__main__":
    test_full_pipeline()