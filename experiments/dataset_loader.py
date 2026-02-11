import os
import tensorflow as tf
from bp_face_recognition.config.settings import settings

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 8
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

DATASET_TYPES = ["train", "test", "val"]

def get_dataset_paths(dataset_source, dataset_type):
    """
    Generates dataset paths based on source and type.
    """
    base_path = settings.CROPPED_DIR / dataset_source / dataset_type
    return {
        "images": str(base_path / "*.jpg"),
    }

def load_image_and_label(image_path):
    """
    Reads and decodes an image from a given file path.
    
    Args:
        image_path (tf.Tensor): The file path of the image.
    
    Returns:
        tuple: (decoded image tensor, label integer)
    """
    image = tf.io.read_file(image_path)  # Read the image file
    image = tf.image.decode_jpeg(image, channels=3)  # Decode the image (ensure 3 color channels)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize pixel values to [0,1]
    
    # Extract the label from the filename
    filename = tf.strings.split(image_path, os.sep)[-1]
    label_str = tf.strings.split(filename, ".")[-2]  # Extract the part before the file extension
    # convert label to integer
    label = tf.strings.to_number(label_str, out_type=tf.int32)
    return image, label

def create_datasets(dataset_source, batch_size=BATCH_SIZE):
    """
    Creates TensorFlow datasets for each split (train, test, val) for a given source.
    """
    datasets = {}
    num_images_train = 0

    for dataset_type in DATASET_TYPES:
        paths = get_dataset_paths(dataset_source, dataset_type)

        # Load image dataset
        # Note: list_files returns a dataset of file strings
        images = tf.data.Dataset.list_files(paths["images"], shuffle=False)
        dataset = images.map(load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)

        # Count images (this might be slow for very large datasets, but fine here)
        num_images = len(tf.io.gfile.glob(paths["images"]))
        print(f"[{dataset_source}][{dataset_type}] {num_images} images, image size: {IMAGE_SIZE}")
        
        if dataset_type == "train":
            num_images_train = num_images
        
        # Shuffle, batch, repeat, prefetch
        dataset = dataset.shuffle(5000 if dataset_type == "train" else 1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        datasets[dataset_type] = dataset

    return datasets, num_images_train
