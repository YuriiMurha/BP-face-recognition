from bp_face_recognition.models.dataset_loader import create_datasets
import tensorflow as tf

print("Running dataset loader smoke test...")
try:
    # Use 'seccam' as we verified it has cropped images now
    datasets, num_train = create_datasets("seccam", batch_size=2)

    print(f"Number of training images: {num_train}")

    # Iterate over one batch from train dataset
    for images, labels in datasets["train"].take(1):
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        print(f"Labels: {labels.numpy()}")
        break

    print("Dataset loader smoke test complete.")
except Exception as e:
    print(f"Dataset loader failed: {e}")
