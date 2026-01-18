import os
import numpy as np
from tqdm import tqdm
from bp_face_recognition.config.settings import settings

def split_dataset(dataset_name, dataset_path):
    subsets = ['train', 'test', 'val']
    images_src = os.path.join(dataset_path, 'images')
    labels_src = os.path.join(dataset_path, 'labels')

    if not os.path.exists(images_src):
        print(f"Skipping {dataset_name}: {images_src} not found.")
        return

    # Create target directories
    for subset in subsets:
        os.makedirs(os.path.join(dataset_path, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, subset, 'labels'), exist_ok=True)

    # Get list of all image files
    all_images = [f for f in os.listdir(images_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    np.random.shuffle(all_images)

    # Calculate splits
    total_images = len(all_images)
    train_count = int(total_images * 0.7)
    test_count = int(np.ceil(total_images * 0.15))
    
    subset_files = {
        'train': all_images[:train_count],
        'test': all_images[train_count:train_count+test_count],
        'val': all_images[train_count+test_count:]
    }

    # Move files
    for subset, files in subset_files.items():
        for filename in tqdm(files, desc=f"Splitting {dataset_name} into {subset}", unit="file", leave=False):
            # Move image
            os.replace(os.path.join(images_src, filename), 
                       os.path.join(dataset_path, subset, 'images', filename))
            
            # Move label if it exists
            label_name = os.path.splitext(filename)[0] + '.json'
            label_src_path = os.path.join(labels_src, label_name)
            if os.path.exists(label_src_path):
                os.replace(label_src_path, 
                           os.path.join(dataset_path, subset, 'labels', label_name))

if __name__ == '__main__':
    base_dir = settings.DATASETS_DIR
    for dataset in ['webcam', 'seccam', 'seccam_2']:
        split_dataset(dataset, os.path.join(base_dir, dataset))
