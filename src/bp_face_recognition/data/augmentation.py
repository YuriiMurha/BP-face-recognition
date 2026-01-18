import os
import cv2
import json
import numpy as np
import albumentations as alb
from bp_face_recognition.config.settings import settings

# Define image sizes for each dataset source (moved from notebook)
IMAGE_SIZES = {
    "webcam": (480, 640),
    "seccam": (800, 1280),
    "seccam_2": (800, 1280)
}

def get_augmentor(height, width):
    """
    Creates an albumentations augmentation pipeline.
    """
    return alb.Compose([
        alb.RandomCrop(width=width, height=height),
        alb.HorizontalFlip(p=0.5),
        alb.RandomBrightnessContrast(p=0.2),
        alb.RandomGamma(p=0.2),
        alb.RGBShift(p=0.2),
        alb.VerticalFlip(p=0.5)
    ], bbox_params=alb.BboxParams(
        format='albumentations', 
        label_fields=['class_labels'], 
        min_area=0.001, 
        min_visibility=0.1
    ))

def augment_dataset(dataset_name, num_augmentations=60):
    """
    Applies augmentation to the specified dataset.
    """
    if dataset_name not in IMAGE_SIZES:
        print(f"Unknown dataset: {dataset_name}")
        return

    height, width = IMAGE_SIZES[dataset_name]
    augmentor = get_augmentor(height, width)
    
    subsets = ['train', 'test', 'val']
    
    # Base paths
    # Note: The notebook used '../data/datasets' relative to notebook location.
    # We use settings.DATASETS_DIR which is absolute.
    source_base_dir = settings.DATASETS_DIR / dataset_name
    
    # Output to augmented dir
    # notebook saved to 'data/datasets/augmented'
    target_base_dir = settings.AUGMENTED_DIR / dataset_name

    for subset in subsets:
        image_dir = source_base_dir / subset / 'images'
        label_dir = source_base_dir / subset / 'labels'
        
        if not image_dir.exists():
            print(f"Directory not found: {image_dir}")
            continue

        print(f"Augmenting {dataset_name} - {subset}...")

        for image_file in os.listdir(image_dir):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = image_dir / image_file
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            coords_list = []
            labels_list = []
            
            label_filename = f'{os.path.splitext(image_file)[0]}.json'
            label_path = label_dir / label_filename
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    label_data = json.load(f)

                for shape in label_data.get('shapes', []):
                    # Ensure coords are in correct order (x_min, y_min, x_max, y_max)
                    points = shape['points']
                    x_min = min(points[0][0], points[1][0])
                    x_max = max(points[0][0], points[1][0])
                    y_min = min(points[0][1], points[1][1])
                    y_max = max(points[0][1], points[1][1])

                    coords = [x_min, y_min, x_max, y_max]
                    # Normalize coordinates
                    coords = list(np.divide(coords, [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]))
                    coords_list.append(coords)
                    labels_list.append(shape['label'])

            try:
                for x in range(num_augmentations):
                    augmented = augmentor(image=img, bboxes=coords_list, class_labels=labels_list)
                    
                    # Create target dirs
                    target_image_dir = target_base_dir / subset / 'images'
                    target_label_dir = target_base_dir / subset / 'labels'
                    target_image_dir.mkdir(parents=True, exist_ok=True)
                    target_label_dir.mkdir(parents=True, exist_ok=True)

                    augmented_image_name = f'{os.path.splitext(image_file)[0]}.{x}.jpg'
                    cv2.imwrite(str(target_image_dir / augmented_image_name), augmented['image'])

                    annotation = {
                        'image': augmented_image_name,
                        'shapes': []
                    }

                    if label_path.exists():
                        for bbox, label in zip(augmented['bboxes'], augmented['class_labels']):
                            annotation['shapes'].append({
                                'label': label,
                                'points': [
                                    [bbox[0] * img.shape[1], bbox[1] * img.shape[0]],
                                    [bbox[2] * img.shape[1], bbox[3] * img.shape[0]]
                                ]
                            })

                    with open(target_label_dir / f'{os.path.splitext(image_file)[0]}.{x}.json', 'w') as f:
                        json.dump(annotation, f)

            except Exception as e:
                print(f"Error augmenting {image_file}: {e}")

if __name__ == "__main__":
    # Example usage
    for dataset in ['webcam', 'seccam', 'seccam_2']:
         augment_dataset(dataset)
