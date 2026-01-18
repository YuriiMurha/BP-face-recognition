import os
import json
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import box
from concurrent.futures import ThreadPoolExecutor

from bp_face_recognition.config.settings import settings
from bp_face_recognition.models.methods.haar_cascade import HaarCascadeDetector
from bp_face_recognition.models.methods.dlib_hog import DlibHOGDetector
from bp_face_recognition.models.methods.mtcnn_detector import MTCNNDetector
from bp_face_recognition.evaluation.detection_methods import detect_faces_face_recognition

# Define the numeric columns
numeric_columns = ['num_faces_detected', 'false_positives', 'not_found', 'detection_time', 'accuracy']

def check_paths_exist(paths, subset=None):
    all_exist = True
    for path in paths:
        if not os.path.isdir(path):
            msg = f"Warning: Path '{path}' does not exist."
            if subset:
                msg += f" Skipping subset '{subset}'."
            print(msg)
            all_exist = False
    return all_exist

def evaluate_method(dataset, dataset_path, detector_instance, method_name):
    results = []
    subsets = ['test', 'train', 'val']

    # Multithreading can be problematic for some dlib/tensorflow versions if not handled carefully
    # But let's try keeping it for MTCNN/Haar
    multithreaded = method_name not in ['Dlib HOG', 'Face Recognition']
    
    for subset in subsets:
        results.extend(process_subset(subset, dataset_path, detector_instance, method_name, dataset, multithreaded=multithreaded))

    return results

def process_subset(subset, dataset_path, detector_instance, method_name, dataset, multithreaded=False):
    images_path = os.path.join(dataset_path, subset, 'images')
    labels_path = os.path.join(dataset_path, subset, 'labels')
    
    if not check_paths_exist([images_path, labels_path], subset=subset):
        return []
        
    image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"Warning: No images found in '{images_path}'. Skipping subset '{subset}'.")
        return []
    
    subset_results = []
    
    if multithreaded:
        with ThreadPoolExecutor() as executor:
            futures = []
            for image_file in image_files:
                image_path = os.path.join(images_path, image_file)
                futures.append(executor.submit(process_image, image_path, image_file, labels_path, detector_instance, method_name, dataset, subset))
            for future in tqdm(futures, desc=f"Processing {subset} ({method_name})", unit="img"):
                result = future.result()
                if result:
                    subset_results.append(result)
    else:
        for image_file in tqdm(image_files, desc=f"Processing {subset} ({method_name})", unit="img"):
            image_path = os.path.join(images_path, image_file)
            result = process_image(image_path, image_file, labels_path, detector_instance, method_name, dataset, subset)
            if result:
                subset_results.append(result)
    return subset_results

def process_image(image_path, image_file, labels_path, detector_instance, method_name, dataset, subset):
    try:
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            return None

        start_time = time.time()
        # Handle both class instances and old functions for now
        if hasattr(detector_instance, 'detect'):
            boxes = detector_instance.detect(image)
        else:
            # Fallback for old functions if needed
            boxes, _ = detector_instance(image)
        detection_time = time.time() - start_time

        # Convert detected boxes to [x1, y1, x2, y2]
        detected_boxes = []
        for box_item in boxes:
            if isinstance(box_item, (list, tuple)):
                x, y, w, h = box_item
                detected_boxes.append((x, y, x + w, y + h))
            else:
                # Dlib rectangle if still returned by old function
                detected_boxes.append((box_item.left(), box_item.top(), box_item.right(), box_item.bottom()))

        # Get ground truth
        labels_file = os.path.join(labels_path, os.path.splitext(image_file)[0] + '.json')
        ground_truth_faces = get_ground_truth_faces(labels_file) if os.path.exists(labels_file) else []

        accuracy, false_positives, not_found = evaluate_accuracy(detected_boxes, ground_truth_faces)
        
        return {
            'method': method_name,
            'dataset': dataset,
            'subset': subset,
            'image': image_file,
            'num_faces_detected': len(boxes),
            'false_positives': false_positives,
            'not_found': not_found,
            'detection_time': detection_time,
            'accuracy': accuracy
        }
    except Exception as e:
        print(f"Error processing {image_file}: {e}")
        return None

def calculate_iou(boxA, boxB):
    bA = box(*boxA)
    bB = box(*boxB)
    try:
        intersection = bA.intersection(bB).area
        union = bA.union(bB).area
        return intersection / union if union > 0 else 0
    except Exception:
        return 0

def evaluate_accuracy(detected_faces, ground_truth_faces, iou_threshold=0.5):
    matched = 0
    gt_matched = [False] * len(ground_truth_faces)
    
    for det_face in detected_faces:
        for i, gt_face in enumerate(ground_truth_faces):
            if not gt_matched[i]:
                if calculate_iou(det_face, gt_face) >= iou_threshold:
                    matched += 1
                    gt_matched[i] = True
                    break
                    
    total_gt = len(ground_truth_faces)
    total_detected = len(detected_faces)
    false_positives = total_detected - matched
    not_found = total_gt - matched
    accuracy = matched / total_gt if total_gt > 0 else 0
    return accuracy, false_positives, not_found

def get_ground_truth_faces(label_file):
    with open(label_file, 'r') as f:
        label_data = json.load(f)
    return [tuple(shape['points'][0] + shape['points'][1]) for shape in label_data['shapes']]

def save_results_to_csv(results, output_file):
    if not results:
        return
    keys = results[0].keys()
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

def plot_results(results):
    df = pd.DataFrame(results)
    assets_path = settings.ASSETS_DIR / 'plots'
    assets_path.mkdir(parents=True, exist_ok=True)

    for metric in numeric_columns:
        pivot_df = df.pivot_table(values=metric, index='method', columns='dataset', aggfunc='mean').fillna(0)
        ax = pivot_df.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set2')
        title = metric.replace('_', ' ').capitalize()
        plt.title(f'{title} comparison by dataset')
        plt.xlabel('Method')
        plt.ylabel(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='center')

        plt.savefig(assets_path / f'{title}.png')
        plt.close()

def summarize_method_performance(results):
    df = pd.DataFrame(results)
    summary = df.groupby(['method'])[numeric_columns].mean()
    print("\nGeneral Reliability Summary:")
    print(summary)
    
    markdown_path = settings.DATA_DIR / 'method_evaluation.md'
    summary.to_markdown(markdown_path, tablefmt="github")    
    return summary

if __name__ == '__main__':
    # Define datasets using settings
    datasets = {name: settings.DATASETS_DIR / name for name in ['seccam', 'webcam', 'seccam_2']}
    
    # Initialize detectors
    methods = [
        (HaarCascadeDetector(), 'Haar Cascade'),
        (DlibHOGDetector(), 'Dlib HOG'),
        (MTCNNDetector(), 'MTCNN'),
        (detect_faces_face_recognition, 'Face Recognition')
    ]

    all_results = []
    for dataset_name, dataset_path in tqdm(datasets.items(), desc="Datasets", unit="dataset"):
        if not dataset_path.exists():
            print(f"Skipping {dataset_name}, path {dataset_path} not found.")
            continue
            
        for detector, method_name in tqdm(methods, desc=f"Methods for {dataset_name}", unit="method", leave=False):
            results = evaluate_method(dataset_name, dataset_path, detector, method_name)
            all_results.extend(results)

    if all_results:
        output_file = settings.DATA_DIR / 'face_detection_results.csv'
        save_results_to_csv(all_results, output_file)
        plot_results(all_results)
        summarize_method_performance(all_results)