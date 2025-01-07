import os
import json
import csv
import dlib
from shapely.geometry import box
from detection_methods import detect_faces_haar, detect_faces_hog, detect_faces_facenet, detect_faces_face_recognition
from concurrent.futures import ThreadPoolExecutor

# Define the numeric columns
numeric_columns = ['num_faces_detected', 'false_positives', 'not_found', 'detection_time', 'accuracy']

def evaluate_method(dataset_path, method_function, method_name):
    results = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for subset in ['test', 'train', 'val']:
            images_path = os.path.join(dataset_path, subset, 'images')
            labels_path = os.path.join(dataset_path, subset, 'labels')
            for image_file in os.listdir(images_path):
                image_path = os.path.join(images_path, image_file)
                futures.append(executor.submit(process_image, image_path, image_file,labels_path, method_function, method_name, subset))
        
        for future in futures:
            result = future.result()
            if result:
                results.append(result)

    return results

def process_image(image_path, image_file, labels_path, method_function, method_name, subset):
    try:
        import cv2
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        faces, detection_time = method_function(image)

        # Get ground truth bounding boxes from Labelme
        labels_file = os.path.join(labels_path, image_file.replace('.jpg', '.json'))  # Adjust for your format
        if os.path.exists(labels_file):
            ground_truth_faces = get_ground_truth_faces(labels_file)
        else:
            ground_truth_faces = []

        # Fix for handling different detection methods
        detected_boxes = []
        for face in faces:
            if isinstance(face, dlib.rectangle):  # Dlib's HOG detector
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                detected_boxes.append((x, y, x + w, y + h))
            elif isinstance(face, dict) and 'box' in face:  # MTCNN detector
                x, y, w, h = face['box']
                detected_boxes.append((x, y, x + w, y + h))
            elif len(face) == 4:  # Assume (x, y, w, h) tuple for other methods
                x, y, w, h = face
                detected_boxes.append((x, y, x + w, y + h))
            else:
                raise ValueError(f"Unexpected face format: {face}")

        # Calculate accuracy
        accuracy, false_positives, not_found = evaluate_accuracy(detected_boxes, ground_truth_faces)
        
        return {
            'method': method_name,
            'subset': subset,
            'image': image_file,
            'num_faces_detected': len(faces),
            'false_positives': false_positives,
            'not_found': not_found,
            'detection_time': detection_time,
            'accuracy': accuracy
        }
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def calculate_iou(boxA, boxB):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.

    Args:
        boxA (tuple or list): [x1, y1, x2, y2] coordinates of the first box
        boxB (tuple or list): [x1, y1, x2, y2] coordinates of the second box

    Returns:
        float: Intersection over Union (IoU) between the two boxes
    """
    boxA = box(*boxA)
    boxB = box(*boxB)
    intersection = boxA.intersection(boxB).area
    union = boxA.union(boxB).area
    return intersection / union if union > 0 else 0


def evaluate_accuracy(detected_faces, ground_truth_faces, iou_threshold=0.5):
    """
    Evaluate the accuracy of face detection by comparing the detected faces with ground truth annotations.

    Args:
        detected_faces (list of tuples or lists): [x1, y1, x2, y2] coordinates of the detected faces
        ground_truth_faces (list of tuples or lists): [x1, y1, x2, y2] coordinates of the ground truth faces
        iou_threshold (float, optional): The minimum Intersection over Union (IoU) required to be considered a match. Defaults to 0.5.

    Returns:
        tuple: A tuple containing the accuracy, the number of false positives, and the number of faces not found
    """
    matched = 0
    for gt_face in ground_truth_faces:
        for detected_face in detected_faces:
            iou = calculate_iou(gt_face, detected_face)
            if iou >= iou_threshold:
                matched += 1
                break  # Move to the next ground truth face once matched
    total_gt = len(ground_truth_faces)
    total_detected = len(detected_faces)

    false_positives = total_detected - matched
    not_found = total_gt - matched
    accuracy = matched / total_gt if total_gt > 0 else 0

    return accuracy, false_positives, not_found

def get_ground_truth_faces(label_file):
    """
    Extract the ground truth bounding boxes from a Labelme JSON file.

    Args:
        label_file (str): The path to the Labelme JSON file

    Returns:
        list of tuples: A list of tuples containing the ground truth bounding boxes in [x1, y1, x2, y2] format
    """
    with open(label_file, 'r') as f:
        label_data = json.load(f)
    # Assuming 'shapes' contains bounding box info in [x1, y1, x2, y2] format
    return [tuple(shape['points'][0] + shape['points'][1]) for shape in label_data['shapes']]


def save_results_to_csv(results, output_file):    
    keys = results[0].keys()
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

def plot_results(results):
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.DataFrame(results)

    # Filter numeric columns
    summary = df.groupby(['method'])[numeric_columns].mean().reset_index()

    # Create plots for each metric
    for metric in numeric_columns:
        pivot = summary.pivot(index='method', columns='method', values=metric)
        pivot.plot(kind='bar', figsize=(12, 8), alpha=0.8)
        
        title = metric.replace('_', ' ').capitalize()
        plt.title(f'{title} comparison')
        plt.ylabel(title)
        plt.xticks(rotation=0)
        plt.legend(title='Method', loc='upper left')
        plt.tight_layout()
        plt.show()
        # save the plot to a file
        plt.savefig(f'{title} comparison.png')
        plt.close()

def summarize_method_performance(results):
    import pandas as pd
    df = pd.DataFrame(results)

    # Group by method and calculate mean for all subsets combined
    summary = df.groupby(['method'])[numeric_columns].mean()

    print("\nGeneral Reliability Summary:")
    print(summary)

    return summary

if __name__ == '__main__':
    # Define dataset paths and methods
    datasets = ['FaceDetection/data/seccam', 'FaceDetection/data/webcam']
    methods = [
        (detect_faces_haar, 'Haar Cascade'),
        (detect_faces_hog, 'Dlib HOG'),
        (detect_faces_facenet, 'FaceNet'),
        (detect_faces_face_recognition, 'Face Recognition')
    ]

    # Evaluate each method on each dataset
    all_results = []
    for dataset_path in datasets:
        for method_function, method_name in methods:
            results = evaluate_method(dataset_path, method_function, method_name)
            all_results.extend(results)

    # Save results to CSV
    save_results_to_csv(all_results, 'face_detection_results.csv')

    # Plot results
    plot_results(all_results)

    # Summarize general reliability
    summarize_method_performance(all_results)
