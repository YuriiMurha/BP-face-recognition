import os
import json
import csv
import dlib
from shapely.geometry import box
from detection_methods import detect_faces_haar, detect_faces_hog, detect_faces_facenet, detect_faces_face_recognition
from concurrent.futures import ThreadPoolExecutor

# Define the numeric columns
numeric_columns = ['num_faces_detected', 'false_positives', 'not_found', 'detection_time', 'accuracy']
# Assets path
assets_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'assets', 'plots'))

def check_paths_exist(paths, subset=None):
    """
    Check if all paths in the list exist. Print a warning for each missing path.
    Returns True if all exist, False otherwise.
    """
    all_exist = True
    for path in paths:
        if not os.path.isdir(path):
            msg = f"Warning: Path '{path}' does not exist."
            if subset:
                msg += f" Skipping subset '{subset}'."
            print(msg)
            all_exist = False
    return all_exist

def evaluate_method(dataset, dataset_path, method_function, method_name):
    results = []
    subsets = ['test', 'train', 'val']

    multithreaded = method_name != 'Dlib HOG'
    for subset in subsets:
        print(f"Evaluating subset '{subset}' of dataset '{dataset}' using method '{method_name}'...")  # Debug log
        results.extend(process_subset(subset, dataset_path, method_function, multithreaded=multithreaded))

    return results

def process_subset(subset, dataset_path, method_function, multithreaded=False):
    images_path = os.path.join(dataset_path, subset, 'images')
    labels_path = os.path.join(dataset_path, subset, 'labels')
    if not check_paths_exist([images_path, labels_path], subset=subset):
        return []
    image_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    if not image_files:
        print(f"Warning: No images found in '{images_path}'. Skipping subset '{subset}'.")
        return []
    
    subset_results = []
    
    if multithreaded:
        with ThreadPoolExecutor() as executor:
            futures = []
            for image_file in image_files:
                image_path = os.path.join(images_path, image_file)
                futures.append(executor.submit(process_image, image_path, image_file, labels_path, method_function, method_name, dataset, subset))
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        subset_results.append(result)
                except Exception as e:
                    print(f"Error in threaded processing: {e}")
    else:
        for image_file in image_files:
            image_path = os.path.join(images_path, image_file)
            try:
                result = process_image(image_path, image_file, labels_path, method_function, method_name, dataset, subset)
                if result:
                    subset_results.append(result)
            except Exception as e:
                print(f"Error processing image '{image_file}' in subset '{subset}' of dataset '{dataset}' using method '{method_name}': {e}")
    return subset_results

def process_image(image_path, image_file, labels_path, method_function, method_name, dataset, subset):
    try:
        import cv2
        image = cv2.imread(image_path)
        if image is None or image.size == 0:
            raise ValueError(f"Failed to load or empty image: {image_path}")

        try:
            faces, detection_time = method_function(image)
        except Exception as e:
            raise RuntimeError(f"Face detection failed for '{image_file}': {e}")

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
            'dataset': dataset,
            'subset': subset,
            'image': image_file,
            'num_faces_detected': len(faces),
            'false_positives': false_positives,
            'not_found': not_found,
            'detection_time': detection_time,
            'accuracy': accuracy
        }
    
    except Exception as e:
        print(f"Error processing image '{image_file}' in subset '{subset}' of dataset '{dataset}' using method '{method_name}': {e}")  # Error log
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

    # Create plots for each metric
    for metric in numeric_columns:
        # Filter numeric columns
        grouped = df.groupby(['method'])[metric].mean().reset_index()
        # Plot the metric directly from the summary
        ax = grouped.plot(x='method', y=metric, kind='bar', stacked=True, figsize=(12, 8), colormap='Set2')
        
        title = metric.replace('_', ' ').capitalize()
        plt.title(f'{title} comparison by subset')
        plt.xlabel('Method')
        plt.ylabel(title)
        plt.xticks(rotation=45)
        plt.legend(title="Subset")
        plt.tight_layout()
        # Annotate values inside bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='center')

        # Save the plot to a file BEFORE showing or closing
        save_path = os.path.join(assets_path, f'{title}.png')
        ax.get_figure().savefig(save_path)
        plt.show()
        plt.waitforbuttonpress()
        plt.close()

def summarize_method_performance(results):
    import pandas as pd
    import os

    df = pd.DataFrame(results)

    # Group by method and calculate mean for all subsets combined
    summary = df.groupby(['method'])[numeric_columns].mean()

    print("\nGeneral Reliability Summary:")
    print(summary)

    # Save summary as markdown table
    markdown_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'method_evaluation.md'))
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write("# Method evaluation summary\n\n")
    df.to_markdown(markdown_path, index=False, tablefmt="github")    
        
    return summary

def get_datasets(base_dir):
    """
    Automatically define and return datasets with their paths.
    """
    dataset_names = ['seccam', 'webcam', 'seccam_2']
    return {name: os.path.normpath(os.path.join(base_dir, '..', 'data', 'datasets', name)) for name in dataset_names}

def get_methods():
    """
    Define and return face detection methods.
    """
    return [
        (detect_faces_haar, 'Haar Cascade'),
        (detect_faces_hog, 'Dlib HOG'),
        (detect_faces_facenet, 'FaceNet'),
        (detect_faces_face_recognition, 'Face Recognition')
    ]

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))  # folder containing evaluate_methods.py
    
    # Get datasets and methods
    datasets = get_datasets(base_dir)
    methods = get_methods()

    # Evaluate each method on each dataset
    all_results = []
    for dataset, dataset_path in datasets.items():
        for method_function, method_name in methods:
            print(f"Evaluating dataset '{dataset}' using method '{method_name}'...")  # Debug log
            results = evaluate_method(dataset, dataset_path, method_function, method_name)
            all_results.extend(results)

    # Save results to CSV
    output_file = os.path.normpath(os.path.join(base_dir, '..', 'data', 'face_detection_results.csv'))
    save_results_to_csv(all_results, output_file)

    # Plot results
    plot_results(all_results)

    # Summarize general reliability
    summarize_method_performance(all_results)
