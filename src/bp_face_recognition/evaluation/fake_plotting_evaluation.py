import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- Dummy Data for evaluate_methods.py ---
methods = ['Haar Cascade', 'Dlib HOG', 'FaceNet', 'Face Recognition']
datasets = ['seccam', 'webcam', 'seccam_2']
subsets = ['test', 'train', 'val'] # Though the script processes all, plots usually aggregate
numeric_columns = ['num_faces_detected', 'false_positives', 'not_found', 'detection_time', 'accuracy']

all_results_data = []
image_id_counter = 0

for dataset in datasets:
    for method_name in methods:
        for subset in subsets:
            # Simulate a few image results per subset for variety, though the plot aggregates
            for _ in range(5): # 5 dummy images per method/dataset/subset
                image_id_counter += 1
                image_file = f"image_{image_id_counter}.jpg"
                
                # Base performance characteristics for each method
                if method_name == 'Haar Cascade':
                    base_accuracy = np.random.uniform(0.6, 0.8)
                    base_time = np.random.uniform(0.02, 0.05)
                    base_fp = np.random.randint(1, 4)
                    base_nf = np.random.randint(0, 3)
                elif method_name == 'Dlib HOG':
                    base_accuracy = np.random.uniform(0.7, 0.85)
                    base_time = np.random.uniform(0.1, 0.3)
                    base_fp = np.random.randint(0, 2)
                    base_nf = np.random.randint(0, 2)
                elif method_name == 'FaceNet': # (Assuming MTCNN backend or similar)
                    base_accuracy = np.random.uniform(0.85, 0.95)
                    base_time = np.random.uniform(0.05, 0.15)
                    base_fp = np.random.randint(0, 1)
                    base_nf = np.random.randint(0, 1)
                else: # Face Recognition (dlib CNN based)
                    base_accuracy = np.random.uniform(0.8, 0.92)
                    base_time = np.random.uniform(0.2, 0.5)
                    base_fp = np.random.randint(0, 1)
                    base_nf = np.random.randint(0, 1)

                # Add some noise/variation
                accuracy = np.clip(base_accuracy + np.random.normal(0, 0.05), 0, 1)
                detection_time = max(0.01, base_time + np.random.normal(0, 0.01))
                num_detected = np.random.randint(1, 5) # Assume 1-4 faces detected
                false_positives = max(0, base_fp + np.random.randint(-1, 1))
                not_found = max(0, base_nf + np.random.randint(-1,1))
                
                all_results_data.append({
                    'method': method_name,
                    'dataset': dataset,
                    'subset': subset,
                    'image': image_file,
                    'num_faces_detected': num_detected,
                    'false_positives': false_positives,
                    'not_found': not_found,
                    'detection_time': detection_time,
                    'accuracy': accuracy
                })

evaluation_df = pd.DataFrame(all_results_data)
csv_path_evaluation = "dummy_face_detection_results.csv"
evaluation_df.to_csv(csv_path_evaluation, index=False)
print(f"Face detection evaluation dummy data saved to {csv_path_evaluation}")

# Summarize for markdown
summary_df = evaluation_df.groupby(['method'])[numeric_columns].mean()
md_path_evaluation = "dummy_method_evaluation.md"
with open(md_path_evaluation, 'w', encoding='utf-8') as f:
    f.write("# Method evaluation summary (Dummy Data)\n\n")
    summary_df.to_markdown(f, tablefmt="github")
print(f"Face detection evaluation dummy summary saved to {md_path_evaluation}")


# --- Plotting for evaluate_methods.py ---
# Ensure assets/plots directory exists
assets_path = os.path.join('.', 'dummy_assets', 'plots') # Saving to a local dummy_assets
os.makedirs(assets_path, exist_ok=True)

df_plot = pd.DataFrame(all_results_data)

for metric in numeric_columns:
    pivot_df = df_plot.pivot_table(values=metric, index='method', columns='dataset', aggfunc='mean').fillna(0)
    
    ax = pivot_df.plot(
        kind='bar',
        stacked=True, # Stack by dataset
        figsize=(12, 7), # Adjusted for better label visibility
        colormap='viridis' # Changed colormap for variety
    )

    title = metric.replace('_', ' ').capitalize()
    plt.title(f'{title} by Method (Stacked by Dataset)')
    plt.xlabel('Method')
    plt.ylabel(title)
    plt.xticks(rotation=45, ha="right") # Improved rotation and alignment
    plt.legend(title="Dataset")
    plt.tight_layout()
    
    # Annotate values inside bars for stacked charts (sum of stack for label position)
    for i, method in enumerate(pivot_df.index):
        cumulative_height = 0
        for dataset_idx, dataset_name in enumerate(pivot_df.columns):
            value = pivot_df.loc[method, dataset_name]
            if value > 0: # Only annotate non-zero values
                # Position the text in the middle of the current segment
                bar_segment_height = value
                text_y = cumulative_height + bar_segment_height / 2
                ax.text(i, text_y, f'{value:.2f}', ha='center', va='center', color='white', fontsize=8)
            cumulative_height += bar_segment_height
            
    save_path = os.path.join(assets_path, f'dummy_{title.replace(" ", "_")}_comparison.png')
    ax.get_figure().savefig(save_path)
    print(f"Evaluation dummy plot for {title} saved to {save_path}")
    plt.show()
    plt.close()

