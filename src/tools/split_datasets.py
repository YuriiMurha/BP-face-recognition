# Get total number of images in source directory
total_images = len(os.listdir(os.path.join('data', 'datasets', dataset, 'images')))

# Calculate splits based on percentages
train_count = int(total_images * 0.7)  # 70% for training
test_count = int(np.ceil(total_images * 0.15))  # 15% ceiling for test
val_count = total_images - train_count - test_count  # Remainder for validation

# Create target directories if they don't exist
for subset in subsets:
    os.makedirs(os.path.join('data', 'datasets', dataset, subset, 'labels'), exist_ok=True)
    os.makedirs(os.path.join('data', 'datasets', dataset, subset, 'images'), exist_ok=True)

# Get list of all image files and shuffle them randomly
all_images = os.listdir(os.path.join('data', 'datasets', dataset, 'images'))
np.random.shuffle(all_images)

# Split indices based on calculated counts
train_files = all_images[:train_count]
test_files = all_images[train_count:train_count+test_count]
val_files = all_images[train_count+test_count:]

# Create dictionary mapping subset to file list
subset_files = {
    subsets[0]: train_files,  # train
    subsets[1]: test_files,   # test
    subsets[2]: val_files     # val
}

# Move files to appropriate directories
if file in all_images:
    for subset, files in subset_files.items():
    if file in files:
        # Move label if it exists
        if os.path.exists(existing_filepath):
        new_filepath = os.path.join('data', 'datasets', dataset, subset, 'labels', filename)
        os.replace(existing_filepath, new_filepath)
        # Move image
        image_src = os.path.join('data', 'datasets', dataset, 'images', file)
        image_dst = os.path.join('data', 'datasets', dataset, subset, 'images', file)
        if os.path.exists(image_src):
        os.replace(image_src, image_dst)
        break