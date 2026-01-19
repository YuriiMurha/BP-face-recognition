from bp_face_recognition.data.augmentation import augment_dataset

print("Running augmentation smoke test...")
# Run for one dataset with minimal augmentations
augment_dataset("seccam", num_augmentations=1)
print("Augmentation smoke test complete.")
