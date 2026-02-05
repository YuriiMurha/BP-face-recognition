from unittest.mock import MagicMock, patch
import tensorflow as tf
import numpy as np
import cv2
import os

print("Running main application logic smoke test...")

# Mock tf.keras.models.load_model BEFORE importing main or model
# We need to patch where it is used.
with patch("tensorflow.keras.models.load_model") as mock_load_model:
    # Setup mock model
    mock_model = MagicMock()
    # predict return shape: (1, 128) embedding
    mock_model.predict.return_value = np.zeros((1, 128), dtype=np.float32)
    mock_load_model.return_value = mock_model

    from bp_face_recognition.main import AttendanceApp
    from bp_face_recognition.config.settings import settings

    # ... rest of the script ...

    # Use an existing image as a "frame"
    image_path = (
        settings.DATASETS_DIR
        / "seccam/test/images/4598b7e9-83ac-11ee-b027-9cfeff47d2fa.jpg"
    )
    if not image_path.exists():
        print(f"Test image not found at {image_path}. Looking for others...")
        # Fallback to search
        test_dir = settings.DATASETS_DIR / "seccam/test/images"
        if test_dir.exists():
            files = os.listdir(test_dir)
            if files:
                image_path = test_dir / files[0]
                print(f"Using {image_path}")

    if not image_path.exists():
        print("No test image found. Aborting.")
        exit(1)

    frame = cv2.imread(str(image_path))
    if frame is None:
        print("Failed to load image.")
        exit(1)

    # Initialize App with CSV DB (no Postgres needed)
    app = AttendanceApp(db_type="csv")

    try:
        print("Processing frame...")
        processed_frame = app.process_frame(frame)

        if processed_frame is not None:
            print("Frame processed successfully.")
            print(f"Processed frame shape: {processed_frame.shape}")

            # Check if logs were created
            log_file = settings.LOGS_DIR / "logs.txt"
            if log_file.exists():
                print(f"Log file exists: {log_file}")
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    if lines:
                        print(f"Last log entry: {lines[-1].strip()}")
                    else:
                        print("Log file is empty.")
        else:
            print("Processed frame is None.")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback

        traceback.print_exc()

    print("Main application logic smoke test complete.")
