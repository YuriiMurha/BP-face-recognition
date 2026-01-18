# Implementation Context

## Project Overview
This project is a Face Recognition system developed as a Bachelor's Thesis. It implements various face detection and recognition methods, including traditional computer vision techniques (Haar Cascades, HOG) and Deep Learning models (FaceNet, MTCNN, Custom CNNs).

## Directory Structure

### Root Directory
- **`assets/`**: Contains static assets like plots and diagrams.
- **`data/`**: Stores datasets (raw, cropped, augmented) and training logs.
- **`notebooks/`**: Jupyter notebooks for exploration and visualization. *Core logic has been migrated to `src/` for production use.*
- **`src/`**: Production source code.
- **`LaTeX/`**: Source files for the thesis document.
- **`noxfile.py`**: Configuration for `nox` (automation, testing, linting).
- **`pyproject.toml`**: Project configuration and dependency management (via `uv`).
- **`uv.lock`**: Exact dependency versions lockfile.
- **`Makefile`**: Command shortcuts for setup, testing, and execution.

### Source Code (`src/`)
The source code is organized into modular packages:

- **`src/main.py`**: The entry point for the real-time attendance system application.
- **`src/config/`**: Configuration management.
  - `settings.py`: Centralized settings using Pydantic (paths, database creds, etc.).
- **`src/data/`**: Data processing pipelines.
  - `augmentation.py`: Data augmentation logic (migrated from notebooks).
- **`src/models/`**: Model definitions and loading logic.
  - `interfaces.py`: Defines abstract base classes (`FaceDetector`, `FaceRecognizer`).
  - `model.py`: Defines the `FaceTracker` class, integrating detection and recognition.
  - `dataset_loader.py`: TensorFlow dataset loading utilities.
  - `load_model.py`: Utilities for loading Keras/TensorFlow models.
  - `methods/`: Implementations of specific detection/recognition methods.
    - `haar_cascade.py`: `HaarCascadeDetector` (OpenCV).
    - `dlib_hog.py`: `DlibHOGDetector` (Dlib).
    - `mtcnn_detector.py`: `MTCNNDetector` (MTCNN).
- **`src/database/`**: Database interaction layer.
  - `database.py`: Handles connections to PostgreSQL or CSV-based storage for face embeddings and logs.
- **`src/evaluation/`**: Scripts for evaluating model performance.
  - `evaluate_methods.py`: Main script for running comparisons.
  - `split_datasets.py`: Script to split raw data into train/val/test sets.
  - `testing.py`: Testing routines for neural networks.
- **`src/utils/`**: Utility functions.
  - `camera.py`: Camera handling and video stream processing.
  - `crop_faces.py`: Tools for cropping faces from datasets.
  - `gpu.py`: TensorFlow GPU configuration utilities.
- **`src/tests/`**: Unit tests for the application (`test_settings.py`, `test_database.py`, etc.).

## Key Components

### FaceTracker (`src/models/model.py`)
The core class for the application. It orchestrates:
1.  **Detection**: Locating faces in a frame (using MTCNN, Haar, or HOG).
2.  **Recognition**: Extracting embeddings (using a custom model or FaceNet) and comparing them against the database.

### FaceDatabase (`src/database/database.py`)
Abstraction for data persistence. Supports:
-   **PostgreSQL**: For production-like environments.
-   **CSV**: For simple local testing without setting up a database server.

## Workflow
1.  **Setup**: Run `make setup` to install dependencies and sync the environment using `uv`.
2.  **Data Preparation**:
    -   Run `src/evaluation/split_datasets.py` to organize raw data.
    -   Run `src/utils/crop_faces.py` to prepare inputs.
    -   Run `src/data/augmentation.py` to augment training data.
3.  **Training**: Custom model training logic is encapsulated in `src/models/` and can be triggered via notebooks or scripts.
4.  **Evaluation**: Run `make evaluate` to compare different detection approaches.
5.  **Application**: Run `make run` to start the real-time face recognition camera feed.
6.  **Quality Assurance**: Run `make test` and `make lint` to verify code integrity using `nox`.

## Dependencies
Managed by `uv` in `pyproject.toml`. Key dependencies include:
-   `tensorflow`: Deep learning framework.
-   `opencv-python` (`cv2`): Image processing.
-   `mtcnn`: Face detection.
-   `psycopg2`: PostgreSQL adapter.
-   `face_recognition`: Wrapper for dlib's face recognition.
-   `nox` & `pytest`: For testing and automation.