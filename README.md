# Bachelor Thesis: Face Recognition Using Surveillance Systems

## Overview
This repository contains the code, data, and documentation for my bachelor thesis on **Face Recognition Using Surveillance Systems**. The project explores the implementation and effectiveness of face recognition technology in surveillance environments, leveraging deep learning algorithms to enhance security and monitoring capabilities.

## Features
- **Real-time Face Detection & Recognition**: Integrated webcam and security camera stream processing.
- **Multiple Detection Methods**: Benchmarks for Haar Cascades, Dlib HOG, MTCNN, and FaceNet.
- **Custom Deep Learning Model**: A CNN-based FaceTracker using EfficientNetB0 as a backbone.
- **Modular Architecture**: Decoupled camera, model, and database components.
- **Robust Development**: Managed with `uv` for dependencies and `nox` for testing/linting.
- **Hybrid Storage**: Support for both PostgreSQL and CSV-based storage for embeddings and logs.
- **Comprehensive Evaluation**: Scripts for benchmarking accuracy, detection time, and false positives.

## Technology Stack
- **Language**: Python 3.x
- **Dependency Management**: uv
- **Automation**: nox, Makefile
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, MTCNN, face_recognition (dlib)
- **Data Augmentation**: Albumentations
- **Database**: PostgreSQL

## Project Structure
```
BP-face-recognition/
├── assets/                  # Plots, diagrams, and static assets
├── data/                    # Datasets and training logs
├── notebooks/               # Jupyter notebooks for visualization
├── src/
│   ├── main.py              # Main application entry point
│   ├── config/              # Configuration (settings.py)
│   ├── data/                # Data augmentation and processing
│   ├── database/            # Database storage logic
│   ├── evaluation/          # Benchmarking scripts
│   ├── models/              # Neural networks and detection methods
│   ├── tests/               # Unit tests
│   └── utils/               # Helper functions
├── LaTeX/                   # Thesis source files
├── noxfile.py               # Automation config
├── pyproject.toml           # Project dependencies
├── Makefile                 # Command shortcuts
└── README.md                # Project documentation
```

## Installation

### Prerequisites
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Visual Studio C++ Build Tools (Windows only, required for `dlib`)

### Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/YuriiMurha/BP-face-recognition.git
    cd BP-face-recognition
    ```

2. Sync dependencies using `make` (uses `uv`):
    ```bash
    make setup
    ```
    *Alternatively, using vanilla `uv`:*
    ```bash
    uv sync
    ```

## Usage

### 1. Attendance System (Real-time)
To start the real-time face recognition system:
```bash
make run
# or: uv run src/main.py
```

### 2. Development & Testing
Run unit tests and linting to ensure code quality:
```bash
make test
make lint
```

### 3. Data Processing
Data preparation logic has been moved from notebooks to scripts:
- **Splitting**: `src/evaluation/split_datasets.py`
- **Cropping**: `src/utils/crop_faces.py`
- **Augmentation**: `src/data/augmentation.py`

### 4. Evaluation
To benchmark the different detection methods:
```bash
make evaluate
```

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Yurii Murha - [GitHub](https://github.com/YuriiMurha)
