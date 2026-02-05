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
- **Language**: Python 3.11
- **Dependency Management**: uv (cross-platform with platform-specific dependencies)
- **Automation**: nox, Makefile
- **Deep Learning**: TensorFlow, Keras, TensorFlow Lite (quantization)
- **Computer Vision**: OpenCV, MTCNN, MediaPipe, face_recognition (dlib)
- **Data Augmentation**: Albumentations
- **Database**: PostgreSQL
- **Platform Support**: Windows (CPU), Linux/WSL (GPU acceleration)
- **ONNX Runtime**: CPU version for Windows, GPU version for Linux

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
- Python 3.11+ 
- [uv](https://github.com/astral-sh/uv) (recommended) for cross-platform dependency management
- Visual Studio C++ Build Tools (Windows only, required for `dlib`)
- WSL2 with GPU support (optional, for Linux GPU acceleration)

### Cross-Platform Setup

This project supports development on both Windows and Linux/WSL with automatic platform-specific dependency resolution.

#### Windows Development
1. Clone the repository:
    ```bash
    git clone https://github.com/YuriiMurha/BP-face-recognition.git
    cd BP-face-recognition
    ```

2. Create Windows-specific environment:
    ```bash
    uv venv .venv-win
    .venv-win\Scripts\activate
    uv sync  # Installs onnxruntime (CPU) and Windows-compatible packages
    ```

#### Linux/WSL Development (GPU Accelerated)
1. In WSL2, navigate to the project:
    ```bash
    cd /mnt/d/Coding/Personal/BP-face-recognition
    ```

2. Create Linux-specific environment:
    ```bash
    uv venv .venv-wsl
    source .venv-wsl/bin/activate
    uv sync  # Installs onnxruntime-gpu, dlib and GPU-optimized packages
    ```

#### Quick Setup with Makefile
```bash
make setup  # Uses uv sync with platform-specific resolution
```

**Note**: The same `uv.lock` file works across both platforms - uv automatically selects the appropriate wheels based on your OS.

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

## Cross-Platform Development

### Platform-Specific Features
- **Windows**: CPU-optimized with onnxruntime for inference
- **Linux/WSL**: GPU-accelerated with onnxruntime-gpu, dlib compilation support
- **Shared Codebase**: Single `pyproject.toml` with automatic platform resolution

### Performance Expectations
- **Windows CPU**: Baseline performance (~2-3 FPS for face detection)
- **WSL GPU**: 5-20x speedup with GPU acceleration (30+ FPS achievable)
- **Model Optimization**: TensorFlow Lite quantization provides 2-3x inference speedup

### Environment Management
The project uses platform markers in `pyproject.toml` to automatically install the appropriate packages:
- `dlib; platform_system == 'Linux'` - Only on Linux (buildable)
- `onnxruntime-gpu>=1.23.2; platform_system == 'Linux'` - GPU version on Linux
- `onnxruntime>=1.23.2; platform_system == 'Windows'` - CPU version on Windows

## License
Distributed under MIT License. See `LICENSE` for more information.

## Contact
Yurii Murha - [GitHub](https://github.com/YuriiMurha)
