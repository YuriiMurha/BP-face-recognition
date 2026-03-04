# Bachelor Thesis: Face Recognition Using Surveillance Systems

## Overview
This repository contains the code, data, and documentation for my bachelor thesis on **Face Recognition Using Surveillance Systems**. The project explores the implementation and effectiveness of face recognition technology in surveillance environments, leveraging deep learning algorithms to enhance security and monitoring capabilities.

The system supports both CPU and GPU training with automated setup via Make commands. GPU training provides 3-4x speedup for model training on NVIDIA hardware.

## Features
- **Real-time Face Detection & Recognition**: Integrated webcam and security camera stream processing.
- **5 Detection Methods**: MediaPipe BlazeFace, MTCNN, Haar Cascades, Dlib HOG, and Face Recognition library.
- **4 Recognition Models**: EfficientNetB0 (baseline, quantized, best) and FaceNet for comprehensive model comparison.
- **GPU Acceleration**: NVIDIA GPU support with 3-4x faster training.
- **Production Model Registry**: Configuration-driven plugin system with runtime model switching.
- **8 Environment Profiles**: Optimized configurations for development, production, and benchmarking.
- **Model Quantization**: TensorFlow Lite optimization achieving 73.6% size reduction.
- **Modular Architecture**: Decoupled detection/recognition, services layer, and plugin-based model management.
- **Hybrid Storage**: Support for both PostgreSQL and CSV-based storage for embeddings and logs.
- **Comprehensive Evaluation**: Scripts for benchmarking accuracy, detection time, and false positives.

## Technology Stack
- **Language**: Python 3.11
- **Dependency Management**: uv
- **Automation**: nox, Makefile
- **Deep Learning**: TensorFlow 2.15, Keras, TensorFlow Lite
- **Computer Vision**: OpenCV, MTCNN, MediaPipe, face_recognition
- **GPU**: NVIDIA CUDA 12.2, cuDNN 8.9 (optional)
- **Data Augmentation**: Albumentations
- **Database**: PostgreSQL
- **Platform Support**: Windows (CPU), Linux/WSL (CPU/GPU)

## Quick Start

### Prerequisites
- Python 3.10+ (3.11 recommended)
- [uv](https://github.com/astral-sh/uv) for dependency management
- Windows 10/11 or WSL2 (Ubuntu)

### Step-by-Step Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/BP-face-recognition.git
cd BP-face-recognition
```

#### 2. Configure Project Path (Optional)
Edit `Makefile` and set your project path:
```makefile
# Default: d:/Coding/Personal/BP-face-recognition
# Change this to your actual project location
WSL_WORKDIR ?= d:/Your/Project/Path/BP-face-recognition
```

#### 3. Windows Setup (CPU Training)
```bash
# Create virtual environment
uv venv .venv-win
.venv-win\Scripts\activate
uv sync

# Test training
make train-cpu backbone=MobileNetV3Small epochs=5
```

#### 4. WSL2 Setup (Recommended for Training)
```bash
# Install WSL2 with Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# In WSL terminal, navigate to project
cd /mnt/d/Coding/Personal/BP-face-recognition  # Adjust path as needed

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create WSL virtual environment
uv venv .venv-wsl
source .venv-wsl/bin/activate

# Install core dependencies (skip dlib which requires cmake)
uv pip install tensorflow opencv-python numpy pillow scikit-learn

# Verify installation
make verify-wsl

# Start training
make train-wsl backbone=MobileNetV3Small epochs=5
```

### Training Commands

#### Windows (CPU)
```bash
make train-cpu backbone=EfficientNetB0 epochs=20 dataset=seccam_2
make train-cpu backbone=MobileNetV3Small epochs=20
```

#### WSL2 (Better Performance)
```bash
# Quick test (5 epochs)
make train-wsl backbone=MobileNetV3Small epochs=5

# Full training (20 epochs)
make train-wsl backbone=EfficientNetB0 epochs=20

# Custom configuration
make train-wsl backbone=MobileNetV3Small epochs=15 dataset=your_dataset
```

## WSL2 GPU Setup (Optional)

For 3-4x faster training on NVIDIA GPUs in WSL2:

### Prerequisites
- NVIDIA GPU (GTX 1650+ recommended)
- 4GB+ VRAM

### Automated Setup
```powershell
# In Windows PowerShell
make setup-wsl-gpu      # Complete GPU setup
make verify-wsl-gpu     # Verify GPU detection
make gpu-status         # Check GPU status
```

### Manual Setup
```bash
# 1. Install NVIDIA drivers on Windows (nvidia.com)
# 2. Verify GPU in WSL:
wsl -d Ubuntu-22.04 nvidia-smi

# 3. Run setup script:
wsl -d Ubuntu-22.04
cd /mnt/d/Coding/Personal/BP-face-recognition
bash scripts/setup_gpu_wsl.sh
```

### GPU Performance

| Model | CPU Time | GPU Time | VRAM |
|-------|----------|----------|------|
| MobileNetV3Small (20 epochs) | ~35 min | ~9 min | ~2GB |
| EfficientNetB0 (20 epochs) | ~60 min | ~15 min | ~3GB |

### GPU Commands
```bash
make train-wsl backbone=MobileNetV3Small epochs=20   # Auto-uses GPU
make gpu-status                                      # Monitor GPU
make train-wsl args="--force-cpu"                    # Force CPU
```

### Troubleshooting

**GPU not detected:**
```bash
# Check drivers
nvidia-smi

# Reinstall libraries
pip install --force-reinstall tensorflow==2.15.0
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12
```

**Out of memory:** Reduce batch size in `production_trainer.py` (default: 32)

## Project Structure
```
BP-face-recognition/
├── assets/                  # Plots, diagrams, and static assets
├── data/                    # Datasets and training logs
│   ├── cropped/            # Cropped face images
│   └── logs/               # Training logs
├── notebooks/               # Jupyter notebooks
├── scripts/                 # Utility scripts
├── src/
│   ├── bp_face_recognition/
│   │   ├── vision/         # Face detection/recognition
│   │   │   ├── detection/  # Detection methods
│   │   │   ├── recognition/# Recognition models
│   │   │   └── training/   # Training pipeline
│   │   ├── services/       # Business logic
│   │   └── utils/          # Helper functions
│   └── scripts/            # Data processing scripts
├── config/models.yaml      # Model registry configuration
├── Makefile                # Command shortcuts
├── pyproject.toml          # Dependencies
└── README.md
```

## Makefile Commands

### Setup & Environment
```bash
make setup              # Install dependencies
make verify-wsl         # Verify WSL environment
make setup-wsl          # Setup WSL environment
make setup-wsl-gpu      # Setup WSL with GPU support
make verify-wsl-gpu     # Verify GPU is detected
make gpu-status         # Check GPU status
```

### Training
```bash
# Windows (CPU)
make train-cpu backbone=EfficientNetB0 epochs=20

# WSL2 (Recommended)
make train-wsl backbone=MobileNetV3Small epochs=5
make train-wsl backbone=EfficientNetB0 epochs=20 dataset=seccam_2

# Parameters:
#   backbone: EfficientNetB0 | MobileNetV3Small
#   epochs: number of training epochs
#   dataset: dataset name (default: seccam_2)
```

### Evaluation
```bash
make evaluate           # Standard evaluation
make benchmark          # Run benchmark suite
```

### Development
```bash
make test              # Run tests
make lint              # Run linter
make type-check        # Run type checker
make clean             # Clean cache files
```

## Training Example

```bash
# Train MobileNetV3Small on GPU (recommended)
make train-wsl backbone=MobileNetV3Small epochs=20

# Expected output:
Dataset: seccam_2, Classes: 15
Found 3540 images in train set
Found 780 images in val set

Training completed successfully!
Model: mobilenetv3small_seccam_2_gpu_final.keras
Final validation accuracy: 1.0000
Total training time: 514.65s (8.58m)
Model size: 14.85 MB
```

## Model Performance Summary

| Model | Platform | Accuracy | Size | Training Time (20 epochs) | Best For |
|-------|----------|----------|------|---------------------------|----------|
| MobileNetV3Small | GPU | 100% | 14.85MB | ~9 min | Fast training |
| MobileNetV3Small | CPU | 100% | 14.85MB | ~35 min | CPU-only systems |
| EfficientNetB0 | CPU | ~76.67% | 19.8MB | ~45-60 min | Research |
| EfficientNetB0 Quantized | CPU | ~75% | 5.0MB | N/A | **Production** |
| FaceNet | Pretrained | N/A | 88MB | N/A | Baseline |

**GPU Acceleration:**
- **Speedup**: 3-4x faster than CPU
- **Requirements**: NVIDIA GPU (GTX 1650+), 4GB+ VRAM
- **Setup**: Run `make setup-wsl-gpu` for automated configuration

## Camera Setup & Usage

### Prerequisites
- A webcam (built-in or USB-connected)
- Or use your phone as a webcam via USB (e.g., using your phone's camera app in "USB camera" mode)

### Environment Variables

Configure camera via environment variables:

```bash
# Camera source type: webcam (or rtsp for network streams)
set CAMERA_SOURCE=webcam

# Webcam device index (default: 0)
set CAMERA_DEVICE=0

# RTSP stream URL (for network cameras)
set CAMERA_RTSP_URL=rtsp://192.168.1.100:8080/live.sdp

# Video resolution (default: 1280x720)
set CAMERA_WIDTH=1280
set CAMERA_HEIGHT=720

# Frame rate (default: 30)
set CAMERA_FPS=30
```

### Camera Commands

```bash
# View camera stream (quick test)
make camera-view

# Run full face recognition app with camera
make run

# Run camera unit tests
make test-camera

# Run camera integration tests
make test-camera-integration
```

### Using Your Phone as Webcam

If your phone is connected via USB and detected as a webcam (device index 0, 1, etc.):

1. Connect your phone via USB
2. Enable "USB camera" or "Camera" mode on your phone
3. Set the device index:
   ```bash
   set CAMERA_DEVICE=0
   make camera-view
   ```

### Troubleshooting

**Camera not opening:**
- Check device index: try `set CAMERA_DEVICE=1`
- Check if another app is using the camera
- Ensure no other video conferencing apps are running

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'bp_face_recognition'**
- Solution: Use Makefile commands which set PYTHONPATH automatically
- Or run: `export PYTHONPATH=src:$PYTHONPATH`

**2. dlib compilation fails on Windows**
- Install Visual Studio Build Tools or use pre-built wheel
- Or skip dlib by installing dependencies manually without it

**3. TensorFlow GPU not detected in WSL**
- Ensure NVIDIA drivers are installed on Windows host
- Run GPU setup: `make setup-wsl-gpu`
- Or manually install:
  ```bash
  sudo apt install nvidia-cuda-toolkit
  pip uninstall tensorflow-cpu -y
  pip install tensorflow==2.15.0
  pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
  ```

**4. Out of memory during training**
- Reduce batch size in production_trainer.py
- Use smaller model (MobileNetV3Small vs EfficientNetB0)
- Close other applications

### Dataset Setup

Prepare your dataset:
```bash
# Initialize dataset structure
python src/scripts/init_dataset.py --name seccam_2

# Add images to data/cropped/seccam_2/train/
# Run preprocessing
python src/scripts/update_pipeline.py
```

## Cross-Platform Notes

### Windows
- Uses CPU-optimized packages (onnxruntime)
- dlib installation requires Visual Studio Build Tools
- Paths use forward slashes (/) in Makefile for WSL compatibility

### WSL2
- Better performance for training
- GPU acceleration available with proper setup
- Can access Windows files via /mnt/d/

## License
Distributed under MIT License. See `LICENSE` for more information.

## Contact
Yurii Murha - [GitHub](https://github.com/yourusername)
