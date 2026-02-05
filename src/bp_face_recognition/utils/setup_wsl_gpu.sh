#!/bin/bash

# WSL GPU Setup Script for BP Face Recognition
# This script automates GPU setup in WSL2 Ubuntu

set -e

echo "=== WSL GPU Setup for BP Face Recognition ==="
echo "This script will install CUDA, cuDNN, and Python dependencies"
echo

# Check if running in WSL
if ! grep -q Microsoft /proc/version; then
    echo "Warning: This script is designed for WSL2 environment"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check NVIDIA GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA GPU drivers first."
    exit 1
fi

echo "✓ NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
echo

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install basic dependencies
echo "Installing basic dependencies..."
sudo apt install -y \
    build-essential \
    curl \
    wget \
    git \
    python3 \
    python3-pip \
    python3-venv \
    software-properties-common

# Install CUDA
echo "Installing CUDA Toolkit..."
CUDA_VERSION=12.3
CUDA_REPO=cuda-repo-ubuntu2204-${CUDA_VERSION//./-}-local_12.3.2-545.23.06-1_amd64.deb

# Download CUDA repo package
if [ ! -f "/tmp/${CUDA_REPO}" ]; then
    echo "Downloading CUDA repository package..."
    wget -P /tmp https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/${CUDA_REPO}
fi

# Install CUDA repo
sudo dpkg -i /tmp/${CUDA_REPO}
sudo cp /var/cuda-repo-ubuntu2204-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update

# Install CUDA toolkit
sudo apt-get -y install cuda-toolkit-12-3

# Install cuDNN (requires manual download)
echo "=== cuDNN Installation ==="
echo "cuDNN requires manual download from NVIDIA Developer Portal"
echo "Please download cuDNN 8.9 for CUDA 12.x from:"
echo "https://developer.nvidia.com/cudnn"
echo
echo "After downloading, extract and run:"
echo "sudo cp cuda/include/cudnn*.h /usr/local/cuda/include"
echo "sudo cp cuda/lib/libcudnn* /usr/local/cuda/lib64"
echo "sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*"
echo

read -p "Press Enter to continue after cuDNN installation..." -r

# Set environment variables
echo "Setting up environment variables..."
if ! grep -q "CUDA_HOME" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA Environment Variables" >> ~/.bashrc
    echo "export PATH=/usr/local/cuda/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
fi

# Export for current session
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

# Verify CUDA installation
echo "Verifying CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA Compiler version:"
    nvcc --version | grep release
else
    echo "⚠ CUDA compiler not found in PATH"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
python3 -m pip install --upgrade pip

# Install TensorFlow with GPU support
echo "Installing TensorFlow with GPU support..."
python3 -m pip install tensorflow[and-cuda]

# Install MediaPipe with GPU support
echo "Installing MediaPipe with GPU support..."
python3 -m pip install mediapipe-gpu

# Install PyOpenGL for MediaPipe
echo "Installing PyOpenGL for MediaPipe..."
python3 -m pip install PyOpenGL PyOpenGL-accelerate

# Navigate to project directory
PROJECT_DIR="/mnt/d/Coding/Personal/BP-face-recognition"
if [ -d "$PROJECT_DIR" ]; then
    echo "Installing project dependencies..."
    cd "$PROJECT_DIR"
    pip install -e .
else
    echo "Project directory not found: $PROJECT_DIR"
    echo "Please update PROJECT_DIR variable in this script"
fi

# Create verification script
echo "Creating GPU verification script..."
cat > ~/gpu_verification.py << 'EOF'
#!/usr/bin/env python3
import tensorflow as tf
import mediapipe as mp
import numpy as np

print("=== GPU Verification ===")
print(f"TensorFlow version: {tf.__version__}")
print(f"MediaPipe version: {mp.__version__}")

# TensorFlow GPU Test
gpus = tf.config.list_physical_devices('GPU')
print(f"TensorFlow GPUs found: {len(gpus)}")
for gpu in gpus:
    print(f"  - {gpu}")

print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# MediaPipe GPU Test
try:
    base_options = mp.tasks.BaseOptions(
        delegate=mp.tasks.BaseOptions.Delegate.GPU
    )
    detector = mp.tasks.vision.FaceDetector.create_from_options(
        mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5,
        )
    )
    
    # Test detection
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=test_image,
    )
    result = detector.detect(mp_image)
    print("✓ MediaPipe GPU delegate working successfully")
    detector.close()
except Exception as e:
    print(f"✗ MediaPipe GPU delegate failed: {e}")

# Test TensorFlow GPU computation
if gpus:
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print("✓ TensorFlow GPU computation successful")
            print(f"  Result shape: {c.shape}")
    except Exception as e:
        print(f"✗ TensorFlow GPU computation failed: {e}")

print("=== Verification Complete ===")
EOF

chmod +x ~/gpu_verification.py

echo
echo "=== Setup Complete ==="
echo "Please restart your shell or run 'source ~/.bashrc' to load environment variables"
echo
echo "Run verification with: ~/gpu_verification.py"
echo
echo "Expected performance improvements:"
echo "  - MediaPipe face detection: 5-10x faster"
echo "  - TensorFlow inference: 10-20x faster"
echo