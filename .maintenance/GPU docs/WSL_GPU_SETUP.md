# WSL GPU Setup Guide for MediaPipe and TensorFlow

## Overview

This guide explains how to set up GPU acceleration for MediaPipe and TensorFlow in Windows Subsystem for Linux (WSL2). Since Windows native TensorFlow/MediaPipe GPU support is limited, WSL2 provides the best GPU acceleration for face recognition tasks.

## Prerequisites

### Hardware Requirements
- **NVIDIA GPU** with CUDA support (GTX 10xx series or newer)
- **Windows 10/11** with WSL2 enabled
- **At least 8GB RAM** (16GB+ recommended for face recognition)

### Software Requirements
- **WSL2** with Ubuntu 20.04/22.04
- **NVIDIA GPU drivers** (latest)
- **CUDA Toolkit** 11.8+ or 12.x
- **cuDNN** 8.6+ (compatible with CUDA version)

## Setup Instructions

### 1. Install WSL2 and Ubuntu

```powershell
# Enable WSL
wsl --install

# Set WSL2 as default
wsl --set-default-version 2

# Install Ubuntu (if not already installed)
wsl --install -d Ubuntu-22.04
```

### 2. Install NVIDIA GPU Drivers

1. Download and install latest NVIDIA drivers for Windows
2. Verify GPU passthrough to WSL:

```bash
# In WSL Ubuntu
nvidia-smi
# Should show your GPU information
```

### 3. Install CUDA Toolkit in WSL

```bash
# Update package list
sudo apt update

# Install CUDA (choose one method)

# Method A: CUDA 12.x (Recommended)
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda-repo-ubuntu2204-12-3-local_12.3.2-545.23.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-3-local_12.3.2-545.23.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3

# Method B: Using Ubuntu packages (simpler)
sudo apt install nvidia-cuda-toolkit
```

### 4. Install cuDNN

```bash
# Download cuDNN (requires NVIDIA developer account)
# Extract and copy to CUDA directory
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### 5. Set Environment Variables

```bash
# Add to ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc

# Reload shell
source ~/.bashrc
```

### 6. Install Python Dependencies

```bash
# Update pip
python3 -m pip install --upgrade pip

# Install TensorFlow with GPU support
python3 -m pip install tensorflow[and-cuda]

# Install MediaPipe with GPU support
python3 -m pip install mediapipe-gpu

# Install other project dependencies
cd /mnt/d/Coding/Personal/BP-face-recognition
pip install -e .
```

## Verification

### Test TensorFlow GPU

```python
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())

# Test GPU computation
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print("GPU Computation Result:", c.numpy())
```

### Test MediaPipe GPU

```python
import mediapipe as mp
import numpy as np

# Test GPU delegate creation
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
print("MediaPipe GPU detector working:", len(result.detections))
detector.close()
```

## Performance Expectations

### MediaPipe Face Detection Performance
- **CPU**: ~50-100ms per 640x480 image
- **GPU (WSL)**: ~5-15ms per 640x480 image (5-10x speedup)

### TensorFlow Inference Performance
- **CPU**: ~200-500ms for face embedding models
- **GPU (WSL)**: ~10-50ms for face embedding models (10-20x speedup)

## Troubleshooting

### Common Issues

1. **"CUDA not found" errors**
   - Verify CUDA installation: `nvcc --version`
   - Check LD_LIBRARY_PATH includes CUDA libs

2. **"GPU delegate not available"**
   - Test with `nvidia-smi` in WSL
   - Verify OpenGL/EGL support: `glxinfo -B`

3. **"Out of memory" errors**
   - Enable TensorFlow memory growth:
     ```python
     gpus = tf.config.experimental.list_physical_devices('GPU')
     if gpus:
         tf.config.experimental.set_memory_growth(gpus[0], True)
     ```

4. **Slow performance despite GPU**
   - Check GPU utilization: `watch -n 1 nvidia-smi`
   - Verify model is using GPU delegate
   - Monitor thermal throttling

### Performance Testing Script

```python
import time
import numpy as np
from src.bp_face_recognition.models.methods.mediapipe_detector import MediaPipeDetector

# Test performance
detector = MediaPipeDetector(auto_gpu_detection=True)
print("GPU Status:", detector.get_gpu_status()['initialization_method'])

# Benchmark
test_images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
               for _ in range(100)]

start_time = time.time()
for img in test_images:
    boxes = detector.detect(img)
end_time = time.time()

avg_time = (end_time - start_time) / len(test_images)
fps = 1.0 / avg_time

print(f"Average detection time: {avg_time*1000:.2f}ms")
print(f"Detection FPS: {fps:.1f}")
```

## Integration with BP Face Recognition

The enhanced MediaPipe detector automatically detects GPU availability in WSL:

```python
from src.bp_face_recognition.models.factory import RecognizerFactory

# This will use GPU if available in WSL
detector = RecognizerFactory.get_optimized_detector()

# Check actual initialization
status = detector.get_gpu_status()
print("Using:", status['initialization_method'])
print("GPU Compatible:", status['gpu_info']['mediapipe_gpu_compatible'])
```

## Additional Resources

- [WSL2 GPU Installation Guide](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [MediaPipe GPU Setup](https://google.github.io/mediapipe/solutions/gpu.html)
- [TensorFlow GPU Setup](https://www.tensorflow.org/install/gpu)

## Next Steps

1. **Set up WSL2 environment** following this guide
2. **Run verification tests** to ensure GPU acceleration
3. **Benchmark performance** vs Windows CPU
4. **Test face recognition pipeline** with GPU acceleration
5. **Monitor performance** and optimize as needed