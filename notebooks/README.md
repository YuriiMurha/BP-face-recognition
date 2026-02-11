# Jupyter Notebooks

These notebooks were used for initial exploration, data visualization, and prototyping.

**Note:** The core logic from these notebooks has been refactored into modular Python scripts in the `src/` directory with the new vision architecture to support better maintainability, testing, and automation.

## Migration Mapping

### Data Processing & Training
- **Preprocessing.ipynb** -> `src/data/augmentation.py` (Data augmentation) & `src/utils/crop_faces.py` (Cropping)
- **DeepLearning.ipynb** -> `src/vision/training/dataset_loader.py` (Dataset loading) & `src/vision/recognition/custom_cnn.py` (Custom CNN model)

### Detection & Recognition Methods
- **FaceDetection.ipynb** -> `src/vision/detection/` (Multiple detection methods)
  - `mediapipe.py` (MediaPipe BlazeFace detector)
  - `haar_cascade.py` (OpenCV Haar Cascade)
  - `mtcnn.py` (MTCNN detector)
  - `dlib_hog.py` (Dlib HOG detector)
- **FaceRecognition.ipynb** -> `src/vision/recognition/` (Multiple recognition methods)
  - `facenet.py` (FaceNet recognizer)
  - `custom_cnn.py` (Custom CNN recognizer)
  - `tflite.py` (TensorFlow Lite optimized recognizer)

### Model Management & Configuration
- **Model Training** -> `src/vision/training/trainer.py` (Enhanced training with multi-architecture support)
- **Model Configuration** -> `config/models.yaml` (Configuration-driven model registry)
- **Model Factory** -> `src/vision/registry.py` (Dynamic plugin loading system)

### Core Application Logic
- **Face Tracking** -> `src/vision/core/face_tracker.py` (Enhanced with plugin system)
- **Recognition Service** -> `src/vision/core/recognition_service.py` (Headless workflow)
- **Database Operations** -> `src/services/database_service.py` (Business logic abstraction)

## New Vision Architecture Features

### 🏗️ **Plugin System**
- **Configuration-Driven**: All models configured in `config/models.yaml`
- **Runtime Switching**: Change detectors/recognizers without code changes
- **Version Support**: Model versioning and experimental feature flags
- **Environment Profiles**: Production, development, testing, WSL GPU configurations

### 📊 **Performance Optimization**
- **GPU Acceleration**: MediaPipe GPU support with intelligent fallback
- **Model Quantization**: TensorFlow Lite optimization for 2-3x speedup
- **Batch Processing**: Optimized for real-time video processing
- **Cross-Platform**: Automatic GPU/CPU configuration per platform

### 🔧 **Production Features**
- **Error Handling**: Comprehensive validation and graceful fallbacks
- **Logging**: Detailed performance metrics and debugging information
- **Services Layer**: Clean business logic abstraction
- **Model Management**: Proper organization of downloaded vs custom models

## Usage Examples

### Basic Model Loading
```python
from src.bp_face_recognition.vision.registry import get_registry

# Load registry with configuration
registry = get_registry()

# Get default detector and recognizer
detector = registry.get_default_detector()
recognizer = registry.get_default_recognizer()
```

### Environment-Specific Configuration
```python
# Load WSL GPU optimized configuration
registry = get_registry()
detector = registry.get_detector('mediapipe_v1', use_gpu=True)
```

### Custom Model Integration
```python
# Add custom model to configuration
# Edit config/models.yaml to add new entries
# Models automatically loaded via plugin system
```

Please refer to the scripts in `src/vision/` for the production-ready code. These notebooks remain for reference and experimental visualization.
