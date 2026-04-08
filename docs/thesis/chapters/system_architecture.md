# System Architecture and Design Decisions

## Chapter X: Software Design and Implementation

---

## 1. Introduction

This chapter describes the software architecture of the face recognition system. The system is designed to support multiple detection and recognition backends, two distinct recognition paradigms (open-set and closed-set), and cross-platform deployment (Windows CPU, WSL2 GPU). The architecture prioritizes extensibility — adding a new detection or recognition model requires only a configuration change, not code modification.

The system is implemented in Python 3.11 using TensorFlow 2.15 for deep learning, OpenCV for image processing, and a layered service architecture for separation of concerns.

---

## 2. High-Level Architecture

### 2.1 Layered Design

The system follows a five-layer architecture:

```
Entry Points         main.py (open-set) | closed_set_main.py (closed-set)
     |
Service Layer        PipelineService | ClosedSetPipelineService | DatabaseService
     |
Vision Core          FaceTracker | FaceDetector (ABC) | FaceRecognizer (ABC)
     |
Plugin System        ModelRegistry (models.yaml) -> RecognizerFactory
     |
Implementations      Detection: MediaPipe, MTCNN, Haar, Dlib HOG, face_recognition
                     Recognition: FaceNet (TL/PU/TLoss), EfficientNetB0, Dlib, TFLite
```

Each layer depends only on the layer directly below it. Entry points depend on services, services depend on vision core abstractions, and the plugin system maps configuration to concrete implementations.

### 2.2 Dual Recognition Paradigms

The system supports two fundamentally different recognition approaches through the same detection infrastructure:

```
Camera Frame
     |
Face Detection (shared)
     |
     +--- Open-Set Path ---+        +--- Closed-Set Path ---+
     |                     |        |                        |
     | get_embedding()     |        | recognize()            |
     | -> 512D vector      |        | -> argmax(softmax)     |
     |                     |        | -> (identity, conf)    |
     | DatabaseService     |        |                        |
     | cosine similarity   |        | Confidence threshold   |
     | vs registered faces |        | >= thresh -> ID        |
     |                     |        | < thresh -> Unknown    |
     +--- Result ----------+        +--- Result -------------+
```

This shared-infrastructure design avoids code duplication while allowing each paradigm to optimize its recognition pipeline independently.

---

## 3. Core Abstractions

### 3.1 FaceDetector Interface

All detection backends implement the `FaceDetector` abstract base class defined in `vision/interfaces.py`:

```python
class FaceDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces and return bounding boxes as (x, y, w, h)."""
        pass
```

Five concrete implementations exist: MediaPipe BlazeFace, MTCNN, Haar Cascade, Dlib HOG, and the `face_recognition` library. Each is a drop-in replacement — the service layer interacts only with the abstract interface.

### 3.2 FaceRecognizer Interface

Recognition backends implement the `FaceRecognizer` abstract base class:

```python
class FaceRecognizer(ABC):
    @abstractmethod
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract embedding vector from a cropped face image."""
        pass
```

The `FinetunedRecognizer` extends this interface with a `recognize()` method for closed-set classification, returning `(identity, confidence)` directly from the softmax output.

### 3.3 Design Decision: Abstract Base Classes

We chose Python ABCs over duck typing or Protocol classes for several reasons:

| Consideration | ABC Approach | Duck Typing |
|---------------|-------------|-------------|
| Method contract | Explicitly enforced | Implicit |
| Error detection | At instantiation time | At call time |
| IDE support | Full autocomplete | Limited |
| Documentation | Self-documenting | Requires docstrings |

The trade-off is additional boilerplate per implementation, but for a system with 10+ concrete implementations, the explicit contract provides significant maintainability benefits.

---

## 4. Plugin System

### 4.1 Configuration-Driven Model Loading

All detector and recognizer models are registered in `config/models.yaml`:

```yaml
recognizers:
  facenet_pu:
    class: "vision.recognition.finetuned_recognizer.FinetunedRecognizer"
    model_file: "bp_face_recognition/models/finetuned/facenet_progressive_v1.0.keras"
    version: "1.0"
    description: "FaceNet Progressive Unfreezing (99.15%)"
```

The `ModelRegistry` (`vision/registry.py`) parses this configuration and performs dynamic import and instantiation:

1. Read `models.yaml` configuration
2. Resolve the class path (e.g., `vision.recognition.finetuned_recognizer.FinetunedRecognizer`)
3. Dynamically import the module
4. Instantiate the class with the specified model file path

### 4.2 RecognizerFactory

The `RecognizerFactory` (`vision/factory.py`) wraps the registry with convenience features:

- **Legacy name fallback**: Maps older model names to current registry entries
- **Default model resolution**: Uses the `global.default_recognizer` configuration key (currently `facenet_pu`)
- **Environment-specific profiles**: Selects appropriate models based on detected platform (Windows/WSL) and available hardware (CPU/GPU)

### 4.3 Environment Profiles

The system defines 8 environment profiles in `models.yaml`, optimized for different deployment scenarios:

| Profile | Platform | Use Case | Default Detector | Default Recognizer |
|---------|----------|----------|-------------------|-------------------|
| dev_windows | Windows | Development | mediapipe | facenet_pu |
| prod_windows | Windows | Production | mediapipe | facenet_pu |
| dev_wsl | WSL2 | Development | mediapipe | facenet_pu |
| prod_wsl | WSL2 | Production | mediapipe | facenet_pu |
| benchmark | Any | Evaluation | all | all |

### 4.4 Design Decision: Configuration over Code

The YAML-based plugin system was chosen over hardcoded model imports:

**Advantages**:
- Adding a new model requires only a YAML entry, no code changes
- Model switching at runtime without recompilation
- Environment-specific defaults without conditional logic in code
- Single source of truth for all model metadata (paths, versions, descriptions)

**Trade-off**: Configuration errors (typos in class paths, missing model files) produce runtime errors rather than import-time errors. This is mitigated by the registry's validation on startup.

---

## 5. Service Layer

### 5.1 PipelineService (Open-Set Recognition)

`PipelineService` (`services/pipeline_service.py`) orchestrates the full open-set recognition pipeline:

1. Receive camera frame
2. Delegate to `FaceTracker.track_faces()` for detection at 50% resolution
3. For each detected face, extract embedding at full resolution via `FaceRecognizer.get_embedding()`
4. Query `DatabaseService.recognize_face()` for cosine similarity matching against registered faces
5. Return `(identity, confidence)` if similarity exceeds threshold, else "Unknown"
6. Track performance statistics (FPS, detection count, recognition latency)

The service also handles face registration: capturing multiple face samples, extracting embeddings, and storing them in the database.

### 5.2 ClosedSetPipelineService

`ClosedSetPipelineService` (`services/closed_set_pipeline_service.py`) implements the simpler closed-set pipeline:

1. Detect faces at 50% resolution
2. For each face, call `FinetunedRecognizer.recognize(face_crop)` directly
3. Model outputs softmax probabilities over 14 classes
4. Return `argmax` identity if confidence exceeds threshold, else "Unknown"

This service has no database dependency — identities are encoded in the model weights.

### 5.3 DatabaseService

`DatabaseService` (`services/database_service.py`) provides an abstraction for face embedding storage and matching:

- **Storage**: CSV-based (`data/faces.csv`) with columns for identity name and 512-dimensional embedding vector
- **Matching**: Cosine similarity between query embedding and all registered embeddings
- **Registration**: Store new identity with multiple embedding samples
- **PostgreSQL**: Interface exists for future PostgreSQL backend

The CSV-based approach was chosen for simplicity appropriate to the thesis scope, avoiding database server dependencies while supporting the full registration and recognition workflow.

---

## 6. Preprocessing Pipeline

The data preprocessing pipeline transforms raw images into training-ready datasets through three stages:

### 6.1 Face Cropping (`preprocessing/crop_faces.py`)

- Uses MediaPipe face detection to locate faces in raw images
- Crops to 160x160 pixels (FaceNet input size)
- Saves with naming convention: `{label}_{uuid}.jpg`

### 6.2 Dataset Splitting (`preprocessing/split_lfw.py`)

- Splits datasets into train/validation/test sets (70/15/15%)
- Stratified splitting to maintain class proportions
- Supports both custom datasets and LFW format

### 6.3 Data Augmentation (`preprocessing/augmentation.py`)

- Uses the Albumentations library for augmentation
- Augmentations: horizontal flip, brightness/contrast adjustment, slight rotation
- Augmented images saved with `.{N}.jpg` suffix (e.g., `Yurii_f3d9f09a.0.jpg`)
- Configurable augmentation factor (default: 10x)

The flat directory structure (all images in one folder per split, no nested identity folders) simplifies data loading and enables consistent naming across all dataset sources.

---

## 7. Training Infrastructure

### 7.1 Three Training Paradigms

The system supports three distinct training approaches, each in a dedicated module under `vision/training/`:

| Paradigm | Module | Output | Use Case |
|----------|--------|--------|----------|
| Classifier | `classifier/trainer.py` | Softmax model (.keras) | Closed-set recognition |
| Metric Learning | `metric/trainer.py` | Embedding model (.keras) | Open-set recognition |
| FaceNet Fine-tuning | `finetune/` | 3 strategy variants | Both paradigms |

### 7.2 Cross-Platform Training

Training is supported on both Windows (CPU) and WSL2 (GPU):

- **Windows**: CPU-only via `tensorflow-cpu`. Suitable for quick experiments and small datasets.
- **WSL2**: Full GPU acceleration via NVIDIA CUDA 12.2 + cuDNN 8.9. Required for production training (3-4x speedup).

The Makefile provides unified commands (`make train-wsl`, `make train-cpu`) that set appropriate environment variables (`XLA_FLAGS`, `PYTHONPATH`) and activate the correct virtual environment.

### 7.3 Model Artifacts

Trained models are stored in standardized formats:
- `.keras`: Full TensorFlow/Keras model format (weights + architecture + optimizer state)
- `.tflite`: TensorFlow Lite quantized format for edge deployment

All fine-tuned FaceNet models reside in `src/bp_face_recognition/models/finetuned/` alongside `dataset_info.json` containing class name mappings.

---

## 8. Cross-Platform Support

The project supports dual-platform development through several mechanisms:

### 8.1 Dependency Management

- **Package Manager**: `uv` for fast, deterministic dependency resolution
- **Platform Markers**: `pyproject.toml` uses `platform_system` markers to install `tensorflow` on Linux and `tensorflow-cpu` on Windows
- **Shared Lock File**: Single `uv.lock` works across both platforms via conditional dependencies
- **Separate Virtual Environments**: `.venv-win` (Windows) and `.venv-wsl` (WSL2) to avoid binary incompatibilities

### 8.2 Build Automation

The Makefile provides platform-aware commands:
- `make run` / `make run-closed-set`: Entry points for both paradigms
- `make train-wsl` / `make train-cpu`: Platform-specific training with correct environment setup
- `make thesis-benchmark`: Unified benchmark suite generating all comparison data

---

## 9. Summary of Design Decisions

| Decision | Choice | Alternative Considered | Rationale |
|----------|--------|----------------------|-----------|
| Model loading | Config-driven (YAML) | Hardcoded imports | Extensibility without code changes |
| Interfaces | Abstract Base Classes | Protocol/duck typing | Explicit contracts, IDE support |
| Embedding storage | CSV + cosine similarity | SQLite, FAISS | Simplicity for thesis scope |
| Training environment | WSL2 with native CUDA | Docker, cloud (AWS/GCP) | Local GPU access, zero cost |
| Package manager | uv | pip, Poetry | Speed, reliable lockfile |
| Recognition paradigms | Dual (open + closed) | Single paradigm | Thesis completeness, comparison |
| Detection default | MediaPipe | MTCNN | 100x faster, sufficient for real-time |
| FaceNet backbone | InceptionResNetV1 | ResNet50, MobileNet | Pre-trained on faces, 512D embeddings |
| Augmentation | Albumentations | tf.image, imgaug | Performance, pipeline integration |

---

## References

1. Abadi, M., et al. (2016). TensorFlow: A system for large-scale machine learning. *OSDI*, 265-283.

2. Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.

3. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *CVPR*, 815-823.

---

**Chapter Status**: Complete

**Last Updated**: March 21, 2026

**Word Count**: ~2,500 words
