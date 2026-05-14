# Chapter 6: Practical Implementation and Codebase Overview

This chapter describes the software implementation of the face recognition system. The system was designed to support multiple detection and recognition backends, two distinct recognition modes (open-set and closed-set), and cross-platform deployment across Windows (CPU) and WSL2 (GPU). The architecture prioritizes extensibility: adding a new detection or recognition model requires only a configuration entry, not a code change.

The system was implemented in Python 3.11 using TensorFlow 2.15 for deep learning, OpenCV for image processing, and a layered service architecture for separation of concerns.

## 6.1 Codebase Structure and Organization

### 6.1.1 Project Layout

The project follows the `src`-layout convention recommended by the Python Packaging Authority [CITE: python packaging src-layout], where all source code resides under a single top-level `src/` directory. This layout prevents accidental imports from the working directory and enforces that tests run against the installed package rather than local files.

```
BP-face-recognition/
├── src/bp_face_recognition/ # All production source code
│ ├── main.py # Open-set entry point (CLI app)
│ ├── closed_set_main.py # Closed-set entry point
│ ├── config/settings.py # Pydantic-based configuration
│ ├── services/ # Service layer (orchestration)
│ ├── vision/ # Detection, recognition, training
│ ├── preprocessing/ # Data pipeline (crop, split, augment)
│ └── models/finetuned/ # Trained model artifacts
├── config/models.yaml # Plugin registry configuration
├── data/ # Datasets, logs, face database
├── noxfile.py # Test and lint session definitions
├── pyproject.toml # Project metadata and dependencies
└── Makefile # Build automation commands
```

The `PYTHONPATH=src` environment variable is required for module resolution; the Makefile sets this automatically for all commands.

### 6.1.2 Key Directories and Their Responsibilities

The codebase is organized into functional modules with clear boundaries:

| Directory | Responsibility |
|-----------|---------------|
| `services/` | Business logic orchestration (pipeline, database) |
| `vision/interfaces.py` | Abstract base classes for detection and recognition |
| `vision/detection/` | Five detector implementations (MediaPipe, MTCNN, Haar, Dlib, face_recognition) |
| `vision/recognition/` | Recognizer implementations (FaceNet variants, Dlib, TFLite, metric learning) |
| `vision/training/` | Three training approaches (classifier, metric, fine-tuning) |
| `vision/core/` | Integration classes (`FaceTracker`) that combine detection with recognition |
| `vision/registry.py` | Configuration-driven dynamic model loading |
| `preprocessing/` | Data pipeline: face cropping, dataset splitting, augmentation |
| `config/` | Settings management and model registry YAML |

This separation ensures that detection algorithms, recognition models, and orchestration logic can evolve independently.

## 6.2 System Architecture

### 6.2.1 Layered Architecture Overview

The system follows a five-layer architecture where each layer depends only on the layer directly below it:

```
┌─────────────────────────────────────────────────────────┐
│ Entry Points │
│ main.py (open-set) │ closed_set_main.py │
├─────────────────────────────────────────────────────────┤
│ Service Layer │
│ PipelineService │ ClosedSetPipelineService │
│ DatabaseService │ │
├─────────────────────────────────────────────────────────┤
│ Vision Core │
│ FaceTracker │ FaceDetector (ABC) │ FaceRecognizer │
├─────────────────────────────────────────────────────────┤
│ Plugin System │
│ ModelRegistry (models.yaml) → RecognizerFactory │
├─────────────────────────────────────────────────────────┤
│ Implementations │
│ Detection: MediaPipe, MTCNN, Haar, Dlib, face_rec │
│ Recognition: FaceNet (TL/PU/TLoss), Dlib, TFLite │
└─────────────────────────────────────────────────────────┘
```

Entry points depend on services; services depend on vision core abstractions; and the plugin system maps YAML configuration to concrete implementation classes. This layering means, for example, that `PipelineService` never imports a specific detector class directly --- it receives one through the factory.

### 6.2.2 Service Layer Design

The service layer contains three classes, each responsible for a distinct workflow:

**`PipelineService`** (`services/pipeline_service.py`) orchestrates the open-set recognition pipeline. For each camera frame, it delegates detection to `FaceTracker.track_faces()`, which runs detection at 50% resolution for speed and extracts embeddings at full resolution for accuracy. The service then queries `DatabaseService.recognize_face()` to match the embedding against registered identities using cosine similarity. It also handles person registration by capturing multiple face samples, extracting embeddings, and storing them in the database.

**`ClosedSetPipelineService`** (`services/closed_set_pipeline_service.py`) implements the closed-set classification path. Rather than querying a database, it calls `FinetunedRecognizer.recognize(face_crop)` directly, which outputs softmax probabilities over a fixed set of 14 classes. The identity with the highest probability is returned if it exceeds a confidence threshold. This service has no database dependency --- identities are encoded in the model weights.

**`DatabaseService`** (`services/database_service.py`) abstracts face embedding storage and retrieval. It wraps the `FaceDatabase` class and provides methods for registration (`register_person`), recognition (`recognize_face`), and management (`delete_person`, `list_all_people`). The recognition method computes dot-product similarity between L2-normalized embeddings, converting similarity scores to distance values for threshold comparison.

### 6.2.3 Dual Recognition Modes

A key architectural decision was to support both open-set and closed-set recognition through the same detection infrastructure, avoiding code duplication while allowing each paradigm to optimize independently:

```
Camera Frame
 │
Face Detection (shared across both paths)
 │
 ├─── Open-Set Path ──────────┐ ├─── Closed-Set Path ────────┐
 │ get_embedding() │ │ recognize() │
 │ → 512D vector │ │ → argmax(softmax) │
 │ DatabaseService │ │ → (identity, confidence) │
 │ cosine similarity match │ │ threshold check │
 └────────────────────────────┘ └─────────────────────────────┘
```

The open-set path can identify anyone who has been registered (even after training), while the closed-set path is limited to identities present during training but requires no registration step. Both paths share the same detector, and the choice between them is made at the entry-point level (`main.py` vs. `closed_set_main.py`).

### 6.2.4 Configuration Management

Configuration is handled at two levels. Application-level settings are managed by Pydantic Settings (`config/settings.py`), which provides typed access to paths, camera parameters, and database credentials with automatic loading from environment variables and `.env` files:

```python
class Settings(BaseSettings):
 ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
 MODELS_DIR: Path = SRC_DIR / "bp_face_recognition" / "models"
 CAMERA_SOURCE: str = "webcam"
 CAMERA_WIDTH: int = 1280
 DB_HOST: str = "localhost"

 model_config = SettingsConfigDict(env_file=".env", extra="ignore")
```

Model-level configuration is managed through the YAML-based plugin registry described in Section 6.2.5. The separation between application settings (Pydantic) and model configuration (YAML) was deliberate: application settings rarely change across environments, while model selection changes frequently during development and benchmarking.

The decision to drive model selection from YAML rather than from hardcoded imports trades import-time safety for runtime flexibility. Adding a new model becomes a single YAML entry with no code change, and per-environment defaults override centrally without conditional logic. The cost is that configuration mistakes — typos in class paths, missing files — surface as runtime errors rather than import errors; the registry mitigates this with startup validation that fails fast on unresolvable references.

## 6.3 Detection Pipeline

### 6.3.1 Plugin Interface Design

All detection backends implement the **`FaceDetector`** abstract base class defined in `vision/interfaces.py`:

```python
class FaceDetector(ABC):
 @abstractmethod
 def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
 """Detect faces and return bounding boxes as (x, y, w, h)."""
 pass

 def detect_with_confidence(self, image):
 """Detect faces with confidence scores. Defaults to 1.0."""
 boxes = self.detect(image)
 return [(box, 1.0) for box in boxes]
```

Python ABCs were chosen over Protocol classes or duck typing for the plugin interfaces. The trade-off is additional boilerplate per implementation, but for a system with ten or more concrete implementations, the explicit contract provides tangible benefits:

| Consideration | ABC Approach | Duck Typing |
|---------------|-------------|-------------|
| Method contract | Explicitly enforced at instantiation | Implicit, errors at call time |
| IDE support | Full autocomplete and type checking | Limited |
| Documentation | Self-documenting via abstract methods | Requires external docstrings |
| Error timing | Immediate on missing methods | Deferred to runtime |

The `detect_with_confidence()` method has a default implementation that wraps `detect()` with a confidence of 1.0, so older detectors that do not report confidence scores remain compatible without modification.

### 6.3.2 Available Detectors

Five detection backends were implemented, each offering different trade-offs between speed, accuracy, and hardware requirements:

| Detector | Class | Speed | Detections (19 frames) | Notes |
|----------|-------|-------|----------------------|-------|
| MediaPipe BlazeFace | `MediaPipeDetector` | 3 ms (326 FPS) | 6 | Default; GPU-capable |
| MTCNN | `MTCNNDetector` | 240 ms (4 FPS) | 25 | Highest recall |
| Haar Cascade | `HaarCascadeDetector` | 31 ms (32 FPS) | 12 | Good speed/accuracy balance |
| Dlib HOG | `DlibHOGDetector` | 156 ms (6 FPS) | 10 | CPU-focused |
| face_recognition | `FaceRecognitionLibDetector` | varies | varies | Wraps dlib internally |

MediaPipe was chosen as the default detector because its 326 FPS throughput is sufficient for real-time processing, and it is the only detector in the system that supports GPU acceleration through the MediaPipe Tasks API [CITE: mediapipe]. The trade-off is that it produces fewer detections on surveillance-style frames compared to MTCNN, which employs a three-stage cascade but runs at only 4 FPS.

### 6.3.3 How Detection Plugins Are Loaded and Switched

All detectors are registered in `config/models.yaml` with their class paths, version numbers, and default parameters:

```yaml
detectors:
 mediapipe_v1:
 class: "vision.detection.mediapipe.MediaPipeDetector"
 model_file: "bp_face_recognition/vision/detection/models/blaze_face_short_range.tflite"
 version: "1.0"
 default_config:
 min_detection_confidence: 0.5
 use_gpu: false
 auto_gpu_detection: true
```

The `ModelRegistry` (`vision/registry.py`) loads this configuration, validates the structure (checking for required fields like `class` and `version`), and performs dynamic import via `importlib`:

```python
def _dynamic_import(self, class_path: str) -> Type:
 module_path, class_name = class_path.rsplit(".", 1)
 module = importlib.import_module(f"bp_face_recognition.{module_path}")
 return getattr(module, class_name)
```

The `RecognizerFactory` (`vision/factory.py`) wraps the registry with two convenience features: legacy name mapping (so that `"mediapipe"` resolves to `"mediapipe_v1"`) and a prioritized fallback chain (`get_optimized_detector()` tries MediaPipe, then MTCNN, then Haar Cascade). Switching detectors at runtime requires changing only the `detector_type` string passed to `PipelineService`.

### 6.3.4 Multi-Tier Fallback in MediaPipe

The `MediaPipeDetector` (`vision/detection/mediapipe.py`) implements a three-tier initialization strategy:

1. **GPU delegate**: If `auto_gpu_detection` is enabled, the detector attempts to create a MediaPipe face detector with GPU acceleration. A test detector is created and immediately closed to validate that the GPU delegate works.
2. **CPU delegate**: If GPU validation fails, the detector falls back to CPU-based MediaPipe inference using the same model file.
3. **OpenCV Haar Cascade**: If MediaPipe itself is unavailable (missing dependency or corrupted model file), the detector instantiates `HaarCascadeDetector` as a last resort.

This fallback chain ensures the system always has a working detector, regardless of the deployment environment.

## 6.4 Recognition Pipeline

### 6.4.1 FaceNet Embedding Extraction

The primary recognition model is FaceNet [CITE: schroff2015facenet], an InceptionResNetV1 architecture pre-trained on MS-Celeb-1M that maps face images to 512-dimensional embeddings. The pre-trained model achieves 99.6% accuracy on the LFW benchmark and is accessed through the `keras-facenet` package (`vision/recognition/facenet_keras.py`).

For fine-tuned models, the `FinetunedRecognizer` (`vision/recognition/finetuned_recognizer.py`) wraps Keras models with FaceNet-compatible preprocessing. Input images are resized to 160x160 pixels, converted to float32, and normalized to the [-1, 1] range:

```python
def preprocess_image(self, image):
 image = tf.image.resize(image, self.input_size).numpy()
 image = image.astype(np.float32)
 if image.max() > 1.0:
 image = image / 255.0
 image = (image - 0.5) * 2.0 # Scale to [-1, 1]
 return np.expand_dims(image, axis=0)
```

This preprocessing matches the convention used during FaceNet pre-training [CITE: schroff2015facenet], ensuring that fine-tuned models receive inputs in the same distribution as the original training data.

### 6.4.2 Three Training Strategies

Three fine-tuning strategies were implemented under `vision/training/finetune/`, each representing a different approach to adapting the pre-trained FaceNet backbone to the target domain of 14 identities (7,080 images from combined webcam and surveillance datasets):

**Option A: Transfer Learning** (`facenet_transfer_trainer.py`) freezes the entire InceptionResNetV1 base and trains only a classification head (Dense(256, ReLU) + Dropout(0.5) + Dense(14, Softmax)). With only 131,000 trainable parameters (0.56% of the network), training completes in approximately 4 minutes and achieves 92.84% test accuracy. This approach is the fastest but cannot adapt the feature extraction layers to the target domain.

**Option B: Progressive Unfreezing** (`facenet_progressive_trainer.py`) trains in four phases, each unfreezing additional layers and reducing the learning rate:

| Phase | Layers Unfrozen | Learning Rate | Epochs |
|-------|----------------|---------------|--------|
| 1 | Classification head only | 1e-3 | 5 |
| 2 | Top 20% of base layers | 1e-5 | 5 |
| 3 | Top 40% of base layers | 5e-6 | 5 |
| 4 | All layers | 1e-6 | 4 |

This strategy gradually adapts the network from high-level features down to lower-level ones, reducing the risk of catastrophic forgetting [CITE: catastrophic forgetting in fine-tuning]. Progressive unfreezing achieved **99.15%** test accuracy (F1 = 0.991) in approximately 50 minutes of GPU training. It was selected as the production default model (`facenet_pu` in `config/models.yaml`).

**Option C: Triplet Loss** (`facenet_triplet_trainer.py`) retrains the network using metric learning with a margin-based triplet loss [CITE: schroff2015facenet]. Unlike Options A and B, which use softmax classification, this approach directly optimizes the embedding space so that embeddings of the same identity cluster together while different identities are pushed apart. The triplet loss with margin 0.2 achieved 94.63% accuracy in 90 minutes. While this approach produces embeddings suitable for open-set recognition (new identities can be added without retraining), it was slower to converge and did not match progressive unfreezing on closed-set metrics.

### 6.4.3 Open-Set vs. Closed-Set Implementation

The distinction between open-set and closed-set recognition is reflected in the recognizer interfaces:

- **Open-set** recognizers implement `get_embedding(face_image) -> np.ndarray`, returning a high-dimensional vector. Identity is determined post-hoc by comparing this embedding against a database of registered embeddings using cosine similarity. New identities can be registered at any time without retraining.
- **Closed-set** recognizers implement `recognize(face_image) -> (identity, confidence)`, returning a classification result directly from the model's softmax output. The set of identities is fixed at training time.

The `FinetunedRecognizer` class implements both methods. For closed-set use, `recognize()` runs a forward pass through the full network including the softmax head. For open-set use, `get_embedding()` extracts the output before the final classification layer. This dual-interface design allows the same model artifact to be used in either paradigm.

The two paradigms also carry different deployment costs. Closed-set requires retraining whenever the identity set changes — roughly fifty minutes for Progressive Unfreezing on this dataset — but adds no per-inference cost beyond a single forward pass. Open-set inverts this: enrolling a new identity takes seconds (capture samples, compute embeddings, append to the database), but every inference adds a similarity search whose cost grows with the gallery (linear with a flat CSV; sub-linear with an approximate nearest-neighbour index such as FAISS). Unknown probes are rejected by a soft confidence threshold on the softmax in closed-set mode and by a principled distance threshold in open-set mode. The dual-interface design supports a hybrid deployment: call `recognize()` first and fall back to `get_embedding()` when the softmax confidence is below threshold.

### 6.4.4 Model Loading and Switching

All recognizer models are declared in `config/models.yaml`. Each entry specifies the implementing class, model file path, version, and metadata:

```yaml
facenet_pu:
 class: "vision.recognition.finetuned_recognizer.FinetunedRecognizer"
 model_file: "bp_face_recognition/models/finetuned/facenet_progressive_v1.0.keras"
 version: "1.0"
 description: "FaceNet Progressive Unfreezing - 99.15% accuracy"
 metadata:
 accuracy: 0.9915
 training_time: "50 min"
 model_size: "272 MB"
 trainable_params: "23.6M (100%)"
 paradigm: "closed-set"
```

The registry currently tracks 17 recognizer configurations, including baseline models, GPU-trained variants, quantized models, and all three FaceNet fine-tuning strategies. The `global.default_recognizer` key is set to `facenet_pu`, meaning that any component requesting the default recognizer receives the progressive unfreezing model.

Environment profiles in the same YAML file allow the system to select different model combinations based on the detected platform. The `ModelRegistry.detect_environment()` method checks for WSL2 (by reading `/proc/version`), tests GPU availability via TensorFlow, and returns the appropriate profile name. On Windows without GPU, the system defaults to `windows_production`; on WSL2 with a detected NVIDIA GPU, it selects `wsl_production`.

## 6.5 Preprocessing and Data Pipeline

The data preprocessing pipeline transforms raw images into training-ready datasets through three sequential stages, each implemented as a standalone script under `preprocessing/`.

### 6.5.1 Face Cropping and Alignment

The `crop_faces.py` script processes raw datasets that contain full-frame images with JSON label files (produced by the LabelMe annotation tool). For each image, it reads the corresponding label file, extracts bounding box coordinates, adds a 10% margin around each face to include context, and writes the cropped region to the output directory.

Output filenames follow the convention `{identity}_{originalStem}.jpg`, for example `Yurii_f3d9f09a-8335-11ee.jpg`. The LFW dataset, which is already distributed as cropped face images, is excluded from the cropping step via the `EXCLUDE_DATASETS` set.

### 6.5.2 Dataset Splitting

The `split_lfw.py` script splits identity-grouped datasets into train/validation/test subsets with a 65/20/15% ratio. Splitting is stratified per identity: each identity's images are shuffled (with a fixed random seed of 42 for reproducibility) and divided proportionally. Output files are renamed to `{identity}_{index:04d}.jpg` and placed in a flat directory structure (no nested identity folders), which simplifies data loading during training.

The choice of flat directories over nested identity folders was motivated by consistency across dataset sources. Custom datasets (webcam, surveillance camera) naturally produce flat label-based filenames, while LFW uses nested folders. Normalizing to a single flat structure allows the same data loader to handle all datasets without format-specific logic.

### 6.5.3 Augmentation Pipeline

The `augmentation.py` script applies data augmentation to cropped faces using the Albumentations library [CITE: albumentations]. The augmentation pipeline consists of:

```python
alb.Compose([
 alb.Resize(height=224, width=224),
 alb.HorizontalFlip(p=0.5),
 alb.RandomBrightnessContrast(p=0.3),
 alb.RandomGamma(p=0.3),
 alb.RGBShift(p=0.2),
 alb.GaussNoise(p=0.2),
 alb.Blur(blur_limit=3, p=0.2),
])
```

Each input image produces 60 augmented variants by default (configurable via `--num-augmentations`). Augmented files are named `{baseName}.{N}.jpg` (e.g., `Yurii_f3d9f09a.0.jpg` through `Yurii_f3d9f09a.59.jpg`), preserving the identity label in the filename for downstream parsing.

Albumentations was chosen over TensorFlow's built-in `tf.image` augmentation for two reasons: its pipeline composition API is more expressive, and augmented images are written to disk once rather than generated on-the-fly during each training epoch. Pre-generating augmented images increases disk usage but provides deterministic training data across runs and eliminates augmentation overhead during training.

### 6.5.4 Full Pipeline

The three stages are chained via Makefile targets:

```makefile
prepare-all: prepare-crop prepare-split prepare-augment
```

Running `make prepare-all` executes crop, split, and augment in sequence. Each stage reads from the previous stage's output directory (`raw/ -> cropped/ -> augmented/`), and each can be run independently for debugging or partial reruns.

## 6.6 Cross-Platform Support

### 6.6.1 TensorFlow Lite Quantization

Trained Keras models can be converted to TensorFlow Lite format for deployment on resource-constrained devices. The quantization pipeline supports three strategies:

| Strategy | Size Reduction | Accuracy Impact | Use Case |
|----------|---------------|----------------|----------|
| Dynamic range | ~75% (19.8 MB -> 5.0 MB) | Minimal (~1-2%) | General deployment |
| Float16 | ~62% (24 MB -> 9.4 MB) | Negligible | GPU-capable devices |
| Int8 (full) | ~75% | Moderate (~3-5%) | Edge devices |

The quantized EfficientNetB0 model achieves 73.6% size reduction (from 19.8 MB to 5.0 MB) using dynamic range quantization, with accuracy dropping from 76.67% to approximately 75%. The `TFLiteRecognizer` (`vision/recognition/tflite_recognizer.py`) loads these models and provides the same `FaceRecognizer` interface as the full Keras models, so quantized and unquantized models are interchangeable at the configuration level.

### 6.6.2 GPU Delegate Support

GPU acceleration is supported through two paths:

1. **MediaPipe GPU delegate**: The `MediaPipeDetector` can run face detection on the GPU via the MediaPipe Tasks API. On WSL2 with an NVIDIA GPU, this provides a 5-20x speedup over CPU inference for detection.
2. **TensorFlow GPU**: Training and recognition inference can run on NVIDIA GPUs via TensorFlow's CUDA integration. The project targets CUDA 12.2 with cuDNN 8.9.

The `cross_platform_gpu.py` utility provides unified GPU detection across Windows, Linux, macOS, and WSL2. On Windows, where native GPU support for MediaPipe is unavailable, the system falls back to CPU inference automatically. On WSL2, it checks for GPU availability via `tf.config.list_physical_devices("GPU")` and enables GPU acceleration when hardware is present.

### 6.6.3 Platform-Specific Dependency Management

The project uses PEP 508 platform markers in `pyproject.toml` for conditional dependency installation:

- `tensorflow` is installed on Linux (with GPU support), while `tensorflow-cpu` is installed on Windows
- `dlib` on Windows uses a pre-built wheel from the `wheels/` directory to avoid compilation, while on Linux it is installed from source with GPU support
- `onnxruntime-gpu` is installed on Linux; `onnxruntime` (CPU-only) is installed on Windows

Separate virtual environments are maintained per platform (`.venv-win` for Windows, `.venv-wsl` for WSL2) to avoid binary incompatibilities. A single `uv.lock` file works across both platforms because `uv` resolves platform markers at install time, selecting the appropriate package variant for the current operating system.

## 6.7 Build and Development Tools

### 6.7.1 Package Manager: uv

The project uses `uv` [CITE: uv package manager] for dependency management, chosen over `pip` and `Poetry` for its speed (10-100x faster resolution) and reliable cross-platform lockfile generation. Dependencies are declared in `pyproject.toml`, and `uv sync` installs exactly the versions specified in `uv.lock`, ensuring reproducible environments.

### 6.7.2 Test Runner: nox

Testing is managed through `nox`, which defines isolated test sessions in `noxfile.py`. Each session creates a fresh environment and runs a specific subset of the test suite:

- `nox -s tests` --- full test suite
- `nox -s test_config` --- configuration tests only
- `nox -s test_training` --- training pipeline tests
- `nox -s test_preprocessing` --- preprocessing pipeline tests
- `nox -s lint` --- Ruff linting

Tests use `pytest` with markers (`unit`, `integration`, `slow`) to control which tests run in different contexts. Unit tests require no external resources; integration tests may need a connected camera or downloaded model files.

### 6.7.3 Makefile Automation

The Makefile provides platform-aware shortcuts for common operations:

```makefile
run: # Launch open-set recognition with camera
run-closed-set: # Launch closed-set recognition with camera
train-wsl: # GPU training via WSL2 (sets XLA_FLAGS, PYTHONPATH)
train-cpu: # CPU-only training on Windows
prepare-all: # Full preprocessing pipeline
thesis-benchmark: # Generate all benchmark data for thesis
register: # Register a new person for open-set recognition
```

WSL training commands set environment variables such as `XLA_FLAGS` (pointing to the CUDA toolkit path) and activate the WSL-specific virtual environment. This encapsulation means that users do not need to remember platform-specific setup steps.

### 6.7.4 Project Configuration

The `pyproject.toml` file is the single source of truth for project metadata, dependencies, and tool configuration. It declares Python 3.11 as the required version, lists all dependencies with version constraints, and configures Ruff (linting), mypy (type checking), and pytest (test framework) in a single file. This consolidation eliminates the need for separate `setup.cfg`, `.flake8`, or `mypy.ini` files.

## 6.8 Summary of Design Decisions

**Table 6.1** summarizes the key architectural decisions, the alternatives that were considered, and the rationale behind each choice.

| Decision | Choice Made | Alternative Considered | Rationale |
|----------|-------------|----------------------|-----------|
| Model loading | Config-driven YAML registry | Hardcoded imports | Extensibility without code changes; single source of truth |
| Plugin interfaces | Abstract Base Classes | Protocol / duck typing | Explicit contracts, early error detection, IDE support |
| Embedding storage | CSV + cosine similarity | SQLite, FAISS | Sufficient for thesis scope; no server dependency |
| Training environment | WSL2 with native CUDA | Docker, cloud (AWS/GCP) | Direct GPU access, zero marginal cost |
| Package manager | uv | pip, Poetry | 10-100x faster resolution, reliable lockfile |
| Recognition modes | Dual (open-set + closed-set) | Single mode | Enables comparative analysis for thesis |
| Default detector | MediaPipe BlazeFace | MTCNN | 100x faster; sufficient accuracy for real-time use |
| FaceNet backbone | InceptionResNetV1 | ResNet50, MobileNet | Pre-trained on faces (MS-Celeb-1M); 512D embeddings |
| Augmentation library | Albumentations | tf.image, imgaug | Expressive pipeline API; disk-based for reproducibility |
| Fine-tuning strategy | Progressive unfreezing (default) | Transfer learning, triplet loss | Best accuracy (99.15%) with acceptable training time |
| Dataset structure | Flat directories | Nested identity folders | Uniform handling across all dataset sources |

The YAML-based plugin system was the most consequential decision. It introduced a layer of indirection between configuration and code that makes runtime errors harder to trace (a typo in a class path produces a runtime `ImportError` rather than a static analysis warning). This trade-off was accepted because the registry validates all entries at startup, catching configuration errors before any processing begins. The benefit --- adding a new model by editing a YAML file rather than modifying Python imports --- was considered more valuable for a research project where model configurations change frequently.
