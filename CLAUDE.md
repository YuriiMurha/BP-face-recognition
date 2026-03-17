# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bachelor thesis project: **Face Recognition Using Surveillance Systems** by Yurii Murha (TUKE). A production-ready face detection and recognition system with multiple detector/recognizer backends, GPU acceleration, model quantization, and metric learning.

- **Python 3.11** with **uv** package manager
- **src-layout**: all source code under `src/bp_face_recognition/`
- `PYTHONPATH=src` is required (Makefile sets this automatically)
- Separate venvs per OS: `.venv-win` (Windows/CPU), `.venv-wsl` (WSL2/GPU)
- Entry point: `src/bp_face_recognition/main.py` (CLI app with camera)

## Build & Run Commands

```bash
uv sync                          # Install dependencies
make test                        # Run full test suite (via nox)
make lint                        # Ruff linting
make type-check                  # mypy type checking
make run                         # Run face recognition app with camera

# Run specific nox sessions
uv run nox -s tests              # All tests
uv run nox -s test_config        # Config tests only
uv run nox -s test_training      # Training tests only
uv run nox -s test_preprocessing # Preprocessing tests
uv run nox -s lint               # Lint only

# Run a single test file directly
uv run pytest src/bp_face_recognition/tests/unit/test_config.py -v

# Training (GPU via WSL2)
make train-wsl backbone=EfficientNetB0 epochs=20
make train-facenet-pu            # Best model: Progressive Unfreezing (99.15%)

# Data preprocessing pipeline
make prepare-all                 # Full pipeline: split → crop → augment
make prepare-crop dataset=lfw    # Crop faces only
make prepare-augment dataset=lfw # Augment only

# Register a person for recognition
make register name="PersonName"
```

## Architecture

### Plugin System (config-driven)

All detectors and recognizers are registered in `config/models.yaml` and loaded dynamically by `vision/registry.py`. The `RecognizerFactory` (`vision/factory.py`) creates instances via the registry with legacy name fallback. 20+ model combinations available across 8 environment profiles (dev/prod/benchmark for Windows/WSL).

### Core Interfaces (`vision/interfaces.py`)

- **`FaceDetector`** — abstract base with `detect(image) -> List[(x,y,w,h)]`
- **`FaceRecognizer`** — abstract base with `get_embedding(face_image) -> ndarray`

### Detection Backends (`vision/detection/`)

5 implementations: MediaPipe BlazeFace (default, GPU-capable), MTCNN, Haar Cascade, Dlib HOG, face_recognition lib. MediaPipe has multi-tier fallback: GPU → CPU → OpenCV.

### Recognition Backends (`vision/recognition/`)

- **TFLite** (`tflite_recognizer.py`) — EfficientNetB0/MobileNetV3 classifiers (closed-set)
- **Dlib** (`dlib_recognizer.py`) — 128D embeddings via face_recognition lib (open-set)
- **FaceNet pretrained** (`facenet_keras.py`) — 512D embeddings, 99.6% LFW (open-set)
- **FaceNet fine-tuned** (`finetuned_recognizer.py`) — 3 variants: TL/PU/TLoss (see below)
- **Metric learning** (`keras_metric_recognizer.py`) — custom triplet loss, 128D (open-set)

### Service Layer

- **`PipelineService`** (`services/pipeline_service.py`) — orchestrates detection → recognition → database, tracks performance stats
- **`DatabaseService`** (`services/database_service.py`) — CSV/PostgreSQL abstraction for embeddings
- **`FaceTracker`** (`vision/core/face_tracker.py`) — combines detector + recognizer into unified tracking interface

### Training (`vision/training/`)

Three training paradigms:
- `classifier/trainer.py` — closed-set classification (EfficientNetB0/MobileNetV3, softmax)
- `metric/trainer.py` — open-set metric learning (triplet loss, 128D embeddings)
- `finetune/` — FaceNet fine-tuning with 3 strategies:
  - `facenet_transfer_trainer.py` — Option A: Transfer Learning (frozen base + head, 92.84%, 4 min)
  - `facenet_progressive_trainer.py` — Option B: Progressive Unfreezing (4-phase, **99.15%**, 50 min) **BEST**
  - `facenet_triplet_trainer.py` — Option C: Triplet Loss (metric learning, 94.63%, 90 min)

### Preprocessing (`preprocessing/`)

Pipeline: `crop_faces.py` (MediaPipe-based) → `split_lfw.py` → `augmentation.py` (Albumentations). Datasets use flat structure with naming: `{label}_{uuid}.jpg`, augmented: `{label}_{uuid}.{N}.jpg`.

### Settings

`config/settings.py` uses Pydantic Settings for configuration. Provides `ROOT_DIR`, `MODELS_DIR`, `DATASETS_DIR`. Auto-detects environment (Windows/WSL, GPU/CPU).

## Current Research Status

Session-based development tracked in `.maintenance/` directory:
- **Sessions 12-14**: FaceNet fine-tuning study complete — 3 approaches compared
- **Session 17**: Runtime testing of FaceNet models (in progress)
- **Best model**: FaceNet Progressive Unfreezing (`facenet_pu`) — 99.15% accuracy on 14-class combined dataset (7,080 images)
- **Production default recognizer**: `facenet_pretrained` (config/models.yaml `global.default_recognizer`)
- **Next priorities**: Runtime model comparison, thesis documentation, production optimization

Key files for context:
- `.maintenance/TODO.md` — current tasks and session plans
- `.maintenance/PROGRESS.md` — detailed session achievements
- `.maintenance/reports/` — comprehensive reports including implementation context

## Testing

- Framework: **pytest** via **nox** sessions (nox uses uv backend)
- Tests location: `src/bp_face_recognition/tests/`
- Markers: `unit` (no external resources), `integration` (may need camera/models), `slow`
- Key test files: `test_config.py`, `test_training.py`, `test_preprocessing.py`, `test_camera_source.py`, `test_data_augmentation.py`

## Cross-Platform Notes

- `pyproject.toml` uses `platform_system` markers: `tensorflow` on Linux, `tensorflow-cpu` on Windows
- dlib on Windows uses a pre-built wheel from `wheels/` directory (`dlib-19.24.1-cp311-cp311-win_amd64.whl`)
- WSL2 training commands in Makefile set `XLA_FLAGS` for CUDA toolkit path
- GPU training requires NVIDIA CUDA 12.2 + cuDNN 8.9, GTX 1650+ with 4GB+ VRAM
- Single `uv.lock` works across both platforms via platform markers

## Key File Conventions

- Models: `.keras` (full) or `.tflite` (quantized) in `src/bp_face_recognition/models/`
- Fine-tuned FaceNet models in `src/bp_face_recognition/models/finetuned/`
- Face embeddings stored in `data/faces.csv`
- Datasets: `data/datasets/{raw,cropped,augmented}/{dataset_name}/`
- Training logs go to `data/logs/`
- Session progress and plans in `.maintenance/` directory
- Thesis documentation in `docs/thesis/`
- Model switching: `scripts/switch_model.py` or `switch.bat` (Windows)
