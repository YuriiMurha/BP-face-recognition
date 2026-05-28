# System Manual

## B.1 Purpose and Scope

This appendix documents the internal organisation of the face recognition codebase. It is intended for a reader who wishes to extend, modify, or audit the system rather than merely operate it. No theoretical material is repeated here; for the algorithms and the rationale behind the methodological choices, the body chapters of the thesis are the authoritative reference.

## B.2 Repository Layout

The repository follows a Python source layout, with all importable code located beneath the src directory. Source files outside this directory are tools, scripts, configuration, or build artefacts that do not form part of the importable package.

```
BP-face-recognition/
  config/                  YAML model registry and runtime settings
  data/                    Datasets and training logs (gitignored)
  scripts/                 Standalone tools: training launchers, plotting, eval
  src/bp_face_recognition/ Importable package
    config/                Typed settings classes
    database/              Embedding storage backends
    evaluation/            Benchmark harnesses and metric computation
    models/                Trained model files (.keras, .tflite)
    preprocessing/         Cropping, splitting, augmentation
    services/              Pipeline service layer (orchestration)
    tests/                 Pytest test suites organised by marker
    utils/                 Cross-cutting helpers (camera, paths, GPU)
    vision/                Detection and recognition implementations
      core/                Shared interfaces and tracking utilities
      detection/           Five detector implementations
      recognition/         Five recognition backend implementations
      training/            Three training paradigms (classifier, metric, fine-tune)
  thesis/                  This thesis: chapters, references, figures
  Makefile                 Cross-platform command shortcuts
  pyproject.toml           Python dependencies and project metadata
  uv.lock                  Exactly reproducible dependency lockfile
```

## B.3 Component Architecture

The system follows a layered design with three principal layers: the vision layer, the service layer, and the application entry points.

The vision layer defines two abstract interfaces, one for face detectors and one for face recognizers, and provides several concrete implementations of each. A detector takes an image and returns a list of rectangles; a recognizer takes a cropped face image and returns a fixed-length embedding vector. The two interfaces are deliberately narrow so that any implementation can be swapped for any other at runtime without modifying calling code.

The service layer orchestrates a single recognition request end to end: it accepts a raw video frame, invokes the configured detector to locate faces, crops each face to the size expected by the recognizer, queries the database for matching identities, applies a similarity threshold, and returns annotated results. The same service is used by both the open-set and closed-set application entry points.

The application entry points are the two scripts that wire the service layer to a camera source and a display window. Open-set mode reads embeddings from a local face database and matches against them by cosine similarity. Closed-set mode skips the database and uses the recognizer's classifier head directly.

## B.4 Model Registry and Plugin Selection

Detectors and recognizers are not hard-wired into the code. They are declared in the YAML model registry, which lists every available implementation together with the fully qualified Python path of its class, the version string, a description, and the default parameter values. At startup the registry is parsed, classes are resolved through dynamic import, and the resulting plugin objects are made available to the service layer through a factory function.

Adding a new detector or a new recognizer is therefore a two-step operation: implement the appropriate interface in a new file, then declare an entry in the registry YAML pointing to the new class. No changes to the service layer, the entry points, or the Makefile are required. The same registry is used to set the default model selection, which is in turn overridable through environment variables for ad-hoc experimentation.

## B.5 Configuration

Application configuration is split between typed runtime settings and the YAML registry described above.

The typed settings class declares fields for filesystem directories, camera parameters, database credentials, and the default model selection. Values come from environment variables, from a local .env file, or from explicit overrides at construction time. Field types are validated at startup, so a misspelled environment variable or a wrong-typed value produces an immediate error rather than a confusing runtime failure later in execution.

The full list of environment variables read by the application is documented in the source of the settings class. The variables most relevant to operation are the camera source type, the camera device index, the camera resolution and frame rate, the default recognizer name, and the database connection string. All of these have sensible defaults; the typical user does not set any of them.

## B.6 Training Workflow

Three training paradigms are implemented, each with its own trainer module: a closed-set classifier trainer for cross-entropy on a fixed set of identities, a metric-learning trainer for triplet-loss embedding optimization, and a FaceNet fine-tuning trainer that supports the three strategies compared in Chapter 3 (feature extraction with a frozen backbone, progressive unfreezing, and triplet loss with online mining).

Training is launched through Makefile targets that handle the cross-platform plumbing. On Windows the targets activate the Windows virtual environment and run the trainer on CPU. On WSL2 the targets activate the WSL virtual environment, set the CUDA toolkit path through XLA flags, convert Windows paths to their WSL equivalents, and launch the same trainer with GPU acceleration enabled. The trainer code itself is identical between the two platforms; the divergence lives entirely in the Makefile.

Trained models are written to the models directory as Keras .keras files, with TensorFlow Lite versions produced by a separate quantization script. Training history (per-epoch loss and accuracy) is serialised to JSON alongside the model file so that training curves can be regenerated for the thesis from a single source.

## B.7 Dataset Preparation

Datasets follow a flat layout: each image is named with the convention identity-underscore-uniqueid (for the custom datasets) or identity-underscore-index (for LFW). The flat layout was chosen over the conventional one-folder-per-class arrangement because it simplifies augmentation accounting and because the identity-from-filename convention removes one indirection during data loading.

The preprocessing pipeline consists of three stages, each invokable as a separate Makefile target. The cropping stage runs MediaPipe over the raw images and writes square face crops; the splitting stage partitions the cropped set into training, validation, and test subsets while preserving per-identity proportions; the augmentation stage applies the Albumentations transform pipeline and writes the augmented images to disk so that training does not pay the augmentation cost on every epoch.

All three stages are idempotent and reproducible: re-running a stage on the same input produces the same output, with random seeds either fixed in code or pinned through Makefile arguments.

## B.8 Build, Test, and Quality Checks

The project's command-line entry points are exposed through Makefile targets that wrap the underlying tool invocations. The most commonly used targets are setup for dependency installation, run for the open-set application, train and its variants for model training, evaluate and benchmark for the thesis evaluation, and test, lint, and type-check for quality assurance.

Tests are organised by pytest marker into three categories: unit tests with no external dependencies, integration tests that may need a connected camera or a downloaded model file, and slow tests that are excluded from the default run to keep iteration time short. The nox configuration declares separate sessions for each category and for the lint pass, with every session creating its own isolated virtual environment.

Linting is performed by ruff, which combines the older flake8-plus-isort-plus-pylint chain into a single binary. Static type checking is performed by mypy with module overrides for libraries that ship no type stubs. Both tools read their configuration from the project's pyproject.toml so that there is one place to inspect when adjusting style or strictness rules.

## B.9 Cross-Platform Notes

The project is supported on Windows (CPU only) and on Ubuntu under WSL2 (CPU or GPU). The dual-platform support is implemented at three levels.

At the dependency level, platform-conditional installs are declared in pyproject.toml using PEP 508 environment markers; the same lockfile resolves to the GPU-capable TensorFlow wheel on Linux and the CPU-only wheel on Windows.

At the binary level, dlib is built from source on Linux but installed from a vendored pre-built wheel on Windows, where building from source requires the full Visual Studio toolchain. The wheel is referenced through uv's local-source mechanism so that uv sync produces the same result on any developer's Windows machine.

At the runtime level, the cross-platform GPU utility module detects whether the application is running under WSL2 with NVIDIA drivers visible and configures the MediaPipe inference path accordingly: the GPU delegate is used when available, the CPU delegate when the GPU is unreachable, and the OpenCV Haar fallback when both deep paths fail. The fallback chain is deliberate; it ensures that the application never silently degrades to a less accurate detector without logging.

## B.10 Reproducing Thesis Results

Every numerical result reported in Chapter 6 can be regenerated from the repository state through a single Makefile target. The thesis-benchmark target trains or loads each model variant, runs the ground-truth detection evaluation, computes the verification and embedding-quality metrics, and writes the resulting JSON files and PNG plots to the results directory. The plotting scripts that turn the JSON files into the figures embedded in this thesis are themselves Makefile targets that depend only on the JSON outputs of the benchmark stage.

Pinned random seeds, exactly reproducible dependency installation through the lockfile, and deterministic Albumentations augmentation together mean that two runs of the benchmark on the same hardware produce bitwise identical output. Cross-hardware runs may produce small floating-point differences but not differences large enough to change the conclusions.
