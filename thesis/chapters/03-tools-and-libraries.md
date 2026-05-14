# Chapter 3: Tools and Libraries

## 3.1 Introduction

The choice of tools shapes what is feasible to build and what is feasible to measure. A face recognition pipeline that targets surveillance camera footage has to handle image acquisition, geometric face localization, deep feature extraction, similarity scoring against a database, and reproducible training of the underlying neural networks. No single library covers that span, so the system is assembled from a stack of specialized components. This chapter introduces those components, grouped by the role they play in the pipeline, and gives the rationale for each selection. Where a credible alternative exists, the alternative is named and the trade-off is stated rather than glossed over.

One constraint runs through every choice in this chapter: the project has to build and run on two operating systems --- Windows for day-to-day development and quick CPU benchmarks, and Ubuntu under WSL2 for GPU training. That dual-platform constraint rules out several otherwise attractive options (CUDA-only libraries, Linux-only build systems) and motivates the use of a package manager that resolves platform markers reliably. The toolchain described below was selected with that constraint in mind.

## 3.2 Programming Language and Runtime

The implementation language is Python 3.11, pinned exactly in `pyproject.toml` (`requires-python = "==3.11.*"`). Python is the default choice in computer vision and machine learning research, not by historical accident, but because the three deep learning frameworks that matter for face recognition --- TensorFlow, PyTorch, and JAX --- all expose their primary APIs in Python, and almost every published model checkpoint is distributed with Python loading code. Choosing anything else would mean reimplementing model loaders before the first experiment.

Python 3.11 was selected over later versions for compatibility with TensorFlow 2.15, which is the latest version that supports the `tf.keras` API used by the `keras-facenet` package [CITE: kerasfacenet]. Newer Python releases broke binary compatibility with several TensorFlow wheels available at the time the project was started. Pinning to 3.11 also pins the interpreter ABI, which matters for the pre-built `dlib` wheel that Windows installation depends on.

Cross-platform behaviour was a key consideration. The standard library's `pathlib` is used throughout for filesystem paths, and the project avoids hard-coded path separators. Where platform-specific behaviour is unavoidable --- GPU detection, WSL path conversion, the choice between `tensorflow` and `tensorflow-cpu` --- the divergence is handled in `pyproject.toml` platform markers and in the `cross_platform_gpu.py` utility, so application code remains portable. No C or C++ wrappers were written for this project. Every backend used here ships its own native code (OpenCV, dlib, MediaPipe, TensorFlow), and the project's own logic is pure Python. This keeps the build simple and removes a class of cross-compilation problems.

## 3.3 Deep Learning Frameworks

The primary deep learning framework is TensorFlow 2.15 with its built-in Keras API [CITE: abadi2016tensorflow; chollet2015keras]. TensorFlow is the framework in which the original FaceNet weights are distributed and is also the framework that MediaPipe's GPU inference path targets. Using TensorFlow keeps the recognition and detection sides of the pipeline on a single tensor backend, which simplifies dependency management and removes one source of subtle floating-point divergence between the two stages.

Keras is the high-level API actually used in the codebase. All three FaceNet fine-tuning trainers (`facenet_transfer_trainer.py`, `facenet_progressive_trainer.py`, `facenet_triplet_trainer.py`) call `keras.Model`, `keras.optimizers.Adam`, and `keras.losses.SparseCategoricalCrossentropy`. The lower-level `tf.GradientTape` is reserved for the triplet-loss trainer, where the gradient computation needs to span an embedding distance rather than a per-class loss. This split --- Keras for ordinary supervised training, raw TensorFlow only when the loss is non-standard --- keeps the trainer code short and readable.

The `keras-facenet` package [CITE: kerasfacenet] bundles the InceptionResNetV1 architecture with pre-trained weights converted from David Sandberg's well-known TensorFlow checkpoint. It exposes a single `FaceNet` class with an `embeddings()` method that returns 512-dimensional vectors. Using this package avoided the considerable effort of porting weight files between formats and verifying that the architecture matches the reference implementation. The package is small and unmaintained but stable, and it was vendored into the project's recognition module (`vision/recognition/facenet_keras.py`) as a thin wrapper.

PyTorch was considered as the primary framework. The `facenet-pytorch` package [CITE: esler2021facenetpytorch] provides a comparable InceptionResNetV1 implementation and a PyTorch port of MTCNN. PyTorch's eager execution model is often easier to debug, and modern face recognition research (ArcFace, SphereFace) is more commonly released in PyTorch. The decision to stay with TensorFlow was driven by two factors: the MediaPipe Tasks API for GPU-accelerated face detection is TensorFlow-native, and the project's earliest models (an EfficientNetB0 closed-set classifier) were already in Keras. Mixing frameworks would have meant maintaining two preprocessing pipelines and two model loaders. PyTorch is still indirectly present in the system through the `mtcnn` PyPI package, which uses its own TensorFlow re-implementation rather than the PyTorch port, so the framework boundary remained clean in practice.

TensorFlow Lite is used for model quantization and resource-constrained deployment. The `TFLiteRecognizer` in `vision/recognition/tflite_recognizer.py` loads `.tflite` files produced by `tf.lite.TFLiteConverter` and exposes the same `FaceRecognizer` interface as the full Keras models. Three quantization strategies were trialled (dynamic range, float16, full int8), with dynamic range chosen as the default because it reduces model size by roughly three-quarters at minimal accuracy cost.

Platform-specific resolution is handled at the dependency level. `pyproject.toml` declares `tensorflow` on Linux and `tensorflow-cpu` on Windows via PEP 508 markers (`platform_system == 'Linux'` and `platform_system == 'Windows'`), so the same `uv.lock` file installs the GPU-capable wheel on WSL2 and the CPU-only wheel on Windows. The `tensorflow-io-gcs-filesystem` package is pinned to `<0.32` on Windows because later versions require Linux-only system libraries.

## 3.4 Computer Vision Libraries

Five computer vision libraries appear in the pipeline. Each has a specific role and was kept rather than removed because no single library covered all of them.

**OpenCV** [CITE: bradski2000opencv] handles low-level image work: camera capture (`cv2.VideoCapture`), colour-space conversion between BGR and RGB, drawing bounding boxes and identity labels on output frames, file I/O for cropping outputs, and the Haar cascade classifier used as one of the four detection baselines. Most of the project's frame-processing utilities (`utils/camera.py`, the per-frame drawing routines in `services/pipeline_service.py`) call OpenCV functions. The library is mature, fast, and ships with pre-built wheels on PyPI for both platforms.

**MediaPipe** [CITE: lugaresi2019mediapipe] provides the default face detector through its BlazeFace model [CITE: bazarevsky2019blazeface]. MediaPipe's Tasks API is loaded by `vision/detection/mediapipe.py`, which implements the three-tier GPU/CPU/Haar fallback described in Chapter 6. BlazeFace processes frames at over 200 FPS on CPU and was the only detector in the comparison that supported a GPU delegate. The trade-off, established in Chapter 7's ground-truth detection evaluation, is that BlazeFace's single-scale anchor design misses many of the small or oblique faces typical of surveillance frames; MTCNN was retained precisely to cover that gap.

**dlib** [CITE: king2009dlib] supplies the HOG-based face detector and the 128-dimensional face embedding network used by the `dlib_v1` recognizer. dlib is C++ at heart with Python bindings, which makes it CPU-efficient but awkward to install on Windows: compilation from source requires CMake and a recent MSVC toolchain. The project sidesteps this by vendoring a pre-built wheel (`wheels/dlib-19.24.1-cp311-cp311-win_amd64.whl`) and referencing it through `tool.uv.sources` in `pyproject.toml`. On Linux, dlib builds cleanly from source, so no wheel is needed.

**MTCNN** is consumed through the `mtcnn` PyPI package, which is a pure-TensorFlow re-implementation of the three-stage cascade described by [CITE: zhang2016mtcnn]. The package is small and depends only on TensorFlow and NumPy, which keeps the import surface narrow. MTCNN was retained despite its slow inference (around 4 FPS on CPU) because its multi-scale pyramid recovers faces that BlazeFace systematically misses --- it is the highest-recall detector in the ground-truth evaluation, and so functions as the practical reference for "what could the pipeline find if speed were free."

**face_recognition** is a thin Python wrapper around dlib that ships with simpler defaults for the common cases (detect-then-embed) and a slightly more pythonic API. It is used in `vision/recognition/dlib_recognizer.py` as one of the open-set recognition backends. The package was kept in addition to direct dlib usage because it bundles the canonical face embedding model and exposes it through a higher-level interface that downstream callers find easier to use.

The multi-backend design --- five detectors and four recognition families behind a single plugin interface --- is a deliberate research choice. The thesis compares detection methods quantitatively, and that comparison is only credible if each method is a real, idiomatic instance of the technique rather than a hand-rolled approximation. Wrapping each library in its own `FaceDetector` subclass keeps the comparison honest while also giving the deployment side an actual choice between speed and recall.

## 3.5 Data Processing and Machine Learning Utilities

Augmentation is performed by Albumentations [CITE: buslaev2020albumentations]. The pipeline in `preprocessing/augmentation.py` chains a `Resize` to 224×224, a horizontal flip with probability 0.5, random brightness/contrast jitter, gamma shifts, RGB channel shifts, additive Gaussian noise, and a small Gaussian blur. Albumentations was chosen over `tf.image` and `imgaug` for two practical reasons: its `Compose` API is more expressive when transformations have to be probabilistic and conditional, and its output is deterministic given a seed, which matters for reproducible thesis benchmarks. Augmented images are materialized to disk once rather than generated on-the-fly during training, which trades disk space for training-time speed and eliminates augmentation as a confound in training-curve comparisons.

scikit-learn [CITE: pedregosa2011scikit] supplies the metrics and the stratified split. `sklearn.metrics` provides per-class precision, recall, F1, and the balanced-accuracy score used throughout Chapter 7. `sklearn.metrics.silhouette_score` is used in the embedding-quality analysis to quantify the separation between identity clusters. The library's `train_test_split` was used during the initial split prototyping, although the production splits in `preprocessing/split_lfw.py` use a hand-written stratified splitter that preserves per-identity ratios more carefully than scikit-learn's default.

NumPy [CITE: harris2020numpy] is the tensor type that crosses every interface boundary: detector outputs are NumPy bounding boxes, recognizer outputs are NumPy embeddings, the face database stores NumPy arrays in CSV form. The project uses NumPy 1.x rather than 2.x (pinned `numpy>=1.20.0,<2.0.0`) because TensorFlow 2.15 was not yet compatible with NumPy 2 when the dependencies were locked.

Pandas [CITE: mckinney2010pandas] handles the face database CSV (`data/faces.csv`) and the benchmark result tables emitted to `results/thesis/`. The choice was made for convenience: CSV is human-readable, version-controllable, and small enough at thesis scale that no further indexing is needed. A SQLite backend was considered but rejected as over-engineering for the dataset size.

Matplotlib and Seaborn produce the training curves, confusion matrices, and embedding-quality plots that appear in Chapter 7. TensorBoard was used during early training experiments for live loss curves but is not part of the final reporting pipeline; static plots saved as PNG were preferred for the thesis because they fit the figure-based format of the document.

## 3.6 Configuration and Plugin Architecture

Configuration is split between typed application settings and a model registry.

Pydantic [CITE: pydanticproject] together with `pydantic-settings` provides the application settings class in `config/settings.py`. The class declares typed fields for directories (`ROOT_DIR`, `MODELS_DIR`, `DATASETS_DIR`), camera parameters, database credentials, and the default model selection. Values come from environment variables, a `.env` file, or explicit code overrides, with the typed fields catching wrong types at startup rather than at first use. The same `Settings` instance is imported throughout the codebase, so there is one source of truth for paths and runtime knobs.

The plugin registry uses YAML rather than Python code. `config/models.yaml` declares each detector and each recognizer with its fully qualified class path, model file location, version string, and default parameter overrides. The `ModelRegistry` in `vision/registry.py` reads this YAML at startup and resolves class paths through `importlib.import_module`. The full pattern is described in Chapter 6 --- here it suffices to note that YAML was chosen over a Python configuration module so that model selection could be changed without code edits and so that the same configuration could be inspected, diffed, and version-controlled as a single human-readable file.

## 3.7 Build and Development Tooling

The project uses `uv` [CITE: astraluv] for dependency resolution and lockfile generation. `uv` was chosen over `pip` and Poetry because it resolves dependency graphs in a fraction of the time (typically a few seconds for this project's lockfile, compared to minutes for Poetry) and because its handling of platform markers is reliable in practice: a single `uv.lock` file works for both Windows and WSL2 installations, with `uv sync` picking the correct wheel for each platform automatically. `uv` also handles the local-wheel reference for `dlib` on Windows through the `tool.uv.sources` table.

`nox` [CITE: noxproject] is the test session orchestrator. Test sessions are declared in `noxfile.py`, with separate sessions for the full test suite, configuration-only tests, training-only tests, preprocessing tests, and the lint pass. Each session creates its own isolated virtual environment, which catches the class of bugs that occur when a test accidentally depends on an unlisted package. `nox` was preferred over `tox` because its session definitions are plain Python, which makes it easy to compose commands and to share helper functions across sessions.

`pytest` is the test framework underneath nox. The project uses three markers (`unit`, `integration`, `slow`) configured in `pyproject.toml` to control which tests run in different contexts. Unit tests have no external dependencies; integration tests may need a connected camera or a downloaded model file; slow tests are excluded from the default run to keep iteration time short.

`ruff` [CITE: astralruff] handles linting. It was chosen over the older flake8/isort/pylint combination because it is roughly two orders of magnitude faster and bundles linting, import sorting, and a subset of pyflakes checks in a single binary. The configuration is short --- a few rule sets enabled, a few rules ignored --- and lives in `pyproject.toml` alongside everything else. Type checking is handled separately by `mypy`, configured in the same file with module overrides for libraries that do not ship type stubs (cv2, tensorflow, dlib).

GNU Make provides the command-line shortcuts that wrap the verbose commands. The Makefile encodes platform-specific environment setup (the WSL training targets export `XLA_FLAGS` for the CUDA toolkit path, activate the right virtual environment, and convert Windows paths to WSL paths) so that ordinary use --- `make run`, `make train-wsl backbone=...`, `make thesis-benchmark` --- requires no platform knowledge from the user. Make is the only non-Python build tool in the stack; it was kept because the shortcuts are short and because it is universally available.

Two separate virtual environments are maintained: `.venv-win` on Windows and `.venv-wsl` on WSL2. The split is necessary because TensorFlow's GPU support is Linux-only, and `tensorflow` and `tensorflow-cpu` cannot coexist in a single environment. Using two environments also keeps the wheel caches clean --- `dlib` on Windows uses the vendored binary wheel, while on Linux it builds from source --- and avoids the brittle situation where `uv` would have to resolve a different package set depending on which platform invoked it.

## 3.8 Annotation and Dataset Preparation

LabelMe [CITE: russell2008labelme] is the annotation tool used to produce the ground-truth bounding boxes for the detection evaluation. The frame-level annotations stored in `data/datasets/raw/{seccam, seccam_2, webcam}/test/labels/*.json` are LabelMe's native JSON format, with each file containing a list of shapes (one per face) and the two corner points that define each rectangle. The detection benchmark in Chapter 7 reads these files directly, extracts the rectangles, scales them to match the resized inference frames, and computes IoU against the predictions of each detector. Twenty-six ground-truth boxes were annotated across nineteen test frames.

It is important to distinguish this use of LabelMe from its role in the wider literature. The training datasets used in this project --- the cropped LFW subset, the webcam dataset, and the surveillance camera dataset --- are not annotated per image. They use a flat-file naming convention (`{label}_{uuid}.jpg`) that encodes the identity in the filename itself, with face crops produced by an automated pipeline rather than by manual annotation. LabelMe was used only where genuine per-pixel bounding boxes were needed, which in this project means the small, hand-curated detection test set. Reporting it as the labelling tool for the whole pipeline would overstate its role.

## 3.9 Summary

Table 3.1 lists every tool described above, its role in the pipeline, and the principal alternative that was considered.

**Table 3.1: Tool selections with considered alternatives.**

| Tool | Role | Alternative considered |
|------|------|-----------------------|
| Python 3.11 | Implementation language | Python 3.12 (rejected for TF 2.15 wheel compatibility) |
| TensorFlow 2.15 + Keras | Deep learning framework | PyTorch + facenet-pytorch |
| keras-facenet | Pre-trained FaceNet weights | Manual weight conversion from David Sandberg's checkpoint |
| TensorFlow Lite | Quantization for edge deployment | ONNX Runtime (still present, secondary) |
| OpenCV | Image I/O, drawing, Haar cascade | PIL/Pillow (too narrow) |
| MediaPipe | Default face detector (BlazeFace) | Direct TF Hub BlazeFace |
| dlib | HOG detector + 128D embeddings | OpenCV's `face` contrib module |
| mtcnn | Three-stage cascade detector | facenet-pytorch's MTCNN port |
| face_recognition | High-level dlib wrapper | Direct dlib API calls |
| Albumentations | Image augmentation | tf.image, imgaug |
| scikit-learn | Metrics, silhouette score, splits | Hand-rolled metric implementations |
| NumPy, Pandas | Numerical arrays, CSV database | xarray, SQLite |
| Matplotlib, Seaborn | Static plots for the thesis | TensorBoard (used during development only) |
| Pydantic + pydantic-settings | Typed application configuration | dataclasses, plain `os.environ` |
| YAML (`config/models.yaml`) | Model plugin registry | Python module with class lists |
| uv | Package manager and lockfile | pip + pip-tools, Poetry |
| nox | Test session orchestration | tox |
| pytest | Test framework | unittest |
| ruff | Linter and import sorter | flake8 + isort + pylint |
| mypy | Static type checker | pyright |
| GNU Make | Command shortcuts | `just`, plain shell scripts |
| LabelMe | Ground-truth bounding-box annotation | CVAT, Roboflow |

The stack is biased toward tools that have a well-defined contract with the rest of the system: Pydantic settings are typed, the plugin registry is schema-validated at startup, `uv.lock` reproduces installations exactly, and `nox` enforces session isolation. Each of these choices reduces the surface area where silent errors can hide, which matters for a thesis that depends on reproducible benchmark numbers. The trade-off is some extra build-time discipline up front; the benefit, visible in Chapter 7, is that the experimental results presented there can be regenerated from the repository state with one Makefile invocation.
