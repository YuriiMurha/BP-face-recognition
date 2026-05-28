# Chapter 3: Tools and Libraries

## 3.1 Introduction

The choice of tools shapes what is feasible to build and what is feasible to measure. A face recognition pipeline that targets surveillance camera footage has to handle image acquisition, geometric face localization, deep feature extraction, similarity scoring against a database, and reproducible training of the underlying neural networks. No single library covers that span, so the system is assembled from a stack of specialized components. This chapter introduces those components and the rationale for each selection, organized by the role each plays in the pipeline.

One constraint runs through every choice: the project has to build and run on both Windows for day-to-day development and Ubuntu under WSL2 for GPU training. That dual-platform requirement rules out several otherwise attractive options and shapes the discussion below.

## 3.2 Programming Language and Deep Learning Stack

The implementation language is Python 3.11. Python remains the default in computer vision and machine learning research because the major deep learning frameworks expose their primary APIs in Python and almost every published model checkpoint is distributed with Python loading code. Python 3.11 specifically was selected for compatibility with TensorFlow 2.15, which is the latest release that still supports the Keras API used by the pre-trained FaceNet weights the project relies on.

TensorFlow 2.15 with Keras is the primary deep learning framework [CITE: abadi2016tensorflow; chollet2015keras]. TensorFlow was chosen over PyTorch because the original FaceNet weights are distributed for TensorFlow and because the MediaPipe Tasks API for GPU-accelerated face detection is TensorFlow-native. Using a single framework on both sides of the pipeline simplifies dependency management and removes one source of floating-point divergence between detection and recognition.

Pre-trained FaceNet weights are obtained through the keras-facenet package [CITE: kerasfacenet], which bundles InceptionResNetV1 with weights converted from David Sandberg's well-known TensorFlow checkpoint. TensorFlow Lite is used for model quantization, with dynamic-range quantization chosen as the default because it reduces model size by roughly three-quarters at minimal accuracy cost.

## 3.3 Computer Vision Libraries

Five computer vision libraries appear in the pipeline. Each has a specific role and was kept rather than removed because no single library covered all of them.

OpenCV [CITE: bradski2000opencv] handles low-level image work: camera capture, colour-space conversion, drawing bounding boxes and identity labels on output frames, file input and output, and the Haar cascade classifier used as one of the four detection baselines. The library is mature, fast, and ships with pre-built wheels on both target platforms.

MediaPipe [CITE: lugaresi2019mediapipe] provides the default face detector through its BlazeFace model [CITE: bazarevsky2019blazeface]. BlazeFace processes frames at over 200 FPS on CPU and was the only detector in the comparison that supported a GPU delegate. The trade-off, established quantitatively in Chapter 7, is that BlazeFace's single-scale anchor design misses many small or oblique faces.

dlib [CITE: king2009dlib] supplies the HOG-based face detector and a 128-dimensional face embedding network used as one of the open-set recognition backends. dlib is C++ at heart with Python bindings, which makes it CPU-efficient. The face_recognition library wraps dlib with a higher-level Python API and bundles the canonical embedding model.

MTCNN is consumed through a small PyPI package that re-implements the three-stage cascade described by [CITE: zhang2016mtcnn]. It was retained despite slow inference because its multi-scale pyramid recovers faces that BlazeFace systematically misses; in the ground-truth evaluation it is the highest-recall detector.

The decision to wrap five detectors and several recognition backends behind a common plugin interface is a deliberate research choice. The thesis compares detection methods quantitatively, and that comparison is only credible if each method is an idiomatic instance of the technique rather than a hand-rolled approximation.

## 3.4 Data Processing and Machine Learning Utilities

Image augmentation is performed by Albumentations [CITE: buslaev2020albumentations]. The pipeline chains a resize, a horizontal flip, random brightness and contrast jitter, gamma shifts, additive Gaussian noise, and a small blur. Albumentations was chosen over the alternatives because its composition API is more expressive when transformations have to be probabilistic and conditional, and its output is deterministic given a seed, which matters for reproducible benchmarks.

scikit-learn [CITE: pedregosa2011scikit] supplies the evaluation metrics: per-class precision, recall, F1, balanced accuracy, and the silhouette score used in the embedding-quality analysis. NumPy [CITE: harris2020numpy] is the array type that crosses every interface boundary, and Pandas [CITE: mckinney2010pandas] handles the face database and benchmark result tables. Matplotlib and Seaborn produce the training curves, confusion matrices, and embedding plots that appear in Chapter 7.

## 3.5 Build and Development Tooling

The project uses uv [CITE: astraluv] for dependency resolution and lockfile generation. uv was chosen over pip and Poetry because it resolves dependency graphs in seconds rather than minutes and because a single lockfile works for both Windows and WSL2 installations, with the correct wheel selected for each platform automatically. Tests are organized by nox [CITE: noxproject] into isolated sessions, each running pytest with its own virtual environment to catch the class of bugs that occur when a test silently depends on an unlisted package. Linting and import sorting are handled by ruff [CITE: astralruff], which bundles the older flake8-plus-isort-plus-pylint combination into a single binary that is roughly two orders of magnitude faster. Static type checking uses mypy with module overrides for libraries that ship no type stubs.

Two separate virtual environments are maintained, one for Windows CPU work and one for WSL2 GPU training. The split is necessary because TensorFlow's GPU support is Linux-only and the GPU and CPU TensorFlow wheels cannot coexist in a single environment.

## 3.6 Summary

The stack is biased toward tools with a well-defined contract: typed configuration, a schema-validated plugin registry, an exactly reproducible lockfile, and session-isolated tests. Each of these choices reduces the surface area where silent errors can hide, which matters for a thesis whose conclusions depend on benchmark numbers being regenerable from the repository state.
