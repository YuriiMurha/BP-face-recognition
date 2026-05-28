# Chapter 6: Practical Implementation and Codebase Overview

This chapter describes the design of the face recognition system at the level of components, layers, and interactions, rather than at the level of files and classes. It explains why the system was organized this way, what the abstractions are, and how they fit together to support the experiments reported in Chapter 7. A complementary system manual in Appendix B documents the concrete repository structure for readers who wish to extend or modify the codebase; this chapter stays at the conceptual level.

## 6.1 Codebase Structure and Organization

The codebase follows a layered Python source layout that distinguishes importable library code from standalone tools. Library code is grouped by concern: a vision layer that contains the detection and recognition implementations, a services layer that orchestrates the pipeline end to end, a database layer that persists registered identities, a preprocessing layer that prepares datasets, and an evaluation layer that runs the experiments reported in Chapter 7. Standalone tools — training launchers, plot generators, benchmark drivers — sit outside the library so that they can be invoked from the command line without importing the entire stack.

The separation between library and tools is deliberate. A reader exploring the codebase to understand how the system works should read the library; a reader running the benchmarks should run the tools. The two have different testing expectations: the library is covered by a pytest suite organized by marker (unit, integration, slow), while the tools are exercised end to end through the Makefile.

## 6.2 System Architecture

### 6.2.1 Layered Architecture

The runtime architecture has three layers: a vision layer that knows how to find and recognize faces in a single image, a service layer that orchestrates a complete recognition request across detection, recognition, database lookup, and result aggregation, and an application layer that wires the service to a camera source and to a display window.

The boundary between the vision layer and the service layer is intentionally narrow: the vision layer exports two abstract interfaces, one for detectors and one for recognizers, and concrete implementations of each plug into a registry. The service layer holds no knowledge of which detector or recognizer it is using; it interacts only with the abstract interfaces. The benefit is concrete and visible in the experimental work: the same service code drives every detector–recognizer combination evaluated in Chapter 7, so cross-method comparisons are honest rather than relying on bespoke harnesses per method.

### 6.2.2 Service Layer Design

The pipeline service accepts a raw frame, invokes the configured detector to obtain bounding boxes, crops each detected face to the input resolution expected by the recognizer, asks the recognizer for an embedding, queries the identity database for nearest-neighbour matches, applies a similarity threshold, and returns annotated results. The same service implementation supports both real-time camera operation and batch evaluation against pre-collected frames; the only difference between the two is the source of the input image.

A separate service handles closed-set classification. It bypasses the database entirely and reads the predicted class directly from the recognizer's classifier head. This split exists because the two paradigms have meaningfully different evaluation protocols and meaningfully different deployment characteristics, and surfacing the distinction at the service-layer interface makes the rest of the system clearer.

### 6.2.3 Dual Recognition Modes

The system supports two recognition modes that share a single detection front end. Open-set mode treats recognition as nearest-neighbour search in an embedding space and accepts new identities at runtime by appending embeddings to the database. Closed-set mode treats recognition as classification against a fixed identity set determined at training time. Chapter 4 derives the algorithmic differences; this chapter notes only that the two modes are exposed as separate application entry points because the operational lifecycle is different: open-set deployments support live enrolment while closed-set deployments require model retraining to add identities.

### 6.2.4 Configuration Management

Application configuration is typed and centralised. A settings object exposes filesystem directories, camera parameters, the default model selection, and database credentials as typed fields with sensible defaults. Values come from environment variables or from a project-local configuration file, and field types are validated at startup so that misspelled variables produce immediate errors rather than confusing runtime failures later. The same settings object is the single source of truth used throughout the codebase.

## 6.3 Detection Pipeline

### 6.3.1 Plugin Interface Design

Detectors implement a single abstract method that maps an image to a list of bounding boxes. The interface is narrow on purpose: it forces every detector to expose the same contract regardless of the library that backs it, which is what makes the Chapter 7 comparison credible. A detector that returned landmarks or confidence scores in addition would still satisfy the contract, but the service layer only uses the bounding-box list, so additional outputs would be inert.

### 6.3.2 Available Detectors

Five detector implementations are registered: MediaPipe BlazeFace (the default, GPU-capable), MTCNN (slow but high-recall), Haar Cascade (the classical baseline), dlib HOG (the classical learned detector), and a thin wrapper around the face_recognition library. Algorithmic details of each method are derived in Chapter 4.2; the practical reason for keeping all five available is that they sit at different operating points on the speed–quality curve evaluated in Chapter 7, and reducing the comparison to a single representative would weaken the empirical claim.

### 6.3.3 Plugin Selection and Switching

Available detectors and recognizers are declared in a YAML registry alongside their default parameters and the fully qualified path to their implementation class. At startup the registry is parsed, classes are resolved through dynamic import, and the resulting plugin objects are made available to the service layer through a factory function. Adding a new detector or recognizer is therefore a two-step operation: implement the appropriate interface in a new module, then declare an entry in the registry. No changes to the service layer or the application entry points are required.

The registry is also the mechanism by which the experimental work was carried out reproducibly. Each detector–recognizer combination evaluated in Chapter 7 corresponds to a registry entry, and the same configuration that selects the model at runtime selects it during the benchmark run.

### 6.3.4 Multi-Tier Fallback in MediaPipe

MediaPipe's runtime supports two backends: a GPU delegate, which is faster but requires hardware and driver compatibility, and a CPU implementation, which works on every platform. The system attempts to use the GPU delegate first, falls back to the CPU implementation if the GPU is unreachable, and finally falls back to the OpenCV Haar cascade if even the CPU MediaPipe path fails. The fallback chain is logged so that silent degradation is visible to the operator; this matters because a GPU-to-CPU fall-back is performance-only while a fall-back to Haar is a meaningful accuracy regression.

## 6.4 Recognition Pipeline

### 6.4.1 FaceNet Embedding Extraction

Every recognition strategy compared in Chapter 7 uses an InceptionResNetV1 backbone derived from FaceNet, with weights initialised from a publicly available checkpoint trained on a large celebrity corpus. The backbone produces a 512-dimensional embedding for each input crop, which is L2-normalised so that all embeddings lie on the unit hypersphere. Cosine similarity between two normalised embeddings is the operating measure used throughout open-set evaluation, while the same embeddings feed into a classification head for closed-set evaluation.

### 6.4.2 Three Training Strategies

Three fine-tuning strategies for adapting the pre-trained backbone to the 14-identity custom dataset are implemented and compared: feature extraction with a frozen backbone, progressive unfreezing of the backbone in four phases with decaying learning rates, and metric learning with triplet loss using random online mining. The algorithmic content of each is in Chapter 4.5; the implementation contributes a common training harness so that all three strategies share the same data pipeline, the same evaluation protocol, and the same logging format. Per-strategy training history is serialised to JSON alongside each saved model so that training curves can be regenerated for the thesis from a single source.

### 6.4.3 Open-Set and Closed-Set Implementations

Open-set recognition is implemented as embedding plus retrieval. Registered identities are stored as embedding vectors in a CSV-backed database; the recognition service computes the cosine similarity between the probe embedding and every entry, and returns the closest match above a configurable threshold. The database also persists basic metadata (identity label, registration timestamp) but the matching itself depends only on the embedding.

Closed-set recognition is implemented as direct classification using the recognizer's classifier head. No database lookup occurs; the predicted identity is the argmax of the softmax output, gated by a confidence threshold. The two implementations share the detection front end and the cropping logic, which is the architectural property that makes Chapter 7's comparison fair: any difference between the two modes is attributable to the recognition step rather than to upstream preprocessing.

### 6.4.4 Model Loading and Switching

Pre-trained and fine-tuned recognizers are loaded by the same factory used for detectors, with the registry pointing to either full or quantized model variants. Switching between the full and the quantized version of the same recognizer is a configuration change rather than a code change, which is what made the size-versus-accuracy comparison in Section 6.6.1 straightforward to run.

## 6.5 Preprocessing and Data Pipeline

The preprocessing pipeline is documented in detail in Chapter 5. From the implementation perspective the relevant points are that the three stages — cropping, splitting, augmentation — are each invokable independently through Makefile targets, that they share a common dataset-discovery convention that requires no per-dataset configuration, and that they are idempotent: re-running a stage on unchanged input produces unchanged output. This last property is what makes the experimental results in Chapter 7 reproducible from a fresh repository checkout.

Augmentation is applied to the training partition only and writes the augmented variants to disk. This trades disk space for training-time speed and eliminates augmentation as a source of run-to-run variance in the training curves: every training run for a given seed sees exactly the same sequence of augmented examples. Validation and test partitions remain unaugmented so that reported metrics reflect behaviour on in-distribution images.

## 6.6 Cross-Platform Support

### 6.6.1 Model Quantization

TensorFlow Lite is used to produce quantized variants of the recognizer for resource-constrained deployment. Three quantization strategies were trialled — dynamic-range, float16, and full int8 — with dynamic-range chosen as the default because it reduces model size by approximately 75% at negligible accuracy cost on the closed-set benchmark. The quantized variant exposes the same recognizer interface as the full Keras model, so consumers of the recognition service do not need to be aware of the underlying representation.

### 6.6.2 GPU Delegate Support

GPU acceleration is supported through MediaPipe's GPU delegate for detection and through TensorFlow's GPU backend for training. Detection runs comfortably on CPU and only invokes the GPU delegate when one is available and reachable. Training, by contrast, is GPU-bound for the FaceNet fine-tuning strategies: a four-phase progressive-unfreezing run on the 14-identity dataset takes approximately 50 minutes on a modest consumer GPU and several hours on CPU. The training tools therefore document GPU as the recommended path, with CPU available as a fallback for quick experiments.

### 6.6.3 Platform-Specific Dependency Management

The system runs on Windows and on Ubuntu under WSL2. Platform divergence is handled at the dependency level by the package manager: the same lockfile resolves to the GPU-capable TensorFlow wheel on Linux and the CPU-only wheel on Windows, and the dlib library is built from source on Linux but installed from a pre-built wheel on Windows where building from source requires additional toolchains. The user is shielded from these details: a single setup command produces a working environment on either platform.

## 6.7 Build and Development Tools

The project uses uv for dependency resolution and lockfile management, nox for orchestrating test sessions in isolated virtual environments, pytest as the test framework, ruff for linting and import sorting, mypy for static type checking, and GNU Make for command-line shortcuts that wrap the underlying invocations. The rationale for each selection is in Chapter 3; the practical effect on the implementation is that quality gates are short to express and fast to run. Lint, type-check, and unit-test passes complete in seconds, which is what makes the cross-validation workflow in Chapter 7 feasible without disrupting interactive development.

## 6.8 Summary of Design Decisions

Four design decisions structure the implementation. First, the narrow vision-layer interface — detectors return rectangles, recognizers return embeddings — makes cross-method comparison credible because the surrounding code is identical for every method evaluated. Second, the YAML model registry decouples plugin selection from code, which is what allowed twenty-plus detector–recognizer combinations to be benchmarked without per-combination harness code. Third, on-disk augmentation eliminates augmentation as a run-to-run confound, which is what makes the cross-validation analysis in Chapter 7 interpretable. Fourth, the dual-paradigm split exposes open-set and closed-set recognition as distinct services rather than as flags on a single service, because the operational semantics differ even when the underlying backbone is shared.

Together these choices reduce the surface area where silent inconsistencies can hide, which is the property a thesis whose conclusions depend on cross-method numerical comparisons needs from its implementation.
