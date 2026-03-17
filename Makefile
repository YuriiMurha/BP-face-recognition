.PHONY: setup install run train train-cpu test lint type-check clean evaluate benchmark

# Configurable Variables
WSL_WORKDIR ?= d:/Coding/Personal/BP-face-recognition
WSL_DISTRO ?= Ubuntu-22.04
PYTHON = uv run python
PYTHONPATH = src

# Convert Windows path to WSL path
WSL_PATH = /mnt/$(subst :,,$(subst \,/,$(WSL_WORKDIR)))

setup:
	uv sync

install: setup

# Run with default recognizer (metric_efficientnetb0_128d)
run: setup
	$(PYTHON) src/bp_face_recognition/main.py

# Run with specific recognizer: make run recog=metric_efficientnetb0_128d, dlib_v1
run-recog: setup
	$(PYTHON) src/bp_face_recognition/main.py --recognizer $(recog)

# Run with specific recognizer and threshold
run-full: setup
	$(PYTHON) src/bp_face_recognition/main.py --recognizer $(recog) --threshold $(threshold)

# Register a new person from camera
# Usage: make register name="Yurii" (uses default from config/models.yaml)
# Override: make register name="Yurii" recog=facenet_pu
register: setup
	$(PYTHON) src/scripts/register_from_camera.py "$(name)" $(if $(recog),--recognizer $(recog),)

# Register with FaceNet PU (recommended)
register-pu: setup
	@echo "Registering with FaceNet PU (99.15% accuracy)..."
	$(PYTHON) src/scripts/register_from_camera.py "$(name)" --recognizer facenet_pu

# Register with FaceNet TL
register-tl: setup
	@echo "Registering with FaceNet TL (92.84% accuracy)..."
	$(PYTHON) src/scripts/register_from_camera.py "$(name)" --recognizer facenet_tl

# Register with FaceNet TLoss
register-tloss: setup
	@echo "Registering with FaceNet TLoss (94.63% accuracy)..."
	$(PYTHON) src/scripts/register_from_camera.py "$(name)" --recognizer facenet_tloss

# ============================================================
# Data Preprocessing Pipeline
# ============================================================

# Crop faces from raw datasets (uses dynamic discovery)
prepare-crop: setup
	$(PYTHON) src/bp_face_recognition/preprocessing/crop_faces.py --dataset $(dataset)

# Crop all datasets (dynamic discovery)
prepare-crop-all: setup
	$(PYTHON) src/bp_face_recognition/preprocessing/crop_faces.py --dataset all

# Split LFW dataset into train/val/test
prepare-split-lfw: setup
	$(PYTHON) src/bp_face_recognition/preprocessing/split_lfw.py

# Augment cropped datasets
prepare-augment: setup
	$(PYTHON) src/bp_face_recognition/preprocessing/augmentation.py --dataset $(dataset)

# Augment all datasets (dynamic discovery)
prepare-augment-all: setup
	$(PYTHON) src/bp_face_recognition/preprocessing/augmentation.py --dataset all

# Full preprocessing pipeline
prepare-all:
	@echo "Running full preprocessing pipeline..."
	@echo "1. Splitting LFW..."
	$(MAKE) prepare-split-lfw
	@echo "2. Cropping faces..."
	$(MAKE) prepare-crop-all
	@echo "3. Augmenting datasets..."
	$(MAKE) prepare-augment-all
	@echo "Preprocessing complete!"

# ============================================================
# Training Commands
# ============================================================

train: setup
	$(PYTHON) src/bp_face_recognition/vision/training/classifier/trainer.py $(args)

train-classifier: train

train-metric: setup
	$(PYTHON) src/bp_face_recognition/vision/training/metric/trainer.py $(args)

# Experiment C: Fine-tune classifier for embeddings (recommended over triplet loss)
# Usage: make train-finetune-wsl datasets="lfw,webcam,seccam" epochs_phase1=15 epochs_phase2=10
train-finetune-wsl:
	@echo "Training fine-tune classifier in WSL2..."
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && \
		export PATH=\"/root/.local/bin:\$$PATH\" && \
		export PYTHONPATH=$(WSL_PATH)/src:\$$PYTHONPATH && \
		export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit && \
		source .venv-wsl/bin/activate && \
		python -m bp_face_recognition.vision.training.metric.finetune_trainer \
		--datasets $(or $(datasets),lfw,webcam,seccam) \
		--backbone $(or $(backbone),EfficientNetB0) \
		--dim $(or $(dim),128) \
		--epochs-phase1 $(or $(epochs_phase1),15) \
		--epochs-phase2 $(or $(epochs_phase2),10) \
		--batch-size $(or $(batch_size),32)"

# Train metric model in WSL
# Usage: make train-metric-wsl datasets="lfw,webcam,seccam_2" epochs=20
# Default: lfw only
train-metric-wsl:
	@echo "Training metric model in WSL2..."
	@echo "Datasets: $(datasets)"
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && \
		export PATH=\"/root/.local/bin:\$$PATH\" && \
		export PYTHONPATH=$(WSL_PATH)/src:\$$PYTHONPATH && \
		export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit && \
		source .venv-wsl/bin/activate && \
		python -m bp_face_recognition.vision.training.metric.trainer \
		--dataset $(or $(datasets),lfw) \
		--backbone $(or $(backbone),EfficientNetB0) \
		--dim $(or $(dim),128) \
		--epochs $(or $(epochs),20) \
		--batch-size $(or $(batch_size),8)"

train-cpu: setup
	$(PYTHON) src/bp_face_recognition/vision/training/classifier/trainer.py --force-cpu $(args)

# WSL Training Commands
# Use --dataset <name> to train on specific dataset, or omit to train all available datasets
train-wsl:
	@echo "Training in WSL2..."
	@echo "Use --dataset <name> to train on specific dataset, or omit to train all available"
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && \
		export PATH=\"/root/.local/bin:\$$PATH\" && \
		export PYTHONPATH=$(WSL_PATH)/src:\$$PYTHONPATH && \
		export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit && \
		source .venv-wsl/bin/activate && \
		python src/bp_face_recognition/vision/training/classifier/trainer.py \
		--backbone $(or $(backbone),EfficientNetB0) \
		--epochs $(or $(epochs),20) \
		$(if $(dataset),--dataset $(dataset),) \
		$(if $(filter false,$(fine_tune)),--no-fine-tune,)"

train-wsl-quick:
	@echo "Quick test training (5 epochs)..."
	$(MAKE) train-wsl backbone=MobileNetV3Small epochs=5

# Train EfficientNetB0 on all available datasets (auto-discovered from augmented folder)
# Uses augmented data by default - set use_augmented=false to use cropped data
train-all-datasets:
	@echo "Training EfficientNetB0 on all available datasets (auto-discovered)..."
	@echo "Using augmented data by default"
	@echo "Optimizations: memory growth, VRAM limit (3.5GB), mixed precision (float16)"
	@echo "Note: Fine-tuning disabled for augmented data (OOM prevention)"
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && \
		export PATH=\"/root/.local/bin:\$$PATH\" && \
		export PYTHONPATH=$(WSL_PATH)/src:\$$PYTHONPATH && \
		export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit && \
		source .venv-wsl/bin/activate && \
		python src/bp_face_recognition/vision/training/classifier/trainer.py \
		--backbone $(or $(backbone),EfficientNetB0) \
		--epochs $(or $(epochs),20) \
		--batch-size $(or $(batch_size),8) \
		--no-fine-tune"

# Train on a specific dataset (use dataset=seccam, dataset=seccam_2, or dataset=webcam)
train-one:
	@echo "Training EfficientNetB0 on dataset: $(dataset)"
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && \
		export PATH=\"/root/.local/bin:\$$PATH\" && \
		export PYTHONPATH=$(WSL_PATH)/src:\$$PYTHONPATH && \
		export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit && \
		source .venv-wsl/bin/activate && \
		python src/bp_face_recognition/vision/training/classifier/trainer.py \
		--backbone $(or $(backbone),EfficientNetB0) \
		--epochs $(or $(epochs),20) \
		--batch-size $(or $(batch_size),8) \
		--dataset $(dataset) \
		$(if $(filter false,$(fine_tune)),--no-fine-tune,)"

verify-wsl:
	@echo "Verifying WSL environment..."
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && \
		source .venv-wsl/bin/activate && \
		python -c 'import tensorflow; print(f\"TensorFlow: {tensorflow.__version__}\")'"

# Setup WSL Environment
setup-wsl:
	@echo "Setting up WSL environment..."
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && \
		curl -LsSf https://astral.sh/uv/install.sh | sh && \
		export PATH=\"/root/.local/bin:\$$PATH\" && \
		uv venv .venv-wsl && \
		source .venv-wsl/bin/activate && \
		uv pip install tensorflow opencv-python numpy pillow scikit-learn pydantic pyyaml"

# Verify GPU setup
verify-wsl-gpu:
	@echo "Verifying GPU setup..."
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && \
		source .venv-wsl/bin/activate && \
		python -c 'import tensorflow as tf; \
		print(\"TensorFlow:\", tf.__version__); \
		gpus = tf.config.list_physical_devices(\"GPU\"); \
		print(\"GPUs detected:\", len(gpus)); \
		print(\"Details:\", gpus)'"

# Check GPU status
gpu-status:
	@echo "Checking GPU status..."
	wsl -d $(WSL_DISTRO) nvidia-smi

# Test Commands (via nox)
test:
	uv run nox -s tests

test-config:
	uv run nox -s test_config

test-training:
	uv run nox -s test_training

test-mediapie:
	uv run nox -s test_mediapipe

test-integration:
	uv run nox -s test_integration

# Camera Tests
test-camera:
	uv run nox -s test_camera

test-camera-integration:
	uv run nox -s test_camera_integration

# Run camera stream viewer (requires camera)
camera-view:
	$(PYTHON) src/bp_face_recognition/evaluation/show_stream.py

# Run full face recognition app with camera (use 'make run' instead)
# make run

lint:
	uv run nox -s lint

type-check:
	uv run nox -s type_check

clean:
	if exist .nox rmdir /s /q .nox
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

# Evaluation
evaluate: setup
	$(PYTHON) src/bp_face_recognition/evaluation/evaluate_methods.py

benchmark: setup
	$(PYTHON) src/benchmark_quantization_mediapipe.py

# Data Processing
init-dataset: setup
	$(PYTHON) src/scripts/init_dataset.py $(name) $(args)

# Quantization
quantize: setup
	$(PYTHON) src/scripts/quantize_model.py --model $(model) --type $(or $(type),float16) --output $(or $(output),src/bp_face_recognition/models/)

# Quantize model in WSL (for GPU-trained models)
quantize-wsl:
	@echo "Quantizing model in WSL: $(model)"
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && \
		export PATH=\"/root/.local/bin:\$$PATH\" && \
		export PYTHONPATH=$(WSL_PATH)/src:\$$PYTHONPATH && \
		export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit && \
		source .venv-wsl/bin/activate && \
		python src/scripts/quantize_model.py \
		--model $(model) \
		--type $(or $(type),float16) \
		--output $(or $(output),src/bp_face_recognition/models/)"

# Quantize all GPU models in WSL
quantize-all:
	@echo "Quantizing all GPU-trained models in WSL..."
	@echo "1. Quantizing webcam model..."
	$(MAKE) quantize-wsl model=src/bp_face_recognition/models/efficientnetb0_webcam_gpu_final.keras type=$(or $(type),float16)
	@echo "2. Quantizing seccam_2 model..."
	$(MAKE) quantize-wsl model=src/bp_face_recognition/models/efficientnetb0_seccam_2_gpu_final.keras type=$(or $(type),float16)
	@echo "3. Quantizing seccam model..."
	$(MAKE) quantize-wsl model=src/bp_face_recognition/models/efficientnetb0_seccam_gpu_final.keras type=$(or $(type),float16)
	@echo "All models quantized successfully!"

# Quantize a specific dataset model (use dataset=seccam, dataset=seccam_2, or dataset=webcam)
quantize-one:
	@echo "Quantizing model for dataset: $(dataset)"
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && \
		export PATH=\"/root/.local/bin:\$$PATH\" && \
		export PYTHONPATH=$(WSL_PATH)/src:\$$PYTHONPATH && \
		export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit && \
		source .venv-wsl/bin/activate && \
		python src/scripts/quantize_model.py \
		--model src/bp_face_recognition/models/efficientnetb0_$(dataset)_gpu_final.keras \
		--type $(or $(type),float16) \
		--output src/bp_face_recognition/models/"

# ============================================================
# FaceNet Fine-Tuning Commands
# ============================================================

# Train FaceNet with transfer learning (Option A) - Fast, 4 min, 92.84% accuracy
train-facenet-transfer:
	@echo "Training FaceNet with Transfer Learning (Option A)..."
	$(PYTHON) src/bp_face_recognition/vision/training/finetune/facenet_transfer_trainer.py \
		--epochs $(or $(epochs),20) --batch-size $(or $(batch_size),32)

# Train FaceNet with progressive unfreezing (Option B) - Best accuracy, 50 min, 99.15%
train-facenet-progressive:
	@echo "Training FaceNet with Progressive Unfreezing (Option B)..."
	$(PYTHON) src/bp_face_recognition/vision/training/finetune/facenet_progressive_trainer.py \
		--epochs-per-phase $(or $(epochs_per_phase),5) --batch-size $(or $(batch_size),32)

# Train FaceNet with triplet loss (Option C) - Metric learning, 90 min, ~97-98%
train-facenet-triplet:
	@echo "Training FaceNet with Triplet Loss (Option C)..."
	$(PYTHON) src/bp_face_recognition/vision/training/finetune/facenet_triplet_trainer.py \
		--epochs $(or $(epochs),30) --batch-size $(or $(batch_size),32) --margin $(or $(margin),0.2)

# Compare all FaceNet fine-tuning results
compare-facenet-results:
	@echo "Generating FaceNet fine-tuning comparison visualizations..."
	$(PYTHON) src/bp_face_recognition/vision/training/finetune/visualize_preliminary_results.py --final

# ============================================================
# Repository Cleanup Commands
# ============================================================

clean-training-logs:
	@echo "Cleaning training log files..."
	-rm -f training_*.log *.log
	-rm -f *.pid
	@echo "Training logs cleaned."

clean-temp-files:
	@echo "Cleaning temporary and cache files..."
	-rm -rf .nox
	-find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	-find . -type f -name "*.pyc" -delete 2>/dev/null || true
	-find . -type f -name "*.pyo" -delete 2>/dev/null || true
	-find . -type f -name "*~" -delete 2>/dev/null || true
	@echo "Temporary files cleaned."

clean-all: clean clean-temp-files clean-training-logs
	@echo "Full cleanup complete."

# ============================================================
# FaceNet Fine-Tuned Models - Professional Naming Convention
# TL = Transfer Learning | PU = Progressive Unfreezing | TLoss = Triplet Loss
# ============================================================

# Training Commands
# Transfer Learning (TL) - Fast baseline, 4 min, 92.84%
train-facenet-tl:
	@echo "Training FaceNet Transfer Learning (TL) - 92.84% target..."
	$(PYTHON) src/bp_face_recognition/vision/training/finetune/facenet_transfer_trainer.py \
		--epochs $(or $(epochs),20) --batch-size $(or $(batch_size),32)

# Progressive Unfreezing (PU) - Best accuracy, 50 min, 99.15% ⭐
train-facenet-pu:
	@echo "Training FaceNet Progressive Unfreezing (PU) - 99.15% target..."
	$(PYTHON) src/bp_face_recognition/vision/training/finetune/facenet_progressive_trainer.py \
		--epochs-per-phase $(or $(epochs_per_phase),5) --batch-size $(or $(batch_size),32)

# Triplet Loss (TLoss) - Metric learning, 90 min, 94.63%
train-facenet-tloss:
	@echo "Training FaceNet Triplet Loss (TLoss) - 94.63% target..."
	$(PYTHON) src/bp_face_recognition/vision/training/finetune/facenet_triplet_trainer.py \
		--epochs $(or $(epochs),30) --batch-size $(or $(batch_size),32) --margin $(or $(margin),0.2)

# WSL GPU versions
train-facenet-tl-wsl:
	@echo "Training FaceNet TL in WSL (GPU)..."
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && source .venv-wsl/bin/activate && \
		python src/bp_face_recognition/vision/training/finetune/facenet_transfer_trainer.py \
		--epochs $(or $(epochs),20) --batch-size $(or $(batch_size),32)"

train-facenet-pu-wsl:
	@echo "Training FaceNet PU in WSL (GPU)..."
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && source .venv-wsl/bin/activate && \
		python src/bp_face_recognition/vision/training/finetune/facenet_progressive_trainer.py \
		--epochs-per-phase $(or $(epochs_per_phase),5) --batch-size $(or $(batch_size),32)"

train-facenet-tloss-wsl:
	@echo "Training FaceNet TLoss in WSL (GPU)..."
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && source .venv-wsl/bin/activate && \
		python src/bp_face_recognition/vision/training/finetune/facenet_triplet_trainer.py \
		--epochs $(or $(epochs),30) --batch-size $(or $(batch_size),32) --margin $(or $(margin),0.2)"

# Testing Commands (with Camera)
test-facenet-tl: setup
	@echo "Testing FaceNet Transfer Learning (92.84%) with camera..."
	$(PYTHON) src/bp_face_recognition/main.py --recognizer facenet_tl

test-facenet-pu: setup
	@echo "Testing FaceNet Progressive Unfreezing (99.15%) with camera - RECOMMENDED..."
	$(PYTHON) src/bp_face_recognition/main.py --recognizer facenet_pu

test-facenet-tloss: setup
	@echo "Testing FaceNet Triplet Loss (94.63%) with camera..."
	$(PYTHON) src/bp_face_recognition/main.py --recognizer facenet_tloss

# Model Switching Commands
switch-pu:
	@echo "Switching to FaceNet PU (Progressive Unfreezing - 99.15%)..."
	$(PYTHON) scripts/switch_model.py pu

switch-tl:
	@echo "Switching to FaceNet TL (Transfer Learning - 92.84%)..."
	$(PYTHON) scripts/switch_model.py tl

switch-tloss:
	@echo "Switching to FaceNet TLoss (Triplet Loss - 94.63%)..."
	$(PYTHON) scripts/switch_model.py tloss

# Database Management
clear-db:
	@echo "Clearing face database (backup created automatically)..."
	$(PYTHON) scripts/switch_model.py clear

# Combined: Switch model and clear database
reset-pu: switch-pu clear-db
	@echo "Reset complete: FaceNet PU selected, database cleared"

reset-tl: switch-tl clear-db
	@echo "Reset complete: FaceNet TL selected, database cleared"

reset-tloss: switch-tloss clear-db
	@echo "Reset complete: FaceNet TLoss selected, database cleared"

# Evaluation Commands
evaluate-facenet-all:
	@echo "Comprehensive evaluation of all FaceNet models..."
	$(PYTHON) src/bp_face_recognition/evaluation/evaluate_comprehensive.py \
		--models src/bp_face_recognition/models/finetuned/facenet_transfer_v1.0.keras \
		       src/bp_face_recognition/models/finetuned/facenet_progressive_v1.0.keras \
		       src/bp_face_recognition/models/finetuned/facenet_triplet_best.keras \
		--output results/evaluation/facenet_comparison

evaluate-facenet-tl-quick:
	@echo "Quick evaluation: Transfer Learning..."
	$(PYTHON) src/bp_face_recognition/evaluation/evaluate_simple.py \
		--model src/bp_face_recognition/models/finetuned/facenet_transfer_v1.0.keras

evaluate-facenet-pu-quick:
	@echo "Quick evaluation: Progressive Unfreezing..."
	$(PYTHON) src/bp_face_recognition/evaluation/evaluate_simple.py \
		--model src/bp_face_recognition/models/finetuned/facenet_progressive_v1.0.keras

evaluate-facenet-tloss-quick:
	@echo "Quick evaluation: Triplet Loss..."
	$(PYTHON) src/bp_face_recognition/evaluation/evaluate_simple.py \
		--model src/bp_face_recognition/models/finetuned/facenet_triplet_best.keras
