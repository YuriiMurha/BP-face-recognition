.PHONY: setup install run evaluate lint clean test train train-cpu train-wsl verify-wsl

# Configurable Variables
# Change these to match your setup
WSL_WORKDIR ?= d:/Coding/Personal/BP-face-recognition
WSL_DISTRO ?= Ubuntu-22.04
PYTHON = uv run python
PYTHONPATH = src

# Convert Windows path to WSL path
WSL_PATH = /mnt/$(subst :,,$(subst \,/,$(WSL_WORKDIR)))

setup:
	uv sync

install: setup

run: setup
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/bp_face_recognition/main.py

train: setup
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/bp_face_recognition/vision/training/production_trainer.py $(args)

train-cpu: setup
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/bp_face_recognition/vision/training/production_trainer.py --force-cpu $(args)

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
		python src/bp_face_recognition/vision/training/production_trainer.py \
		--backbone $(or $(backbone),EfficientNetB0) \
		--epochs $(or $(epochs),20) \
		$(if $(dataset),--dataset $(dataset),) \
		$(if $(filter false,$(fine_tune)),--no-fine-tune,)"

train-wsl-quick:
	@echo "Quick test training (5 epochs)..."
	$(MAKE) train-wsl backbone=MobileNetV3Small epochs=5

# Train EfficientNetB0 on all available datasets (auto-discovered from augmented folder)
# Uses augmented data by default - set use_augmented=false to use cropped data
# Note: Fine-tuning disabled by default for augmented data due to GPU memory constraints
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
		python src/bp_face_recognition/vision/training/production_trainer.py \
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
		python src/bp_face_recognition/vision/training/production_trainer.py \
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

# Setup WSL GPU Environment
setup-wsl-gpu:
	@echo "Setting up WSL GPU environment..."
	@echo "This will install CUDA toolkit and TensorFlow with GPU support"
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && \
		chmod +x scripts/setup_gpu_wsl.sh && \
		bash scripts/setup_gpu_wsl.sh"

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

# Fix GPU library paths (if GPU not detected)
fix-wsl-gpu:
	@echo "Fixing GPU library paths..."
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && \
		chmod +x scripts/fix_gpu_libs.sh && \
		bash scripts/fix_gpu_libs.sh"

# Development Commands
test:
	uv run nox -s tests

lint:
	uv run nox -s lint

type-check:
	uv run nox -s type_check

clean:
	if exist .nox rmdir /s /q .nox
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

# Evaluation Commands
evaluate: setup
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/bp_face_recognition/evaluation/evaluate_methods.py

benchmark: setup
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/benchmark_quantization_mediapipe.py

# Data Processing
init-dataset: setup
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/scripts/init_dataset.py $(name) $(args)

register: setup
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/scripts/register_person.py $(name) $(dir) --db $(db)

# Quantization Commands
quantize: setup
	@echo "Quantizing model: $(model)"
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) src/scripts/quantize_model.py --model $(model) --type $(or $(type),float16) --output $(or $(output),src/bp_face_recognition/models/)

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
