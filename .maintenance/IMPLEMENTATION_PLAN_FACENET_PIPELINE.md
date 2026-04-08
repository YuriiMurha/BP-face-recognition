# Implementation Plan: Automated FaceNet Model Pipeline

**Date**: March 12, 2026  
**Status**: Planning Phase  
**Objective**: Create automated, modular pipeline for training, quantizing, evaluating, and registering FaceNet models

---

## 📋 Overview

This plan describes a **modular, cross-platform** approach to managing FaceNet fine-tuned models. Each step (train → quantize → evaluate → register) is **separate** to allow flexibility and avoid long-running unified commands.

---

## Phase 1: Update Models Registry (config/models.yaml)

### What to Add

Add 6 new recognizer entries to `config/models.yaml`:

```yaml
# FaceNet Fine-Tuned Models (Session 14)
recognizers:
  # Option A: Transfer Learning (Frozen Base)
  facenet_transfer:
    class: "vision.recognition.finetuned_recognizer.FinetunedRecognizer"
    model_file: "bp_face_recognition/models/finetuned/facenet_transfer_v1.0.keras"
    version: "1.0"
    description: "FaceNet Transfer Learning (Option A) - 92.84% accuracy, frozen base"
    metadata:
      accuracy: 0.9284
      training_time: "4 min"
      model_size: "93 MB"
      trainable_params: "131K (0.56%)"
      approach: "Transfer Learning"
      paradigm: "closed-set"
      dataset: "combined (7,080 images, 14 classes)"
    default_config:
      input_size: [160, 160]
      normalize: true
      preprocessing: "facenet_standard"  # [-1, 1] range
  
  facenet_transfer_quantized:
    class: "vision.recognition.tflite_recognizer.TFLiteRecognizer"
    model_file: "bp_face_recognition/models/finetuned/facenet_transfer_v1.0_float16.tflite"
    version: "1.0"
    description: "FaceNet Transfer Learning - float16 quantized, ~23 MB"
    metadata:
      accuracy: "~92% (estimated)"
      quantization: "float16"
      compression: "~75%"
      approach: "Transfer Learning"
      paradigm: "closed-set"
    default_config:
      input_size: [160, 160]
      normalize: true

  # Option B: Progressive Unfreezing (BEST)
  facenet_progressive:
    class: "vision.recognition.finetuned_recognizer.FinetunedRecognizer"
    model_file: "bp_face_recognition/models/finetuned/facenet_progressive_v1.0.keras"
    version: "1.0"
    description: "FaceNet Progressive Unfreezing (Option B) - 99.15% accuracy ⭐ RECOMMENDED"
    metadata:
      accuracy: 0.9915
      validation_accuracy: 0.9953
      training_time: "50 min"
      model_size: "272 MB"
      trainable_params: "23.6M (100%)"
      approach: "Progressive Unfreezing"
      paradigm: "closed-set"
      phases: 4
      recommended: true
      dataset: "combined (7,080 images, 14 classes)"
    default_config:
      input_size: [160, 160]
      normalize: true
      preprocessing: "facenet_standard"
  
  facenet_progressive_quantized:
    class: "vision.recognition.tflite_recognizer.TFLiteRecognizer"
    model_file: "bp_face_recognition/models/finetuned/facenet_progressive_v1.0_float16.tflite"
    version: "1.0"
    description: "FaceNet Progressive - float16 quantized, ~68 MB, ~98.5% accuracy"
    metadata:
      accuracy: "~98.5% (estimated)"
      quantization: "float16"
      compression: "75%"
      approach: "Progressive Unfreezing"
      paradigm: "closed-set"
      recommended: true
    default_config:
      input_size: [160, 160]
      normalize: true

  # Option C: Triplet Loss
  facenet_triplet:
    class: "vision.recognition.finetuned_recognizer.FinetunedRecognizer"
    model_file: "bp_face_recognition/models/finetuned/facenet_triplet_best.keras"
    version: "1.0"
    description: "FaceNet Triplet Loss (Option C) - 94.63% accuracy, metric learning"
    metadata:
      accuracy: 0.9463
      training_time: "90 min"
      model_size: "271 MB"
      trainable_params: "23.6M (100%)"
      approach: "Triplet Loss"
      paradigm: "open-set"
      margin: 0.2
      dataset: "combined (7,080 images, 14 classes)"
    default_config:
      input_size: [160, 160]
      normalize: true
      preprocessing: "facenet_standard"
  
  facenet_triplet_quantized:
    class: "vision.recognition.tflite_recognizer.TFLiteRecognizer"
    model_file: "bp_face_recognition/models/finetuned/facenet_triplet_best_float16.tflite"
    version: "1.0"
    description: "FaceNet Triplet Loss - float16 quantized, ~68 MB"
    metadata:
      accuracy: "~94% (estimated)"
      quantization: "float16"
      compression: "75%"
      approach: "Triplet Loss"
      paradigm: "open-set"
    default_config:
      input_size: [160, 160]
      normalize: true

# Add environment profiles for testing
environments:
  # Test FaceNet models
  test_facenet_a:
    detector: "mediapipe_v1"
    recognizer: "facenet_transfer"
    use_gpu: false
    log_level: "INFO"
  
  test_facenet_a_quantized:
    detector: "mediapipe_v1"
    recognizer: "facenet_transfer_quantized"
    use_gpu: false
    log_level: "INFO"
  
  test_facenet_b:
    detector: "mediapipe_v1"
    recognizer: "facenet_progressive"
    use_gpu: false
    log_level: "INFO"
  
  test_facenet_b_quantized:
    detector: "mediapipe_v1"
    recognizer: "facenet_progressive_quantized"
    use_gpu: false
    log_level: "INFO"
  
  test_facenet_c:
    detector: "mediapipe_v1"
    recognizer: "facenet_triplet"
    use_gpu: false
    log_level: "INFO"
  
  test_facenet_c_quantized:
    detector: "mediapipe_v1"
    recognizer: "facenet_triplet_quantized"
    use_gpu: false
    log_level: "INFO"
```

**Note**: Need to create `FinetunedRecognizer` class wrapper for Keras models.

---

## Phase 2: Modular Makefile Commands (Split, Not Unified)

### Training Commands (Individual)

```makefile
# ============================================================
# FaceNet Training Commands (Individual)
# ============================================================

# Option A: Transfer Learning - Fast, 4 min, 92.84%
train-facenet-a:
	@echo "Training FaceNet Option A (Transfer Learning)..."
	$(PYTHON) src/bp_face_recognition/vision/training/finetune/facenet_transfer_trainer.py \
		--epochs $(or $(epochs),20) --batch-size $(or $(batch_size),32)

# Option B: Progressive Unfreezing - Best, 50 min, 99.15%
train-facenet-b:
	@echo "Training FaceNet Option B (Progressive Unfreezing)..."
	$(PYTHON) src/bp_face_recognition/vision/training/finetune/facenet_progressive_trainer.py \
		--epochs-per-phase $(or $(epochs_per_phase),5) --batch-size $(or $(batch_size),32)

# Option C: Triplet Loss - Metric Learning, 90 min, 94.63%
train-facenet-c:
	@echo "Training FaceNet Option C (Triplet Loss)..."
	$(PYTHON) src/bp_face_recognition/vision/training/finetune/facenet_triplet_trainer.py \
		--epochs $(or $(epochs),30) --batch-size $(or $(batch_size),32) --margin $(or $(margin),0.2)

# WSL GPU versions (cross-platform support)
train-facenet-a-wsl:
	@echo "Training FaceNet Option A in WSL (GPU)..."
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && \
		source .venv-wsl/bin/activate && \
		python src/bp_face_recognition/vision/training/finetune/facenet_transfer_trainer.py \
		--epochs $(or $(epochs),20) --batch-size $(or $(batch_size),32)"

train-facenet-b-wsl:
	@echo "Training FaceNet Option B in WSL (GPU)..."
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && \
		source .venv-wsl/bin/activate && \
		python src/bp_face_recognition/vision/training/finetune/facenet_progressive_trainer.py \
		--epochs-per-phase $(or $(epochs_per_phase),5) --batch-size $(or $(batch_size),32)"

train-facenet-c-wsl:
	@echo "Training FaceNet Option C in WSL (GPU)..."
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && \
		source .venv-wsl/bin/activate && \
		python src/bp_face_recognition/vision/training/finetune/facenet_triplet_trainer.py \
		--epochs $(or $(epochs),30) --batch-size $(or $(batch_size),32) --margin $(or $(margin),0.2)"
```

### Quantization Commands (Individual)

```makefile
# ============================================================
# FaceNet Quantization Commands (Individual)
# ============================================================

# Quantize Option A
quantize-facenet-a:
	@echo "Quantizing FaceNet Option A (Transfer Learning)..."
	$(PYTHON) src/scripts/quantize_model.py \
		--model src/bp_face_recognition/models/finetuned/facenet_transfer_v1.0.keras \
		--type float16 \
		--output src/bp_face_recognition/models/finetuned/

# Quantize Option B
quantize-facenet-b:
	@echo "Quantizing FaceNet Option B (Progressive Unfreezing)..."
	$(PYTHON) src/scripts/quantize_model.py \
		--model src/bp_face_recognition/models/finetuned/facenet_progressive_v1.0.keras \
		--type float16 \
		--output src/bp_face_recognition/models/finetuned/

# Quantize Option C
quantize-facenet-c:
	@echo "Quantizing FaceNet Option C (Triplet Loss)..."
	$(PYTHON) src/scripts/quantize_model.py \
		--model src/bp_face_recognition/models/finetuned/facenet_triplet_best.keras \
		--type float16 \
		--output src/bp_face_recognition/models/finetuned/

# Quantize all 3 models (convenience command)
quantize-facenet-all:
	@echo "Quantizing all FaceNet models..."
	$(MAKE) quantize-facenet-a
	$(MAKE) quantize-facenet-b
	$(MAKE) quantize-facenet-c
	@echo "All models quantized!"
```

### Evaluation Commands (Individual)

```makefile
# ============================================================
# FaceNet Evaluation Commands (Individual)
# ============================================================

# Evaluate Option A (regular)
evaluate-facenet-a:
	@echo "Evaluating FaceNet Option A (Transfer Learning)..."
	$(PYTHON) src/bp_face_recognition/vision/training/finetune/evaluate_finetuned_model.py \
		--model src/bp_face_recognition/models/finetuned/facenet_transfer_v1.0.keras \
		--output results/evaluation/facenet_transfer_results.json

# Evaluate Option A (quantized)
evaluate-facenet-a-quantized:
	@echo "Evaluating FaceNet Option A (Quantized)..."
	$(PYTHON) src/bp_face_recognition/vision/training/finetune/evaluate_finetuned_model.py \
		--model src/bp_face_recognition/models/finetuned/facenet_transfer_v1.0_float16.tflite \
		--output results/evaluation/facenet_transfer_quantized_results.json

# Evaluate Option B (regular)
evaluate-facenet-b:
	@echo "Evaluating FaceNet Option B (Progressive Unfreezing)..."
	$(PYTHON) src/bp_face_recognition/vision/training/finetune/evaluate_finetuned_model.py \
		--model src/bp_face_recognition/models/finetuned/facenet_progressive_v1.0.keras \
		--output results/evaluation/facenet_progressive_results.json

# Evaluate Option B (quantized)
evaluate-facenet-b-quantized:
	@echo "Evaluating FaceNet Option B (Quantized)..."
	$(PYTHON) src/bp_face_recognition/vision/training/finetune/evaluate_finetuned_model.py \
		--model src/bp_face_recognition/models/finetuned/facenet_progressive_v1.0_float16.tflite \
		--output results/evaluation/facenet_progressive_quantized_results.json

# Evaluate Option C (regular)
evaluate-facenet-c:
	@echo "Evaluating FaceNet Option C (Triplet Loss)..."
	$(PYTHON) src/bp_face_recognition/vision/training/finetune/evaluate_triplet_model.py \
		--model src/bp_face_recognition/models/finetuned/facenet_triplet_best.keras \
		--output results/evaluation/facenet_triplet_results.json

# Evaluate Option C (quantized)
evaluate-facenet-c-quantized:
	@echo "Evaluating FaceNet Option C (Quantized)..."
	$(PYTHON) src/bp_face_recognition/vision/training/finetune/evaluate_finetuned_model.py \
		--model src/bp_face_recognition/models/finetuned/facenet_triplet_best_float16.tflite \
		--output results/evaluation/facenet_triplet_quantized_results.json

# Evaluate all 6 variants and generate comparison report
evaluate-facenet-all:
	@echo "Evaluating all FaceNet variants (this will take time)..."
	$(MAKE) evaluate-facenet-a
	$(MAKE) evaluate-facenet-a-quantized
	$(MAKE) evaluate-facenet-b
	$(MAKE) evaluate-facenet-b-quantized
	$(MAKE) evaluate-facenet-c
	$(MAKE) evaluate-facenet-c-quantized
	@echo "Generating final comparison report..."
	$(PYTHON) src/bp_face_recognition/vision/training/finetune/compare_all_models.py
```

### Registry Update Commands (Individual)

```makefile
# ============================================================
# Model Registry Update Commands (Individual)
# ============================================================

# Update registry with Option A results
update-registry-a:
	@echo "Updating registry for Option A..."
	$(PYTHON) scripts/update_models_registry.py \
		--model-type facenet_transfer \
		--results results/evaluation/facenet_transfer_results.json

# Update registry with Option B results
update-registry-b:
	@echo "Updating registry for Option B..."
	$(PYTHON) scripts/update_models_registry.py \
		--model-type facenet_progressive \
		--results results/evaluation/facenet_progressive_results.json

# Update registry with Option C results
update-registry-c:
	@echo "Updating registry for Option C..."
	$(PYTHON) scripts/update_models_registry.py \
		--model-type facenet_triplet \
		--results results/evaluation/facenet_triplet_results.json

# Update all registries
update-registry-all:
	@echo "Updating all model registries..."
	$(MAKE) update-registry-a
	$(MAKE) update-registry-b
	$(MAKE) update-registry-c
```

### Testing Commands (For Camera Integration)

```makefile
# ============================================================
# FaceNet Testing Commands (With Camera)
# ============================================================

# Test Option A with camera
test-facenet-a:
	@echo "Testing FaceNet Option A with camera..."
	$(PYTHON) src/bp_face_recognition/main.py --recognizer facenet_transfer

# Test Option B with camera (RECOMMENDED)
test-facenet-b:
	@echo "Testing FaceNet Option B with camera (BEST MODEL)..."
	$(PYTHON) src/bp_face_recognition/main.py --recognizer facenet_progressive

# Test Option C with camera
test-facenet-c:
	@echo "Testing FaceNet Option C with camera..."
	$(PYTHON) src/bp_face_recognition/main.py --recognizer facenet_triplet

# Test quantized versions
test-facenet-a-quantized:
	$(PYTHON) src/bp_face_recognition/main.py --recognizer facenet_transfer_quantized

test-facenet-b-quantized:
	$(PYTHON) src/bp_face_recognition/main.py --recognizer facenet_progressive_quantized

test-facenet-c-quantized:
	$(PYTHON) src/bp_face_recognition/main.py --recognizer facenet_triplet_quantized
```

---

## Phase 3: Evaluation Approaches (Detailed Analysis)

### Current Evaluation Scripts Analysis

#### Option 1: Use `evaluate_recognition.py`
**What it does:**
- Builds gallery from training embeddings
- Tests on test set using nearest neighbor
- Calculates Top-1, Top-3, Top-5 accuracy
- Generates confusion matrix
- Creates classification report

**Pros:**
- ✅ Comprehensive (Top-K accuracy)
- ✅ Already exists
- ✅ Uses gallery approach (realistic)

**Cons:**
- ❌ Requires recognizer to be loaded via FaceTracker
- ❌ Complex integration for new models
- ❌ Assumes specific model structure

**Best for:** Final validation, production testing

#### Option 2: Use `evaluate_triplet_model.py` (Created)
**What it does:**
- Direct model loading
- KNN classifier on embeddings
- Classification report per class
- JSON output with metrics

**Pros:**
- ✅ Simple and direct
- ✅ Works with any Keras model
- ✅ JSON output for automation
- ✅ Easy to modify

**Cons:**
- ❌ Basic (only Top-1 accuracy)
- ❌ No Top-K metrics
- ❌ Requires embedding extraction

**Best for:** Quick evaluation, CI/CD, automated pipelines

#### Option 3: Create New Unified Evaluator
**What it would do:**
- Support both classifier and embedding models
- Comprehensive metrics (accuracy, precision, recall, F1, Top-K)
- Confusion matrix visualization
- Per-class analysis
- Export to JSON and CSV
- Compare multiple models

**Pros:**
- ✅ Most comprehensive
- ✅ Designed for FaceNet models
- ✅ Can compare quantized vs regular
- ✅ Publication-ready outputs

**Cons:**
- ❌ Needs to be created
- ❌ More complex
- ❌ Takes time to implement

**Best for:** Academic paper, thesis, final report

### Recommendation

**Use Option 2** (`evaluate_triplet_model.py` approach) with enhancements:

1. **Quick evaluation** for pipeline automation
2. **Create unified evaluator** (Option 3) for final thesis comparison
3. **Use existing `evaluate_recognition.py`** for production camera testing

### Proposed Evaluation Script: `evaluate_finetuned_model.py`

```python
"""
Unified evaluator for FaceNet fine-tuned models.
Supports both Keras and TFLite models.
"""

def evaluate_model(model_path, test_ds, model_type='keras'):
    """
    Evaluate a fine-tuned FaceNet model.
    
    Args:
        model_path: Path to model file (.keras or .tflite)
        test_ds: Test dataset
        model_type: 'keras' or 'tflite'
    
    Returns:
        dict: Evaluation metrics
    """
    # Load model (Keras or TFLite)
    # Extract embeddings or get predictions
    # Calculate metrics:
    #   - Overall accuracy
    #   - Per-class accuracy
    #   - Precision, Recall, F1
    #   - Confusion matrix
    #   - Inference time
    # Save to JSON
    pass
```

**Metrics to Collect:**
1. **Classification Metrics:**
   - Overall accuracy
   - Per-class accuracy
   - Macro/Weighted precision, recall, F1
   - Confusion matrix

2. **Performance Metrics:**
   - Inference time (ms per image)
   - Throughput (FPS)
   - Model loading time

3. **Model Info:**
   - Model size (MB)
   - Number of parameters
   - Quantization type (if applicable)
   - Compression ratio

---

## Phase 4: Auto-Updater Script (scripts/update_models_registry.py)

### Purpose
Automatically update `config/models.yaml` with evaluation results.

### Functionality

```python
#!/usr/bin/env python3
"""
Update models.yaml registry with evaluation results.
"""

import yaml
import json
import argparse
from pathlib import Path

def update_model_registry(model_type, results_file, config_path='config/models.yaml'):
    """
    Update models.yaml with evaluation results.
    
    Args:
        model_type: e.g., 'facenet_transfer', 'facenet_progressive'
        results_file: Path to evaluation JSON
        config_path: Path to models.yaml
    """
    # Load results
    with open(results_file) as f:
        results = json.load(f)
    
    # Load models.yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Update recognizer entry
    if model_type in config['recognizers']:
        config['recognizers'][model_type]['metadata']['accuracy'] = results['test_accuracy']
        config['recognizers'][model_type]['metadata']['evaluated'] = True
        config['recognizers'][model_type]['metadata']['evaluated_at'] = results['timestamp']
        
        # Add per-class metrics if available
        if 'classification_report' in results:
            config['recognizers'][model_type]['metadata']['per_class_metrics'] = results['classification_report']
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Updated {model_type} in registry")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--config', default='config/models.yaml')
    args = parser.parse_args()
    
    update_model_registry(args.model_type, args.results, args.config)
```

### Features
1. ✅ Updates accuracy in metadata
2. ✅ Adds evaluation timestamp
3. ✅ Stores per-class metrics
4. ✅ Validates model exists before updating
5. ✅ Creates backup of original config

---

## README.md Updates Required

### Section: FaceNet Fine-Tuning Guide

```markdown
## FaceNet Fine-Tuning Guide

This project includes three FaceNet fine-tuning approaches with automated training, 
quantization, and evaluation pipeline.

### Available Models

| Model | Approach | Accuracy | Training Time | Size | Status |
|-------|----------|----------|---------------|------|--------|
| Option A | Transfer Learning | 92.84% | 4 min | 93 MB | ✅ Ready |
| Option B | Progressive Unfreezing | 99.15% | 50 min | 272 MB | ⭐ **Best** |
| Option C | Triplet Loss | 94.63% | 90 min | 271 MB | ✅ Ready |

### Quick Start

#### 1. Train a Model

```bash
# Option A: Fast (4 min, 92.84%)
make train-facenet-a

# Option B: Best accuracy (50 min, 99.15%) ⭐ RECOMMENDED
make train-facenet-b

# Option C: Metric learning (90 min, 94.63%)
make train-facenet-c

# WSL GPU versions (3-4x faster)
make train-facenet-b-wsl
```

#### 2. Quantize Model (Optional)

Reduces model size by 75% (272 MB → 68 MB) with minimal accuracy loss (~0.5%).

```bash
# Quantize specific model
make quantize-facenet-b

# Quantize all models
make quantize-facenet-all
```

#### 3. Evaluate Model

```bash
# Evaluate Option B
make evaluate-facenet-b

# Evaluate quantized version
make evaluate-facenet-b-quantized

# Evaluate all variants
make evaluate-facenet-all
```

#### 4. Update Model Registry

```bash
# Update registry with evaluation results
make update-registry-b

# Update all registries
make update-registry-all
```

#### 5. Test with Camera

```bash
# Test Option B (recommended)
make test-facenet-b

# Test quantized version
make test-facenet-b-quantized
```

### Complete Workflow Example

Train, quantize, evaluate, and test the best model:

```bash
# Step 1: Train
make train-facenet-b

# Step 2: Quantize (optional but recommended)
make quantize-facenet-b

# Step 3: Evaluate both versions
make evaluate-facenet-b
make evaluate-facenet-b-quantized

# Step 4: Update registry
make update-registry-b

# Step 5: Test with camera
make test-facenet-b
```

### Model Comparison

After evaluating all variants, generate comparison report:

```bash
make evaluate-facenet-all
```

This creates `results/evaluation/facenet_comparison_report.md` with:
- Accuracy comparison table
- Speed benchmarks
- Size comparison
- Recommendation

### Production Deployment

Best model for production: **Option B (Progressive Unfreezing)**

**Regular version:**
- Accuracy: 99.15%
- Size: 272 MB
- Speed: ~50ms (CPU), ~10ms (GPU)

**Quantized version:**
- Accuracy: ~98.5% (estimated)
- Size: 68 MB (75% reduction)
- Speed: ~30ms (CPU), faster on GPU

```bash
# Use in your application
make run recog=facenet_progressive
```
```

---

## Implementation Checklist

### Phase 1: Models.yaml Updates
- [ ] Add 6 recognizer entries (A, B, C + quantized)
- [ ] Add 6 environment profiles for testing
- [ ] Create `FinetunedRecognizer` wrapper class
- [ ] Verify paths and file names

### Phase 2: Makefile Commands
- [ ] Training commands (6 variants: A/B/C × CPU/WSL)
- [ ] Quantization commands (4 variants: A/B/C/all)
- [ ] Evaluation commands (7 variants: A/B/C/all × regular/quantized)
- [ ] Registry update commands (4 variants: A/B/C/all)
- [ ] Testing commands (7 variants)

### Phase 3: Evaluation Scripts
- [ ] Enhance `evaluate_triplet_model.py` → `evaluate_finetuned_model.py`
- [ ] Support both Keras and TFLite models
- [ ] Collect comprehensive metrics
- [ ] Export to JSON format
- [ ] Create comparison report generator

### Phase 4: Auto-Updater
- [ ] Create `scripts/update_models_registry.py`
- [ ] Parse evaluation JSON results
- [ ] Update models.yaml metadata
- [ ] Add validation and error handling
- [ ] Create backup mechanism

### Documentation
- [ ] Update README.md with FaceNet section
- [ ] Add command reference table
- [ ] Include workflow examples
- [ ] Document troubleshooting

---

## Questions for Confirmation

1. **FinetunedRecognizer Class**: Should I create a wrapper class to load Keras models via the registry, or use existing `TFLiteRecognizer` for both? (TFLiteRecognizer might work if we convert first)

2. **Evaluation Priority**: Should I enhance the existing evaluation script, or is the current `evaluate_triplet_model.py` sufficient for your needs?

3. **Comparison Report**: Do you want a Markdown report comparing all 6 variants, or is the JSON output sufficient?

4. **Cross-Platform**: The WSL commands assume Ubuntu-22.04. Should I add configurable distro support?

5. **Parallel Execution**: Should I add a flag to evaluate multiple models in parallel (faster but complex), or keep sequential (simpler)?

6. **README Location**: Should the FaceNet guide be a separate section in README.md, or a separate document linked from README?

---

**Ready to proceed with implementation?** Please confirm or suggest changes to this plan.
