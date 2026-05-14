# GPU Training Setup Plan

## Executive Summary

**Current State:** WSL2 training infrastructure is operational with CPU-only TensorFlow 2.15.0
**Goal:** Enable GPU-accelerated training on NVIDIA GeForce GTX 1650
**Estimated Time:** 30-45 minutes
**Risk Level:** Low (fully reversible, CPU fallback always available)

---

## System Information

| Component | Details |
|-----------|---------|
| **GPU** | NVIDIA GeForce GTX 1650 (4GB VRAM) |
| **WSL Version** | 2.3.26.0 (Latest) |
| **WSL Distribution** | Ubuntu-22.04 |
| **Current TensorFlow** | 2.15.0 (CPU) |
| **Training Framework** | production_trainer.py (GPU-aware) |

**Good News:**
- WSL2 version is up-to-date with GPU support
- GTX 1650 is CUDA-capable (Compute Capability 7.5)
- All training code is already GPU-compatible
- Existing setup_wsl_gpu.sh script available

---

## Prerequisites Checklist

- [x] NVIDIA GPU detected in Windows (GTX 1650)
- [x] WSL2 Ubuntu-22.04 installed
- [x] NVIDIA drivers installed on Windows (required for GPU passthrough)
- [ ] CUDA Toolkit installed in WSL (NEEDED)
- [ ] TensorFlow with CUDA support (NEEDED)

### Pre-Setup Verification

Before starting, verify these work:

```powershell
# Check 1: Windows GPU
wmic path win32_VideoController get name
# Expected: "NVIDIA GeForce GTX 1650"

# Check 2: WSL GPU passthrough
wsl -d Ubuntu-22.04 nvidia-smi
# Expected: GPU info table (if drivers installed)

# Check 3: Current TensorFlow
wsl -d Ubuntu-22.04 bash -c "source .venv-wsl/bin/activate && python -c 'import tensorflow as tf; print(tf.__version__)'"
# Expected: "2.15.0"
```

---

## Setup Options

### Option A: Simplified APT Installation (RECOMMENDED)
**Time:** ~20-30 minutes  
**Complexity:** Low  
**Reliability:** High

Uses Ubuntu's package manager for simpler installation.

**Pros:**
- Easier and faster
- Automatic dependency management
- Well-tested on Ubuntu 22.04

**Cons:**
- May install slightly older CUDA version
- Less control over exact versions

### Option B: NVIDIA Repository Installation
**Time:** ~30-45 minutes  
**Complexity:** Medium  
**Reliability:** High

Uses official NVIDIA repositories for latest CUDA.

**Pros:**
- Latest CUDA version
- Exact version control
- Better for specific requirements

**Cons:**
- More steps
- Manual cuDNN download required

---

## Detailed Implementation Plan

### Phase 1: Pre-Installation (5 minutes)

#### Step 1.1: Backup Current Environment
```bash
# In WSL
# Create backup of current venv (optional but recommended)
cd /mnt/d/Coding/Personal/BP-face-recognition
cp -r .venv-wsl .venv-wsl-backup-cpu
```

#### Step 1.2: Verify GPU Passthrough
```bash
# In Windows PowerShell
wsl -d Ubuntu-22.04 nvidia-smi
```

**If this fails:** Install NVIDIA drivers on Windows from https://www.nvidia.com/drivers

---

### Phase 2: CUDA Installation (Option A - APT Method)

#### Step 2.1: Update System
```bash
wsl -d Ubuntu-22.04
sudo apt update && sudo apt upgrade -y
```

#### Step 2.2: Install CUDA Toolkit
```bash
sudo apt install -y nvidia-cuda-toolkit
```

**What this installs:**
- CUDA Toolkit 11.8 (Ubuntu 22.04 default)
- nvcc compiler
- CUDA libraries

**Note:** Ubuntu 22.04 provides CUDA 11.8 which works with TensorFlow 2.15

#### Step 2.3: Verify CUDA Installation
```bash
nvcc --version
# Expected: Cuda compilation tools, release 11.8

nvidia-smi
# Expected: GPU info and CUDA version
```

---

### Phase 3: TensorFlow GPU Installation (10 minutes)

#### Step 3.1: Activate Virtual Environment
```bash
cd /mnt/d/Coding/Personal/BP-face-recognition
source .venv-wsl/bin/activate
```

#### Step 3.2: Backup Current TensorFlow
```bash
pip show tensorflow
# Note the version, then uninstall
pip uninstall tensorflow -y
```

#### Step 3.3: Install TensorFlow with CUDA
```bash
pip install tensorflow[and-cuda]==2.15.0
```

**Alternative if above fails:**
```bash
pip install tensorflow==2.15.0
# Then install CUDA-specific TF package
pip install tensorflow-gpu==2.15.0
```

#### Step 3.4: Verify GPU Detection
```bash
python -c "import tensorflow as tf; print('GPUs:', len(tf.config.list_physical_devices('GPU'))); print(tf.config.list_physical_devices('GPU'))"
```

**Expected Output:**
```
GPUs: 1
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

### Phase 4: Configuration (5 minutes)

#### Step 4.1: Set Environment Variables
Add to `~/.bashrc`:
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Step 4.2: Test TensorFlow GPU Computation
```bash
python << 'EOF'
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"CUDA built: {tf.test.is_built_with_cuda()}")
print(f"GPUs: {len(tf.config.list_physical_devices('GPU'))}")

if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print(f"GPU computation test: SUCCESS")
        print(f"Result:\n{c}")
EOF
```

---

### Phase 5: Training Validation (10 minutes)

#### Step 5.1: Quick GPU Training Test
```bash
# From Windows PowerShell
cd D:\Coding\Personal\BP-face-recognition
make train-wsl backbone=MobileNetV3Small epochs=2
```

**Expected Output Indicators:**
- Training should start without errors
- You should see "Platform: GPU" in logs
- Training time should be significantly faster than CPU
- GPU memory should be utilized (check with `nvidia-smi`)

#### Step 5.2: Monitor GPU Usage
```bash
# In another WSL terminal
watch -n 1 nvidia-smi
```

You should see:
- GPU utilization > 0% during training
- Memory allocation ~1-2GB for MobileNetV3Small

---

## Alternative: Option B (NVIDIA Repository Method)

If Option A doesn't work or you need specific CUDA versions:

### Step 2.2B: Install from NVIDIA Repository
```bash
# Download and install CUDA 12.3 (latest stable)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-3

# Install cuDNN (requires manual download from NVIDIA)
# Download from: https://developer.nvidia.com/cudnn
# Then extract and copy files:
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

---

## Risk Mitigation

### If GPU Setup Fails:

1. **Restore CPU Environment:**
```bash
cd /mnt/d/Coding/Personal/BP-face-recognition
rm -rf .venv-wsl
mv .venv-wsl-backup-cpu .venv-wsl
source .venv-wsl/bin/activate
pip install tensorflow opencv-python numpy pillow scikit-learn pydantic pyyaml matplotlib pandas tqdm
```

2. **Use CPU Training:**
```bash
make train-wsl backbone=MobileNetV3Small epochs=20
# (Automatically falls back to CPU if no GPU)
```

3. **Force CPU Mode:**
```bash
make train-cpu backbone=MobileNetV3Small epochs=20
```

---

## Post-Setup Makefile Enhancement

Consider adding to Makefile:

```makefile
# GPU Setup Commands
setup-gpu:
	@echo "Setting up GPU environment in WSL2..."
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && \
		sudo apt update && \
		sudo apt install -y nvidia-cuda-toolkit && \
		source .venv-wsl/bin/activate && \
		pip uninstall tensorflow -y && \
		pip install tensorflow[and-cuda]==2.15.0"

verify-gpu:
	@echo "Verifying GPU setup..."
	wsl -d $(WSL_DISTRO) bash -c "cd $(WSL_PATH) && \
		source .venv-wsl/bin/activate && \
		python -c 'import tensorflow as tf; \
		print(\"GPUs:\", len(tf.config.list_physical_devices(\"GPU\"))); \
		print(\"CUDA:\", tf.test.is_built_with_cuda())'"
```

---

## Performance Expectations

### Training Time Comparison (20 epochs, MobileNetV3Small):

| Platform | Time | Speedup |
|----------|------|---------|
| CPU (Current) | ~30-35 min | 1x (baseline) |
| GPU (GTX 1650) | ~6-10 min | **3-5x faster** |

### GPU Utilization:
- **MobileNetV3Small**: ~1.5-2GB VRAM, 60-80% GPU utilization
- **EfficientNetB0**: ~2.5-3GB VRAM, 70-90% GPU utilization

### Batch Size Recommendations:
- **CPU Current**: 32 (optimal for CPU)
- **GPU GTX 1650**: Can increase to 64-128 for better throughput

---

## Success Criteria

✅ **Setup Complete When:**
1. `nvidia-smi` shows GPU in WSL
2. `tensorflow` detects 1 GPU device
3. Test training runs without errors
4. Training logs show "Platform: GPU"
5. `nvidia-smi` shows GPU utilization during training

---

## Next Steps After Setup

1. **Train MobileNetV3Small (20 epochs)** for full comparison
2. **Train EfficientNetB0 (20 epochs)** on GPU
3. **Compare CPU vs GPU metrics** (speed, accuracy, power)
4. **Quantize trained models** using existing pipeline
5. **Update PROGRESS.md** with GPU training results

---

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| `nvidia-smi` not found in WSL | Install NVIDIA drivers on Windows first |
| TensorFlow doesn't see GPU | Reinstall with `pip install tensorflow[and-cuda]` |
| CUDA out of memory | Reduce batch size: `--batch-size 16` |
| cuDNN errors | Install cuDNN manually from NVIDIA website |
| Slow GPU training | Check `nvidia-smi` to verify GPU is being used |
| Import errors | Ensure `.venv-wsl` is activated |

---

## Summary

**Recommended Path:**
1. Verify `nvidia-smi` works in WSL
2. Use **Option A** (APT method) for simplicity
3. Install TensorFlow with `[and-cuda]` extra
4. Test with `make train-wsl backbone=MobileNetV3Small epochs=2`
5. Proceed to full training once verified

**Fallback:** CPU training is fully functional and can be used anytime via `make train-wsl` or `make train-cpu`.
