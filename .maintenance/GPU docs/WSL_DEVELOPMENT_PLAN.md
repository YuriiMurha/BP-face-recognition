# WSL Development Migration Plan

## 🎯 Objective
Transition from Windows development to WSL2 GPU-accelerated development environment while maintaining code continuity and leveraging OpenCode for guidance.

## 📋 Your Requirements
- ✅ **Primary Environment**: WSL2 (GPU development, terminal-based)
- ✅ **IDE**: VS Code with OpenCode integration 
- ✅ **Timeline**: Switch in a few days
- ✅ **Learning Method**: Step-by-step OpenCode guidance
- ✅ **Development Tools**: Minimal - Python, uv, make, git
- ✅ **Git Operations**: Prefer Windows for commits, WSL for development
- ✅ **Code Access**: Single repo shared between environments

## 🗺️ Development Strategy

### **Phase 1: WSL Foundation (Days 1-2)**

#### **1.1 VS Code + WSL Setup**
```bash
# In WSL terminal
sudo apt update && sudo apt install code

# Open VS Code with WSL integration
code .

# Install OpenCode extension in VS Code
# Search for "OpenCode" in extensions
```

#### **1.2 Environment Configuration**
```bash
# Verify Python installation
python3 --version

# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# Navigate to project (using Windows mount)
cd /mnt/d/Coding/Personal/BP-face-recognition

# Install dependencies
uv sync
```

#### **1.3 Basic Linux Commands**
```bash
# Learn essential navigation
ls -la                    # List files
pwd                        # Current directory  
cd ..                       # Parent directory
cp file.py file_backup.py  # Copy files
mv old_name.py new_name.py # Rename files

# Learn text editing basics
nano file.py              # Simple editor
vim file.py               # Advanced editor
```

### **Phase 2: WSL Development Workflow (Days 3-5)**

#### **2.1 Development Environment**
```bash
# Daily startup sequence
wsl
cd /mnt/d/Coding/Personal/BP-face-recognition

# Check GPU status
uv run python src/bp_face_recognition/utils/cross_platform_gpu.py

# Start VS Code from WSL
code . --new-window
```

#### **2.2 GPU Validation**
```bash
# Run GPU diagnostics
~/gpu_verification.py

# Run performance benchmarks  
uv run python scripts/benchmark_gpu_performance.py

# Monitor GPU utilization
watch -n 1 nvidia-smi
```

#### **2.3 Development Commands**
```bash
# Core development workflow
uv run python src/main.py --help              # Test app
uv run nox -s type_check                 # Type checking
uv run nox -s lint                       # Lint code
uv run pytest src/tests/unit/              # Run tests
make benchmark                           # Run benchmarks
```

### **Phase 3: Git Workflow (Ongoing)**

#### **3.1 Development in WSL**
```bash
# In WSL - daily development
git status                    # Check changes
git add .                    # Stage changes
git commit -m "feat: WSL GPU validation"  # Commit in WSL
git push                       # Push from WSL
```

#### **3.2 Windows for Git Operations (Optional)**
```bash
# In Windows terminal - when preferred
cd D:\Coding\Personal\BP-face-recognition
git status
git commit --amend            # Fix commit message
git tag -a v1.0              # Create tags
git log --graph --oneline     # Review history
```

## 🎓 Learning Progression

### **Week 1: WSL Basics**
- [ ] **Day 1**: Set up VS Code + WSL, basic Linux commands
- [ ] **Day 2**: Environment setup, project access, uv commands
- [ ] **Day 3**: Simple Python development, basic debugging
- [ ] **Day 4**: Git operations in WSL, first WSL commits
- [ ] **Day 5**: Code navigation and file management

### **Week 2: Development Mastery**  
- [ ] **Day 6**: VS Code Server configuration, remote development
- [ ] **Day 7**: Advanced Git operations, branching/merging
- [ ] **Day 8**: Debugging tools and techniques
- [ ] **Day 9**: System monitoring and performance tools
- [ ] **Day 10**: Scripting and automation basics

### **Week 3: GPU Development**
- [ ] **Day 11**: GPU validation and verification
- [ ] **Day 12**: MediaPipe GPU testing and benchmarking
- [ ] **Day 13**: Performance optimization and monitoring
- [ ] **Day 14**: GPU-specific debugging techniques
- [ ] **Day 15**: Complete GPU vs CPU performance comparison

## 🔧 Quick Reference Commands

### **Essential WSL Commands**
```bash
# Navigation
pwd                    # Where am I?
ls -la                  # What's here?
cd path                  # Go to directory

# File Operations  
cp source dest            # Copy files
mv old new               # Move/rename
rm file                  # Delete files
mkdir folder             # Create directory

# Text Editing
nano file.py             # Simple editor
vim file.py              # Advanced editor
code file.py             # Open in VS Code

# Development
uv run python script.py    # Run Python
uv sync                  # Install dependencies
pytest tests/            # Run tests
nox -s lint              # Lint code
```

### **GPU Monitoring Commands**
```bash
nvidia-smi                     # GPU status
watch -n 1 nvidia-smi          # Real-time monitoring
htop                           # System resources
nvcc --version                 # CUDA version
```

### **Development Workflow Commands**
```bash
make clean                # Clean build
make test                 # Run tests  
make benchmark             # Run benchmarks
git status               # Check changes
git diff                 # View changes
git log --oneline       # Recent commits
```

## 🆘 **Troubleshooting Guide**

### **Common WSL Issues**
```bash
# "command not found" errors
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc

# Permission issues with Windows files
sudo chmod -R 755 /mnt/d/Coding/Personal/BP-face-recognition

# VS Code WSL extension not working
code --install-extension ms-vscode-remote.remote-wsl

# GPU not detected
~/gpu_verification.py
nvidia-smi
```

### **Performance Issues**
```bash
# MediaPipe not using GPU
export CUDA_VISIBLE_DEVICES=0
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Slow GPU performance
nvidia-smi -q -d QUERY
# Check for thermal throttling
```

## 📅 Daily Development Checklist

### **Morning Setup (10 minutes)**
```bash
□ Start WSL
□ Navigate to project directory  
□ Check GPU status (nvidia-smi)
□ Open VS Code with project
□ Quick git status check
```

### **Development Session**
```bash
□ Make changes
□ Run tests locally
□ Use OpenCode for guidance
□ Monitor GPU usage if relevant
□ Commit changes with meaningful messages
```

### **Evening Review (5 minutes)**
```bash
□ Review changes with git diff
□ Push commits to remote
□ Note blockers for tomorrow
□ Update TODO.md if needed
□ Backup important changes
```

## 🎯 Success Metrics

### **Week 1 Success Criteria**
- [ ] Can navigate Linux filesystem confidently
- [ ] Comfortable with basic text editing in terminal
- [ ] Successfully runs Python code in WSL
- [ ] Can perform basic Git operations in WSL
- [ ] VS Code + WSL workflow established

### **Week 2 Success Criteria**  
- [ ] Mastered WSL-specific development tools
- [ ] Can debug issues in WSL environment
- [ ] Efficient workflow with keyboard shortcuts
- [ ] Understanding of WSL-specific file system behavior
- [ ] Can handle environment configuration independently

### **Week 3 Success Criteria**
- [ ] GPU acceleration verified and working
- [ ] Can run MediaPipe with GPU delegate
- [ ] Performance benchmarks completed
- [ ] Can monitor and optimize GPU usage
- [ ] Ready for production deployment from WSL

## 🔄 **Next Steps After Migration**

### **Immediate (Week 4)**
- [ ] **WSL GPU Performance Testing** - Run comprehensive benchmarks
- [ ] **Production Configuration** - Set up WSL for deployment
- [ ] **Documentation Updates** - Update guides with WSL-specific notes
- [ ] **Tool Mastery** - Advanced WSL development techniques

### **Medium Term (Month 2-3)**
- [ ] **Performance Optimization** - Fine-tune GPU usage
- [ ] **Advanced Debugging** - GPU-specific issue resolution
- [ ] **Automation** - Scripting common development tasks
- [ ] **Team Integration** - WSL workflows for collaboration

## 📚 **Learning Resources**

### **WSL Documentation**
- [Windows Subsystem for Linux Documentation](https://learn.microsoft.com/en-us/windows/wsl/)
- [WSL Command Reference](https://learn.microsoft.com/en-us/windows/wsl/wsl_reference)

### **Linux Commands**
- [Linux Command Line Basics](https://www.ryanstutorials.net/linuxtutorial/)
- [Vim Tutorial](https://vim.adventures.com/)
- [Bash Scripting Guide](https://www.gnu.org/software/bash/manual/)

### **GPU Development**
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [MediaPipe GPU Documentation](https://google.github.io/mediapipe/solutions/gpu.html)
- [TensorFlow GPU Setup](https://www.tensorflow.org/install/gpu)

---

**Goal**: By end of Week 3, you'll be proficient in WSL development with GPU acceleration, able to develop efficiently and independently in the WSL environment while having Windows available for specialized tasks when needed.