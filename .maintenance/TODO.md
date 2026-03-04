# TODO - Next Session

## 🎯 **SESSION 7: Real-Time Pipeline & Camera Integration**

### Image Source Configuration 🔥 **HIGH PRIORITY** (BLOCKS Real-Time Detection)

- **Goal**: Configure image source via environment variable
- **Tasks**:
  - [x] Research Android phone USB tethering camera access
  - [x] Support RTSP stream URLs (e.g., `rtsp://147.232.24.197/live.sdp`)
  - [x] Support USB-connected Android device (Pixel 10 via ADB/USB camera)
  - [x] Add environment variable configuration (`CAMERA_SOURCE`)
  - [x] Handle source switching gracefully
- **Deliverable**: Configurable camera source system ✅

### Real-Time Detection

#### **1. Camera Stream Connection** 
- **Goal**: Connect to camera and process frames in real-time
- **Depends on**: Image Source Configuration
- **Tasks**:
  - [x] Test camera stream handling
  - [x] Implement frame capture loop
  - [x] Add frame preprocessing
- **Deliverable**: Working camera capture ✅

#### **2. Face Detection Pipeline** 
- **Goal**: Integrate MediaPipe face detection
- **Tasks**:
  - [x] Test face detection (MediaPipe)
  - [x] Test detection + recognition pipeline
  - [x] Handle no-face case gracefully
- **Deliverable**: Detection pipeline working ✅

#### **3. Performance Optimization** 
- **Goal**: Achieve 30+ FPS target
- **Tasks**:
  - [x] Measure FPS and identify bottlenecks
  - [x] Implement frame skipping (current: skip 3)
  - [x] Implement resolution scaling (current: 0.5x)
  - [ ] Benchmark original vs quantized models
  - [ ] Compare CPU vs GPU inference
- **Deliverable**: Performance benchmarks

### **SESSION 8: Multi-Paradigm Recognition (Classifier vs Metric)** 🎯

#### **1. Structural Separation**
- [ ] Move existing classifier training to `training/classifier/`
- [ ] Create skeleton for metric learning in `training/metric/`
- [ ] Update `models.yaml` with paradigm metadata (Closed-Set vs Open-Set)

#### **2. Metric Learning Implementation (Open-Set)**
- [ ] Implement Custom Triplet Loss for Keras 3
- [ ] Add L2-Normalization layer to EfficientNet backbone
- [ ] Update data loader for balanced triplet batching
- [ ] Train 128D/512D feature extractor on `seccam_2` (15 identities)
- [ ] Verify embedding variance on unseen faces

#### **3. Classifier Refinement (Closed-Set)**
- [ ] Implement stepped fine-tuning (backbone unfreezing)
- [ ] Add Softmax-based recognizer for direct classification
- [ ] Add entropy-based "Unknown" detection for classifiers

#### **4. Comparative Analysis**
- [ ] Benchmark Accuracy vs Generalization (unseen people)
- [ ] Benchmark Latency (Softmax vs Euclidean Vector Search)
- [ ] Quantize both models and verify CPU inference speed

---

## 📋 **BACKLOG - Future Sessions**

### Database Implementation
- [ ] Implement FaceDatabase class
- [ ] Add database connection handling
- [ ] Add CRUD operations for face embeddings
- [ ] Add database migration system
- [ ] Add database tests (with mocked DB)

### Documentation
- [ ] Add docstrings to all public APIs
- [ ] Generate API documentation
- [ ] Add usage examples
- [ ] Create architecture diagram

### Testing (Ongoing)
- [ ] Verify accuracy after quantization
- [ ] Add more edge case tests
- [ ] Increase test coverage

### TensorFlow Lite Migration
- [ ] Migrate from deprecated `tf.lite.Interpreter` to `ai_edge_litert` package
- [ ] Reference: https://ai.google.dev/edge/litert/migration
- [ ] Warning: "tf.lite.Interpreter is deprecated and is scheduled for deletion in TF 2.20"

---

## ✅ COMPLETED (Session 7)

### Camera Integration & Pipeline Fixes
- [x] Camera stream works with USB phone (webcam device 0)
- [x] Fixed MediaPipe detector attribute errors in both `detect()` and `detect_with_confidence()`
- [x] `make run` - main command for face recognition app (with enhanced logging)
- [x] `make register name="YourName"` - new command to register yourself from camera
- [x] Boxes now show "Unknown" for unregistered faces
- [x] Added `src/scripts/register_from_camera.py` for easy user onboarding
- [x] Fixed `main.py` to use BGR frames for detector while keeping RGB for display

### Image Source Configuration
- [x] Add `CAMERA_SOURCE`, `CAMERA_DEVICE`, `CAMERA_RTSP_URL`, `CAMERA_WIDTH`, `CAMERA_HEIGHT`, `CAMERA_FPS` to settings.py
- [x] Create `utils/camera_source.py` with:
  - `CameraConfig` dataclass
  - `CameraSource` abstract base class
  - `WebcamSource` - local webcam support
  - `RTSPSource` - RTSP stream URLs with auto-reconnect
  - `USBDeviceSource` - USB-connected Android via ADB
  - `CameraManager` - unified interface with source switching
- [x] Add unit tests in `test_camera_source.py` (15 tests, all passing)
- [x] Add nox session `test_camera` in noxfile.py
- [x] All lint checks pass for new files
- [x] No type errors in new files

**Usage**:
```bash
export CAMERA_SOURCE=webcam  # or rtsp, usb
export CAMERA_DEVICE=0
export CAMERA_RTSP_URL=rtsp://147.232.24.197/live.sdp
export CAMERA_WIDTH=1280
export CAMERA_HEIGHT=720
export CAMERA_FPS=30
```

### Test Infrastructure
- [x] Create pytest test suite in `src/bp_face_recognition/tests/`
- [x] Add unit tests: test_config.py, test_training.py, test_preprocessing.py
- [x] Add integration tests: test_full_pipeline.py
- [x] Add conftest.py with shared fixtures
- [x] Update noxfile.py with test sessions
- [x] Update Makefile with test commands
- [x] Add setuptools to pyproject.toml

**Note**: Tests that import from `vision` module are skipped in CI due to face_recognition package issue with pkg_resources in uv test environment. These tests work when run with proper pip setup.
