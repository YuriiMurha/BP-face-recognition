# TODO - Next Session

## 🎯 **SESSION 10: Fix Keras Serialization & Complete Metric Model Training**

### Current Session Progress ✅ (COMPLETED)
- [x] Identified root cause: Lambda layer serialization incompatibility between WSL/Windows
- [x] Created custom `L2NormalizeLayer` with `@tf.keras.utils.register_keras_serializable()`
- [x] Updated `model.py` to use custom layer instead of Lambda
- [x] Updated `finetune_trainer.py` to use custom layer
- [x] Updated `keras_metric_recognizer.py` to pass custom_objects when loading
- [x] Verified custom layer works for save/load on Windows
- [x] Fixed duplicate model.fit() call in finetune_trainer.py

### Next Steps for Session 10
- [ ] Run full training on Windows (10 epochs Phase 1, 5 epochs Phase 2)
- [ ] Verify trained model achieves >80% validation accuracy
- [ ] Test model loading in KerasMetricRecognizer
- [ ] Clear database and re-register with new model
- [ ] Verify face recognition works with custom metric model
- [ ] If successful: set metric_efficientnetb0_128d as default in models.yaml

### Alternative Approaches (if current fails)
- [ ] Train directly on Windows without WSL
- [ ] Use pre-trained FaceNet model
- [ ] Implement ArcFace loss instead of classification pre-training

---

## 🎯 **SESSION 9: Custom Metric Model Training & Testing**

### Preprocessing Pipeline ✅ (COMPLETED)
- [x] Rename `data/` to `preprocessing/`
- [x] Create universal dataset structure (flat, label prefix)
- [x] Create `crop_faces.py` - crop from raw datasets
- [x] Create `split_lfw.py` - split LFW into train/val/test
- [x] Create `augmentation.py` - augment cropped faces
- [x] Support dynamic dataset discovery
- [x] Make commands: `prepare-crop-all`, `prepare-augment-all`, `prepare-all`

### Dataset Restructuring ✅ (COMPLETED)
- [x] Move webcam, seccam, seccam_2 to `raw/`
- [x] Move triplet_gallery to `raw/lfw`
- [x] Delete `research/` folder
- [x] Fix labels: "1" → "Yurii", "2" → "Stranger_2"
- [x] Universal filename format:
  - Custom: `{label}_{uuid}.jpg` → `Yurii_f3d9f09a.jpg`
  - LFW: `{identity}_{index}.jpg` → `George_W_Bush_0000.jpg`
- [x] Augmented format: `.{N}.jpg` suffix → `Yurii_f3d9f09a.0.jpg`

### Metric Model Training ✅ (COMPLETED)
- [x] Train on LFW dataset (5 epochs)
- [x] Model saved: `metric_efficientnetb0_128d_final.keras`
- [x] Update models.yaml to use metric model by default

### Testing Custom Model ✅ (COMPLETED - Session 9)
- [x] Fix register_from_camera.py to use configurable recognizer from models.yaml
- [x] Fix KerasMetricRecognizer to load trained weights
- [x] Clear old database (dlib embeddings)
- [x] Register yourself: `make register name="Yurii"` 
- [x] Run app: `make run`
- [x] Verify recognition works (using dlib_v1)

### Known Issues
- Custom metric model weights fail to load (Keras version mismatch) - needs retraining
- Fallback: Use dlib_v1 which works reliably

### Session 9 Complete ✅
- dlib_v1 recognizer works for registration and recognition
- Default recognizer changed to dlib_v1 (reliable)
- Metric model training infrastructure fixed (trainer.py saves backbone weights)

### Extended Training (OPTIONAL)
- [ ] Train with more epochs (20+)
- [ ] Include webcam + seccam_2 datasets
- [ ] Train 64D vs 128D comparison

---

## 📋 **BACKLOG**

### Warnings Cleanup
- [ ] Address dlib/face_recognition warnings (optional dependency)
- [ ] Either install or suppress warnings gracefully

### Performance Optimization
- [ ] Benchmark original vs quantized models
- [ ] Compare CPU vs GPU inference

### Database Implementation
- [ ] Implement FaceDatabase class
- [ ] Add database connection handling
- [ ] Add CRUD operations for face embeddings

### Documentation
- [ ] Add docstrings to all public APIs
- [ ] Generate API documentation
- [ ] Create architecture diagram

### Testing
- [ ] Verify accuracy after quantization
- [ ] Add more edge case tests

### TensorFlow Lite Migration
- [ ] Migrate from deprecated `tf.lite.Interpreter` to `ai_edge_litert`
