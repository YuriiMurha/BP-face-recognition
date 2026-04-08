# Progress Log

## 21-Mar-26 (Session 21) — Thesis Writing & Final Polish

### Updated Existing Documents
- Filled all TBD Option C (Triplet Loss) results in `facenet_finetuning.md` (7 locations)
- Updated 6 tables in `facenet_results_tables.md` with Option C metrics (94.63%, F1=0.9455)
- Completed RQ3 answer: triplet loss partially confirmed (+1.79% over frozen TL)

### New Chapters Written
1. **Detection Methods** (`docs/thesis/chapters/detection_methods.md`, ~2,500 words) — MediaPipe, MTCNN, Haar, Dlib HOG comparison with speed-recall trade-off analysis
2. **System Architecture** (`docs/thesis/chapters/system_architecture.md`, ~2,500 words) — Layered design, plugin system, service layer, design decisions table
3. **Open-Set vs Closed-Set** (`docs/thesis/chapters/open_vs_closed_set.md`, ~1,800 words) — Paradigm definitions, architecture comparison, trade-offs, use cases
4. **Results Analysis** (`docs/thesis/chapters/results_analysis.md`, ~3,000 words) — Per-class accuracy (14 classes), confusion matrix interpretation, class imbalance impact, training curves

### Final Polish
- Created thesis chapter index (`docs/thesis/README.md`)
- All chapters include LaTeX table versions
- Updated TODO.md and PROGRESS.md

### Key Findings Documented
- Stranger_6 at 66.7% accuracy: only 42 training images (0.8% of dataset)
- Progressive unfreezing helps small classes most: Stranger_9 improved +44.4%
- TL confusion matrix shows majority-class bias; PU matrix is near-diagonal
- Classes with <60 images are the primary source of remaining errors
- Per-class accuracy unreliable for classes with 9 test samples (CI width +/-20%)

---

## 18-Mar-26 (Sessions 19-20) — Codebase Cleanup & Thesis Benchmark

### Session 19: Codebase Cleanup

**Deleted 12 dead files:**
- 2 unused recognizers: `softmax_recognizer.py`, `metric_recognizer.py`
- 5 unused scripts: `register_person.py`, `export_mtcnn.py`, `validate_training_setup.py`, `cleanup_repo.sh`, `deploy_option_b.sh`
- 4 unused evaluation files: `testing.py`, `detection.py`, `fake_plotting_evaluation.py`, `fake_plotting_deeplearning.py`
- 1 stale test: `test_architecture.py`

**Fixed 3 bugs:**
- `database_service.py` — removed 63 lines of unreachable dead code (duplicate copy-paste after `return`)
- `evaluate_simple.py` — fixed broken import (`load_finetuned_facenet` → `load_finetuned_facenet_robust`)
- `models.yaml` — standardized all model_file paths to `bp_face_recognition/` prefix

**Cleaned 20 files:** Removed unnecessary `sys.path.insert` hacks

### Session 20: Thesis Benchmark

**Created `evaluation/thesis_benchmark.py`** — unified benchmark for all models.

**Bugs found & fixed during benchmarking:**
1. **Double normalization** — `create_combined_dataset()` already normalizes to [-1,1], but benchmark applied normalization again, destroying all data. All models showed 25.42% (majority class). Fix: removed redundant normalization.
2. **Detection on face crops** — was loading 26x28 pixel face crops instead of full 1280x800 surveillance frames. Fix: changed to `raw/*/test/images/` directories.
3. **OOM on large images** — Haar/MTCNN crashed on 1280x800 frames. Fix: resize to ≤800px before detection.
4. **TLoss crash** — Triplet loss model has no classification head, can't be evaluated via argmax. Fix: report training-report accuracy (94.63%) with note.
5. **Empty confusion matrix crash** — seaborn crashed on empty matrix for TLoss/EfficientNetB0. Fix: skip empty matrices.

**Final benchmark results:**

| Detection Method | Avg Time | FPS | Faces Detected (19 images) |
|-----------------|----------|-----|---------------------------|
| MediaPipe | 3.1 ms | 325.6 | 6 |
| MTCNN | 240.3 ms | 4.2 | 25 |
| Haar Cascade | 31.4 ms | 31.8 | 12 |
| Dlib HOG | 155.7 ms | 6.4 | 10 |

| Recognition Model | Accuracy | F1 | Precision | Recall | Size (MB) |
|-------------------|----------|------|-----------|--------|-----------|
| FaceNet TL | 92.84% | 0.9276 | 0.9316 | 0.9284 | 92.7 |
| **FaceNet PU** | **99.15%** | **0.9912** | **0.9918** | **0.9915** | 271.9 |
| FaceNet TLoss* | 94.63% | 0.9455 | 0.9460 | 0.9463 | 270.4 |
| EfficientNetB0** | 100% | 1.0 | 1.0 | 1.0 | 23.8 |
| EfficientNetB0 float16** | 100% | 1.0 | 1.0 | 1.0 | 9.0 (62% reduction) |

\* Training-report metrics (embedding model)
\** Different dataset (seccam_2, 15 classes)

**Generated outputs in `results/thesis/`:**
- `thesis_benchmark_report.md` — Markdown + LaTeX comparison tables
- `facenet_tl_confusion_matrix.png`, `facenet_pu_confusion_matrix.png`
- `training_curves_comparison.png`
- JSON results for programmatic access

---

## 17-Mar-26 (Session 18) — Closed-Set Face Recognition System

### Key Insight
The fine-tuned FaceNet models are 14-class softmax classifiers. `FinetunedRecognizer.recognize()` uses argmax to return `(identity, confidence)` directly. The closed-set system bypasses the database entirely.

### Files Created
1. `services/closed_set_pipeline_service.py` — detect → classify directly (no database)
2. `closed_set_main.py` — entry point with camera loop
3. `models/finetuned/dataset_info.json` — 14 class names metadata
4. `tests/unit/test_closed_set_pipeline.py` — 8 unit tests (all passing)

### Camera Test Results
- Recognized user correctly even in bad lighting
- Older photo recognized at ~60% confidence
- Unknown faces flicker between classes (expected — no "none of the above" in softmax)

---

## 12-Mar-26 (Sessions 12-14) — FaceNet Fine-Tuning Study

- **Session 12**: Pre-trained FaceNet integrated (99.6% LFW accuracy)
- **Session 13**: Three fine-tuning strategies implemented
- **Session 14**: Final analysis — TL 92.84%, PU 99.15%, TLoss 94.63%
- Progressive Unfreezing (PU) exceeded 97% target by 2.15%
- Complete thesis documentation structure created

---

**Last Updated**: Session 21 — March 21, 2026
