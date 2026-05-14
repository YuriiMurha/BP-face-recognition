# Progress Log

## 13-May-26 (Session 22) — Thesis Rigor: GT Detection, Embedding Geometry, Verification, CV

### TLoss Retrain (Clean)
Fixed two bugs in `vision/training/finetune/facenet_triplet_trainer.py`:
1. `--batch-size` CLI flag was hardcoded to 32 inside `TripletDataGenerator`, causing OOM on GTX 1650. Now properly propagated end-to-end.
2. `save_model()` tried to save as `.keras`, which fails on FaceNet's Lambda layers (not JSON-serializable). Switched to weights-only `.weights.h5`.

Result: 22 epochs, best `val_loss = 0.0083` at epoch 17.

### New Evaluators (`src/bp_face_recognition/evaluation/`)
- **`detection_eval_with_groundtruth.py`** — greedy IoU matching at threshold 0.5 against manually annotated boxes from `data/datasets/raw/{seccam,seccam_2,webcam}/test/labels/` (19 frames, 26 GT boxes total). Produces P/R/F1/Mean-IoU per detector.
- **`embedding_quality.py`** — intra/inter L2 distance, silhouette (cosine), separation ratio at the 512D FaceNet backbone output.
- **`verification_eval.py`** — TAR/FAR/EER pairs protocol. **IN PROGRESS** (parallel agent).

### New Scripts
- **`scripts/run_detection_eval.py`** — invokes `detection_eval_with_groundtruth.py` in isolated CPU-only subprocesses per detector (avoids TF state collisions where MTCNN's library hangs after MediaPipe has loaded XNNPACK).
- **`scripts/plot_training_curves.py`** — produces 3 per-approach PNGs in `thesis/figures/` (replaces the old single combined plot).

### Key Numerical Findings

**Detection (GT-based, F1):** MTCNN 0.706 dominates; MediaPipe 0.250 (R=15%); Haar 0.105 (P=17%); Dlib HOG 0.056 (P=10%). The count-only methodology from Session 20 was misleading — Haar's 12 detections were 10 false positives, Dlib HOG's 10 detections were 9 false positives.

**Embedding geometry (512D backbone):**

| Metric | TL | PU | TLoss |
|---|---|---|---|
| Intra-class L2 | 0.651 | 0.575 | **0.337** |
| Inter-class L2 | 0.866 | 1.092 | **1.254** |
| Silhouette (cos) | 0.111 | **0.320** | 0.170 |
| Separation ratio | 1.330 | 1.901 | **3.724** |

Triplet Loss separation ratio = 3.72 vs PU's 1.90 — direct geometric evidence that triplet loss does what it claims, even though PU wins classification accuracy.

### Thesis Chapter 7 Updates (`thesis/chapters/07-results.md`)
- §7.1 rewrote detection section to use GT-based numbers; replaced Table 7.2 with P/R/F1/IoU.
- §7.2.5 added refs to the 3 new per-approach figures.
- §7.4.2 added new "Embedding Quality Analysis" subsection (Table 7.9b).
- §7.6 Limitations updated to remove now-resolved bullets.

### Cleanup (Delete-and-Regenerate)
- DELETED `results/evaluation/` (5 files, broken KNN-on-softmax methodology reporting PU=21.94% vs real 99.15%).
- DELETED `results/visualizations/` (preliminary 2/3-model plots from Mar 12).
- DELETED `docs/thesis/` (superseded by canonical `thesis/`).
- FLATTENED `results/thesis/` → `results/`.
- DELETED 7 broken/redundant Python scripts: `evaluate_simple.py`, `evaluate_comprehensive.py`, `generate_comparison_report.py`, two `compare_models.py` copies, `visualize_preliminary_results.py`, `scripts/evaluate_all_facenet.sh`.
- REMOVED 5 Makefile targets that invoked the broken scripts.
- ADDED 3 new Makefile targets: `make detection-eval`, `make embedding-quality`, `make training-curves`.

### Path Changes
- `results/thesis/` → `results/` (flattened)
- `docs/thesis/` → deleted (canonical: `thesis/`)

### Cross-Validation (Complete)
5-seed cross-validation across all 3 approaches (TL/PU/TLoss): infrastructure built and fully executed. Total runtime: 3 hours 22 minutes across two sessions (interrupted once by a session crash and a Docker GPU-memory contention episode; resumed cleanly via `--skip-existing`). All 15 (approach × seed) cells completed.

Final results:
- **TL: 96.52% ± 0.46%** (seeds 42/123/456/789/1024: 96.89, 97.18, 96.42, 95.95, 96.14). Canonical 92.84% was a low-tail outlier — the canonical run early-stopped at epoch 2 while seed-aware reruns trained to full 20 epochs.
- **PU: 94.11% ± 0.59%** (95.10, 93.79, 94.16, 94.16, 93.31). Canonical 99.15% was a high-tail outlier (~8σ above CV mean).
- **TLoss: 87.08% ± 10.33%** (91.43, 90.02, 96.89, **67.04**, 90.02). Catastrophic divergence on seed=789. Random triplet mining is highly seed-unstable; Hermans 2017 batch-hard mining would likely fix this.

Headline finding: **the accuracy ranking reverses under CV.** Single-split: PU > TLoss > TL. CV: TL > PU > TLoss. Documented honestly in §7.4.4 with full discussion of why each canonical number is an outlier. Also updated §7.2.4 (Table 7.5 now has both single-split and CV rows), §7.4.5 (published comparison softened), §7.5.1 (PU-wins explanation now scoped to single-split), §7.6 (limitation rephrased from "partial" to "five-seed sweep").

---

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

**Last Updated**: Session 22 — May 13, 2026
