# Thesis Benchmark Report

Canonical comparison of detection methods, recognition models, and embedding
geometry for the BP-face-recognition thesis. Numbers in this file are derived
from these source artifacts in `results/`:

- Detection (with ground truth, IoU ≥ 0.5): `detection_results_groundtruth.json`
- Recognition (closed-set accuracy on 1,062 test samples): `recognition_results.json`
- Embedding geometry (512D FaceNet backbone, test set): `embedding_quality.json`

The detection benchmark was run on 19 manually annotated surveillance frames
(26 ground-truth boxes total, sources: seccam, seccam_2, webcam). All detectors
were evaluated in isolated CPU-only subprocesses to avoid cross-detector state
collisions. Recognition was evaluated on the held-out test split of the
combined webcam + seccam_2 dataset (1,062 samples, 14 classes, 65/20/15 split).

## Detection: Quality (with ground truth)

| Method | TP | FP | FN | Precision | Recall | F1 | Mean IoU |
|---|---|---|---|---|---|---|---|
| MTCNN | 18 | 7 | 8 | **0.720** | **0.692** | **0.706** | 0.693 |
| MediaPipe | 4 | 2 | 22 | 0.667 | 0.154 | 0.250 | 0.720 |
| Haar Cascade | 2 | 10 | 24 | 0.167 | 0.077 | 0.105 | 0.774 |
| Dlib HOG | 1 | 9 | 25 | 0.100 | 0.038 | 0.056 | 0.580 |

MTCNN dominates F1 on every annotated face it can find. MediaPipe has the best
precision-per-detection but very low recall (15.4%). Haar Cascade and Dlib HOG
both have terrible precision because the bulk of their predictions are spurious
non-face regions.

## Detection: Speed (CPU-only, 800px max)

| Method | Avg (ms) | Min (ms) | Max (ms) | FPS |
|---|---|---|---|---|
| MediaPipe | 4.2 | 3.0 | 6.0 | 238.3 |
| Haar Cascade | 39.2 | 19.7 | 56.4 | 25.5 |
| Dlib HOG | 167.8 | 126.1 | 200.8 | 6.0 |
| MTCNN | 283.3 | 247.9 | 316.8 | 3.5 |

For real-time surveillance (30 FPS = 33 ms / frame), only MediaPipe clears the
bar comfortably. MTCNN and Dlib HOG require frame-skipping or GPU acceleration
to be deployable.

## Recognition: Accuracy and per-class breakdown

| Model | Accuracy | Precision | Recall | F1 | Avg Inference (ms) | Size (MB) |
|---|---|---|---|---|---|---|
| FaceNet TL (Transfer Learning) | 92.84% | 0.932 | 0.928 | 0.928 | 509.6 | 92.7 |
| FaceNet PU (Progressive Unfreezing) | **99.15%** | **0.992** | **0.992** | **0.991** | 487.6 | 271.9 |
| FaceNet TLoss (Triplet Loss) | 94.63% | 0.946 | 0.946 | 0.946 | — | 270.4 |
| EfficientNetB0 (full, seccam_2) | 100.00%¹ | 1.00 | 1.00 | 1.00 | — | 23.8 |
| EfficientNetB0 (float16, seccam_2) | 100.00%¹ | 1.00 | 1.00 | 1.00 | — | 9.0 |

¹ Trained on `seccam_2` (15 classes), not the combined 14-class dataset that
TL/PU/TLoss use. Numbers from training report, not live benchmark on this test
set. Listed for size/quantization comparison only.

TLoss inference time isn't reported here because the saved model is an
embedding-only checkpoint and its accuracy figure is from a KNN-on-embeddings
evaluation rather than a live softmax classifier benchmark.

## Recognition: Per-class accuracy

| Class | Test Samples | TL | PU | TLoss | Training Images |
|---|---|---|---|---|---|
| Yurii | 270 | 98.1% | 100.0% | 98.9% | 1,260 |
| Stranger_1 | 234 | 93.6% | 100.0% | 96.2% | 1,092 |
| Stranger_2 | 180 | 90.6% | 99.4% | 93.9% | 840 |
| Stranger_11 | 108 | 99.1% | 100.0% | 98.1% | 504 |
| Stranger_3 | 108 | 90.7% | 99.1% | 89.8% | 504 |
| Stranger_4 | 63 | 90.5% | 98.4% | 84.1% | 294 |
| Stranger_14 | 18 | 83.3% | 100.0% | 83.3% | 84 |
| Stranger_5 | 18 | 77.8% | 88.9% | 83.3% | 84 |
| Stranger_8 | 18 | 83.3% | 100.0% | 88.9% | 84 |
| Stranger_10 | 9 | 66.7% | 88.9% | 88.9% | 42 |
| Stranger_12 | 9 | 100.0% | 100.0% | 100.0% | 42 |
| Stranger_7 | 9 | 88.9% | 100.0% | 88.9% | 42 |
| Stranger_9 | 9 | 55.6% | 100.0% | 88.9% | 42 |
| Stranger_6 | 9 | 55.6% | 66.7% | 100.0% | 42 |
| **Overall** | **1,062** | **92.84%** | **99.15%** | **94.63%** | **4,956** |

## Embedding geometry (FaceNet backbone, 512D)

All three approaches were re-evaluated at the FaceNet backbone output to
produce a comparison that is fair to Triplet Loss (which optimizes embedding
distances, not class probabilities). Because Transfer Learning's backbone is
frozen, its row also serves as the vanilla pre-trained FaceNet baseline.

| Metric | Transfer Learning | Progressive Unfreezing | Triplet Loss |
|---|---|---|---|
| Avg Intra-class Distance (L2, lower is better) | 0.651 | 0.575 | **0.337** |
| Avg Inter-class Distance (L2, higher is better) | 0.866 | 1.092 | **1.254** |
| Silhouette Score (cosine, higher is better) | 0.111 | **0.320** | 0.170 |
| Separation Ratio (inter / intra) | 1.330 | 1.901 | **3.724** |

**Findings.**

- Triplet Loss produces the *most geometrically separated* embedding space:
  separation ratio of 3.724, almost twice Progressive Unfreezing's 1.901. This
  is direct evidence that triplet loss did what it was designed to do, even
  though it underperformed on classification accuracy.
- Progressive Unfreezing wins the silhouette score (0.320 vs Triplet Loss's
  0.170), reflecting that classification-style training keeps boundary samples
  away from *all* other clusters, not just centroids.
- The two perspectives are complementary: Progressive Unfreezing is the right
  choice for closed-set classification; Triplet Loss is the right choice for
  open-set verification.

## Quantization impact (EfficientNetB0 only)

| Variant | Size (MB) | Compression vs full |
|---|---|---|
| EfficientNetB0 (full) | 23.8 | — |
| EfficientNetB0 (float16) | 9.0 | 62% smaller |

## How to reproduce

```bash
# Detection (ground-truth evaluation, isolated subprocesses per detector)
uv run python scripts/run_detection_eval.py

# Recognition (TL, PU, TLoss live benchmark; closed-set accuracy)
uv run python src/bp_face_recognition/evaluation/thesis_benchmark.py --skip-detection

# Embedding geometry (intra/inter distance, silhouette, separation ratio)
PYTHONPATH=src uv run python src/bp_face_recognition/evaluation/embedding_quality.py

# Training-curve figures (per-approach loss & accuracy plots)
uv run python scripts/plot_training_curves.py
```
