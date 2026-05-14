# FaceNet Fine-Tuning: Final Results and Deployment Recommendations

**Last updated**: May 13, 2026 (final, all approaches complete)
**Status**: ✅ Closed-set + open-set evaluation complete across all three approaches

This document summarizes deployment-relevant findings from the thesis. For the
underlying numbers and methodology see
`results/thesis_benchmark_report.md` and the source JSON artifacts in
the same directory.

## At-a-glance

| Approach | Closed-set Accuracy | Training Time | Embedding Separation Ratio | Recommended Use Case |
|---|---|---|---|---|
| **Transfer Learning (TL)** | 92.84% | ~4 min | 1.33 (= vanilla FaceNet) | Rapid prototyping; resource-constrained baseline |
| **Progressive Unfreezing (PU)** | **99.15%** | ~50 min | 1.90 | Closed-set production identification |
| **Triplet Loss (TLoss)** | 94.63% | ~90 min | **3.72** | Open-set verification (1:1) and registration-based identification |

Numbers come from a 1,062-sample held-out test set (14 classes, 65/20/15 split).
The separation ratio is the average inter-class L2 distance divided by the
average intra-class L2 distance, computed on the 512-dimensional FaceNet
backbone output — higher is better.

## Key findings

1. **Progressive Unfreezing is the closed-set classification winner.** It
   beats Transfer Learning by 6.31 percentage points and Triplet Loss by 4.52
   points, and does so in roughly half the training time of Triplet Loss.

2. **Triplet Loss is the embedding-geometry winner.** Its separation ratio of
   3.72 is almost twice Progressive Unfreezing's 1.90 — direct evidence that
   triplet training does what it was designed to do, even though it loses on
   the classification metric. This makes it the right choice for open-set
   verification despite the classification accuracy gap.

3. **The original hypothesis H3 (Triplet Loss > 97% classification accuracy)
   was not confirmed** on this dataset, primarily because we used random
   online triplet mining rather than semi-hard negative mining as in the
   original FaceNet paper. The geometric metrics indicate this is a training
   strategy issue, not a model-capacity issue.

4. **The frozen-backbone baseline (TL) is essentially vanilla pre-trained
   FaceNet** for the purposes of embedding geometry. Its separation ratio of
   1.33 is the floor we improve against.

## Deployment guidance

| Scenario | Recommended Approach |
|---|---|
| Need results in under 10 minutes of training | TL |
| Maximum closed-set accuracy on a fixed identity set | PU |
| Open-set verification (face matching against a database that can be edited) | TLoss |
| Resource-constrained edge device, fixed identity set | TL with smaller head + post-training quantization |
| Pipeline that already runs FaceNet without fine-tuning | TL (no architectural change required) |

## Detection pairing

The recognition models above pair with one of four detection methods:

| Detector | F1 (IoU≥0.5) | Speed (CPU, 800px) | Use case |
|---|---|---|---|
| MediaPipe BlazeFace | 0.250 | 238 FPS | Real-time monitoring (only viable >30 FPS option) |
| MTCNN | **0.706** | 3.5 FPS | Offline batch / attentive replay (best F1) |
| Haar Cascade | 0.105 | 25 FPS | Not recommended (low precision) |
| Dlib HOG | 0.056 | 6 FPS | Not recommended (low precision) |

The system defaults to MediaPipe for real-time work because it is the only
method that meets the 30 FPS real-time threshold without GPU. MTCNN is
exposed as a configurable alternative for offline batch processing where
recall matters more than latency.

## Artifacts

- Trained models: `src/bp_face_recognition/models/finetuned/`
  - `facenet_transfer_v1.0.keras` (93 MB)
  - `facenet_progressive_v1.0.keras` (272 MB)
  - `facenet_triplet_v1.0.weights.h5` (95 MB, weights-only — load by rebuilding the FaceNet base via `keras_facenet.FaceNet()` and calling `load_weights`)
- Training histories: `*_history.json` in the same directory
- Benchmark results: `results/` (this directory has the canonical numbers)
- Thesis: `thesis/chapters/07-results.md` (Section 7.4.2 documents the
  embedding-geometry analysis in detail)
