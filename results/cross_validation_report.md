# 5-Seed Cross-Validation: FaceNet Fine-Tuning Approaches

_Generated: 2026-05-14T18:09:03.616464+00:00_

_Total wallclock: 201.6 min_

_Seeds: [42, 123, 456, 789, 1024]_


## Summary

| Approach | Mean | Std (pop) | Min | Max | n |
|---|---|---|---|---|---|
| **TL** (Transfer Learning (Option A)) | 96.52% | ±0.46% | 95.95% | 97.18% | 5/5 |
| **TLoss** (Triplet Loss (Option C)) | 87.08% | ±10.33% | 67.04% | 96.89% | 5/5 |
| **PU** (Progressive Unfreezing (Option B)) | 94.11% | ±0.59% | 93.31% | 95.10% | 5/5 |

## Per-seed accuracies

| Approach | seed=42 | seed=123 | seed=456 | seed=789 | seed=1024 |
|---|---|---|---|---|---|
| **TL** | 96.89% | 97.18% | 96.42% | 95.95% | 96.14% |
| **TLoss** | 91.43% | 90.02% | 96.89% | 67.04% | 90.02% |
| **PU** | 95.10% | 93.79% | 94.16% | 94.16% | 93.31% |

## Configuration

Each per-seed run was a separate WSL2 subprocess. Approaches:

- **TL**: `bp_face_recognition.vision.training.finetune.facenet_transfer_trainer` 
- **TLoss**: `bp_face_recognition.vision.training.finetune.facenet_triplet_trainer` --batch-size 8
- **PU**: `bp_face_recognition.vision.training.finetune.facenet_progressive_trainer` 
