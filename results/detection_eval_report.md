# Detection Evaluation with Ground Truth

**Test set:** 19 surveillance frames (26 annotated face bounding boxes).

**Matching:** greedy IoU at threshold 0.5. Images resized so max(h,w) ≤ 800. Each detector evaluated in an isolated subprocess (CPU-only) to avoid cross-detector state collisions.

## Detection Quality (P / R / F1)

| Method | TP | FP | FN | Precision | Recall | F1 | Mean IoU |
|---|---|---|---|---|---|---|---|
| MediaPipe | 4 | 2 | 22 | 0.667 | 0.154 | 0.250 | 0.720 |
| Haar Cascade | 2 | 10 | 24 | 0.167 | 0.077 | 0.105 | 0.774 |
| Dlib HOG | 1 | 9 | 25 | 0.100 | 0.038 | 0.056 | 0.580 |
| MTCNN | 18 | 7 | 8 | 0.720 | 0.692 | 0.706 | 0.693 |

## Detection Speed (CPU-only)

| Method | Avg (ms) | Min (ms) | Max (ms) | FPS |
|---|---|---|---|---|
| MediaPipe | 4.2 | 3.0 | 6.0 | 238.3 |
| Haar Cascade | 39.2 | 19.7 | 56.4 | 25.5 |
| Dlib HOG | 167.8 | 126.1 | 200.8 | 6.0 |
| MTCNN | 283.3 | 247.9 | 316.8 | 3.5 |
