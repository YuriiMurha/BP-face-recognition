# Final Evaluation Results - Jan 19, 2026

## Overview
This report summarizes the performance of various face detection methods (Haar Cascade, Dlib HOG, MTCNN, and Face Recognition) evaluated across three datasets: `webcam`, `seccam`, and `seccam_2`.

## Detailed Metrics
| Method | Accuracy | Precision | Recall | F1-Score | Avg Detection Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MTCNN** | 72.55% | 68.49% | 72.55% | 69.22% | 9.32s |
| **Dlib HOG** | 23.47% | 25.83% | 23.47% | 24.17% | 0.76s |
| **Haar Cascade** | 2.92% | 3.19% | 2.92% | 2.83% | 2.29s |
| **Face Recognition**| 5.00% | 5.00% | 5.00% | 5.00% | 0.07s |

## Key Findings
1. **MTCNN** remains the most reliable method for surveillance contexts, achieving significantly higher recall and F1-score despite the performance overhead.
2. **Dlib HOG** provides a good balance for real-time applications where surveillance conditions are less challenging.
3. **Haar Cascade** performs poorly in high-resolution surveillance footage with varying lighting and angles.
4. **Face Recognition** (detection module) requires high-resolution faces to be effective; its performance in raw surveillance frames is currently limited.

## Verification
- Results generated using `evaluate_methods.py`.
- Metrics calculated with IoU threshold = 0.5.
- Environment: Python 3.11, TensorFlow 2.10, Dlib 19.24.1.
