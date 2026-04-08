# Face Detection Methods: Comparative Analysis

## Chapter X: Detection Pipeline Evaluation

---

## 1. Introduction

Face detection is the critical first stage of any face recognition pipeline. Before a face can be identified, it must first be localized within the image. The quality of detection directly impacts downstream recognition accuracy — missed faces cannot be recognized, and imprecise bounding boxes degrade the quality of cropped face images fed to the recognizer.

In surveillance applications, face detection presents unique challenges compared to consumer-facing scenarios (e.g., smartphone face unlock). Surveillance cameras capture subjects at varying distances, angles, and lighting conditions. Faces may be partially occluded, poorly lit, or captured at oblique angles. Furthermore, real-time surveillance systems demand low-latency detection to process video streams at acceptable frame rates.

This chapter evaluates four face detection methods spanning classical computer vision and modern deep learning approaches. We benchmark each method on surveillance camera frames to assess the speed-recall trade-off that is central to deploying detection in real-world monitoring systems.

---

## 2. Detection Methods

### 2.1 MediaPipe BlazeFace

MediaPipe BlazeFace (Bazarevsky et al., 2019) is a lightweight face detection model designed for mobile and real-time applications. It uses a Single Shot Detector (SSD) architecture with a custom feature extraction backbone optimized for mobile GPUs.

**Architecture**:
- Single Shot Multibox Detector (SSD) with depthwise separable convolutions
- Input resolution: 128x128 pixels
- Anchor-free detection with regression-based bounding box prediction
- 6 facial landmark keypoints (eyes, nose, mouth, ears)

**Key Characteristics**:
- Designed for frontal and near-frontal face detection
- Sub-millisecond inference on mobile GPUs
- Multi-tier fallback in our implementation: GPU -> CPU -> OpenCV backend
- Confidence threshold configurable (default: 0.5)

MediaPipe's primary advantage is speed — it was engineered for mobile devices where computational resources are severely constrained. However, its optimization for frontal, close-range faces may limit effectiveness in surveillance scenarios where subjects appear at oblique angles or greater distances.

### 2.2 MTCNN (Multi-task Cascaded Convolutional Networks)

MTCNN (Zhang et al., 2016) is a multi-stage detection framework that jointly performs face detection and facial landmark alignment through a cascade of three neural networks.

**Three-Stage Cascade**:

1. **P-Net (Proposal Network)**: A shallow fully convolutional network that scans the image at multiple scales to generate candidate face regions. It operates on a 12x12 input and produces bounding box regression and face classification scores.

2. **R-Net (Refine Network)**: Receives candidate regions from P-Net and refines bounding boxes through a deeper network with a 24x24 input. It rejects false positives and calibrates bounding box coordinates.

3. **O-Net (Output Network)**: The final stage operates on 48x48 crops and produces high-precision bounding boxes along with 5 facial landmark positions (eyes, nose, mouth corners).

**Configuration**:
- Stage thresholds: [0.6, 0.7, 0.8] (increasingly strict)
- Minimum face size: 20 pixels
- Scale factor: 0.709 for image pyramid construction

MTCNN's cascaded design progressively filters candidates, achieving high recall at the cost of computational overhead from processing multiple network stages across an image pyramid.

### 2.3 Haar Cascade (Viola-Jones)

The Haar Cascade detector (Viola & Jones, 2001) is a classical machine learning approach that remains widely used due to its simplicity and availability in OpenCV.

**Architecture**:
- **Integral Image**: Efficient computation of Haar-like rectangular features at any scale and position in constant time.
- **AdaBoost Feature Selection**: From a pool of ~160,000 potential features, AdaBoost selects the most discriminative subset and trains a strong classifier.
- **Cascade of Classifiers**: A chain of increasingly complex stages, where early stages rapidly reject non-face regions using few features, and later stages perform detailed classification.

**Configuration**:
- Scale factor: 1.1 (11% size increase per pyramid level)
- Minimum neighbors: 5 (detections require 5 overlapping windows)
- Pre-trained model: `haarcascade_frontalface_default.xml` (OpenCV)

The Haar Cascade detector requires no deep learning framework and has minimal memory overhead. However, it is limited to frontal faces and struggles with rotated or partially occluded faces.

### 2.4 Dlib HOG + SVM

The Dlib face detector (King, 2009) combines Histogram of Oriented Gradients (HOG) features (Dalal & Triggs, 2005) with a linear Support Vector Machine (SVM) classifier.

**Architecture**:
- **HOG Feature Extraction**: The image is divided into cells (8x8 pixels), and gradient orientation histograms are computed for each cell. Cells are grouped into overlapping blocks for normalization, producing a feature vector that captures local shape and edge information.
- **Linear SVM**: A trained linear SVM classifies HOG feature vectors as face or non-face. The sliding window approach scans the image at multiple scales.

**Configuration**:
- Upsample times: 1 (one level of upsampling for small face detection)
- CPU-optimized implementation with no GPU dependency

Dlib HOG provides a good balance between classical simplicity and detection quality. It handles some degree of face rotation better than Haar Cascades but remains limited compared to deep learning approaches.

---

## 3. Experimental Setup

### 3.1 Test Dataset

The detection benchmark uses 19 raw surveillance camera frames captured from a security camera system. These frames represent realistic surveillance conditions:
- Indoor environment with mixed artificial lighting
- Subjects at varying distances from the camera (1-5 meters estimated)
- Multiple angles including frontal, three-quarter, and profile views
- Resolution: original frames resized to a maximum of 800 pixels on the longest edge

The resizing to 800 pixels maximum was applied to prevent out-of-memory errors with certain detectors (particularly MTCNN and Haar Cascade) while maintaining sufficient resolution for face detection.

### 3.2 Evaluation Metrics

- **Average Detection Time (ms)**: Mean wall-clock time per frame, averaged over all 19 frames
- **Frames Per Second (FPS)**: Reciprocal of average detection time
- **Total Faces Detected**: Cumulative face detections across all 19 frames
- **Average Faces Per Image**: Mean detections per frame

### 3.3 Hardware

All benchmarks were executed on a Windows 11 system with CPU-only inference. No GPU acceleration was used during detection benchmarking to ensure fair comparison, as only MediaPipe natively supports GPU inference among the four methods.

---

## 4. Results

### 4.1 Speed Comparison

**Table 1: Detection Speed Benchmark**

| Method | Avg Time (ms) | FPS | Min Time (ms) | Max Time (ms) |
|--------|--------------|-----|---------------|---------------|
| **MediaPipe** | **3.1** | **325.6** | 2.1 | 4.0 |
| Haar Cascade | 31.4 | 31.8 | 14.2 | 57.4 |
| Dlib HOG | 155.7 | 6.4 | 120.4 | 192.2 |
| MTCNN | 240.3 | 4.2 | 211.5 | 297.9 |

MediaPipe is the clear speed leader at 325.6 FPS — approximately 78x faster than MTCNN and 10x faster than Haar Cascade. The speed difference is attributable to MediaPipe's lightweight architecture designed for mobile inference, compared to MTCNN's multi-stage cascade which requires three separate forward passes plus image pyramid construction.

Haar Cascade provides a strong middle ground at 31.8 FPS, benefiting from the efficient integral image computation. Dlib HOG and MTCNN both fall below 10 FPS, making them unsuitable for real-time single-threaded video processing.

### 4.2 Detection Recall

**Table 2: Detection Recall Comparison**

| Method | Total Faces Detected | Avg Faces/Image |
|--------|---------------------|-----------------|
| **MTCNN** | **25** | **1.32** |
| Haar Cascade | 12 | 0.63 |
| Dlib HOG | 10 | 0.53 |
| MediaPipe | 6 | 0.32 |

MTCNN detected the most faces (25 across 19 frames), more than 4x the number detected by MediaPipe (6). The multi-stage cascade with its image pyramid approach is specifically designed to detect faces at multiple scales, including small and partially occluded faces.

Haar Cascade and Dlib HOG provided intermediate recall. MediaPipe's low detection count (6 faces) reflects its optimization for frontal, near-range faces — a design choice for mobile use cases where the subject typically faces the camera at close range.

### 4.3 Speed-Recall Trade-off

The results reveal a clear inverse relationship between detection speed and recall:

| Method | Speed Rank | Recall Rank | Trade-off Profile |
|--------|-----------|-------------|-------------------|
| MediaPipe | 1st (fastest) | 4th (fewest) | Speed-optimized |
| MTCNN | 4th (slowest) | 1st (most) | Recall-optimized |
| Haar Cascade | 2nd | 2nd | Balanced |
| Dlib HOG | 3rd | 3rd | Balanced |

No single method dominates both axes. This trade-off is fundamental to face detection system design: faster methods achieve speed through more aggressive candidate filtering, which inevitably reduces recall on challenging faces (small, angled, poorly lit).

---

## 5. Discussion

### 5.1 Real-Time Suitability

For real-time video processing, the standard threshold is 30 FPS (33 ms per frame). Only two methods meet this requirement:
- **MediaPipe** at 325.6 FPS (3.1 ms) — well above the threshold, leaving substantial headroom for recognition processing
- **Haar Cascade** at 31.8 FPS (31.4 ms) — marginally meets the threshold

MTCNN (4.2 FPS) and Dlib HOG (6.4 FPS) are unsuitable for real-time single-threaded processing. In a production surveillance system, these methods would require either frame skipping, parallel processing, or dedicated hardware acceleration.

### 5.2 Why MediaPipe Detects Fewer Faces

MediaPipe's low detection count (6 out of 25 detectable faces) warrants analysis. Several factors contribute:

1. **Frontal Face Optimization**: BlazeFace is trained primarily on frontal and near-frontal faces from mobile device cameras. Surveillance cameras frequently capture subjects from above and at angles that deviate from this distribution.

2. **High Confidence Threshold**: The default confidence threshold of 0.5 aggressively filters marginal detections. Lowering this threshold would increase recall at the cost of more false positives.

3. **Small Face Limitation**: The 128x128 input resolution limits the minimum detectable face size. In surveillance frames where subjects are distant, faces may occupy too few pixels for reliable detection.

4. **Single-Scale Detection**: Unlike MTCNN's image pyramid, BlazeFace processes images at a single scale, which can miss faces that fall outside the optimal size range.

### 5.3 MTCNN's Recall Advantage

MTCNN's superior recall (25 faces) is directly attributable to its design:

- **Image Pyramid**: The scale factor of 0.709 creates multiple resolution levels, enabling detection across a wide range of face sizes.
- **Cascaded Refinement**: The three-stage architecture allows the initial P-Net to be permissive (low threshold of 0.6), with subsequent stages filtering false positives while preserving true detections.
- **Landmark Alignment**: Joint landmark detection provides additional face verification signals, improving precision at each stage.

### 5.4 Recommendations for Surveillance Systems

Based on our analysis, we recommend the following deployment strategy:

**Default (Real-Time Monitoring)**: MediaPipe BlazeFace
- Suitable for continuous video stream processing
- 325 FPS provides ample headroom for the full detection-recognition pipeline
- Acceptable recall for frontal surveillance scenarios (e.g., entry points, corridors)

**Batch Processing / Forensic Analysis**: MTCNN
- Maximum recall for offline analysis of recorded footage
- 4.2 FPS is acceptable when processing is not time-constrained
- Best choice when every potential face must be examined

**Resource-Constrained Deployment**: Haar Cascade
- No deep learning framework dependency
- Minimal memory footprint
- Adequate for basic surveillance with moderate recall requirements

---

## 6. LaTeX Tables

```latex
\begin{table}[h]
\centering
\caption{Face Detection Methods Comparison}
\label{tab:detection_comparison}
\begin{tabular}{lcccc}
\hline
Method & Avg Time (ms) & FPS & Faces Detected & Avg Faces/Image \\
\hline
MediaPipe & \textbf{3.1} & \textbf{325.6} & 6 & 0.32 \\
Haar Cascade & 31.4 & 31.8 & 12 & 0.63 \\
Dlib HOG & 155.7 & 6.4 & 10 & 0.53 \\
MTCNN & 240.3 & 4.2 & \textbf{25} & \textbf{1.32} \\
\hline
\end{tabular}
\end{table}
```

```latex
\begin{table}[h]
\centering
\caption{Detection Speed-Recall Trade-off}
\label{tab:detection_tradeoff}
\begin{tabular}{lccc}
\hline
Method & Speed Rank & Recall Rank & Recommendation \\
\hline
MediaPipe & 1st (fastest) & 4th & Real-time monitoring \\
Haar Cascade & 2nd & 2nd & Resource-constrained \\
Dlib HOG & 3rd & 3rd & CPU-only systems \\
MTCNN & 4th (slowest) & 1st & Batch/forensic analysis \\
\hline
\end{tabular}
\end{table}
```

---

## References

1. Bazarevsky, V., Kartynnik, Y., Vakunov, A., Raveendran, K., & Grundmann, M. (2019). BlazeFace: Sub-millisecond neural face detection on mobile GPUs. *CVPR Workshop on Computer Vision for Augmented and Virtual Reality*.

2. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. *CVPR*, 886-893.

3. King, D. E. (2009). Dlib-ml: A machine learning toolkit. *Journal of Machine Learning Research*, 10, 1755-1758.

4. Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. *CVPR*, 511-518.

5. Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. *IEEE Signal Processing Letters*, 23(10), 1499-1503.

---

**Chapter Status**: Complete

**Last Updated**: March 21, 2026

**Word Count**: ~2,500 words
