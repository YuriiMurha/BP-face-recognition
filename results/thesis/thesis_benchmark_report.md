# Thesis Benchmark Report

Generated: 2026-03-18 19:07:23

## Detection Methods Comparison

| Method | Avg Time (ms) | FPS | Faces Detected | Avg Faces/Image |
|--------|--------------|-----|----------------|------------------|
| MediaPipe | 3.1 | 325.6 | 6 | 0.32 |
| MTCNN | 240.3 | 4.2 | 25 | 1.32 |
| Haar Cascade | 31.4 | 31.8 | 12 | 0.63 |
| Dlib HOG | 155.7 | 6.4 | 10 | 0.53 |

## Recognition Models Comparison

| Model | Accuracy (%) | Precision | Recall | F1 | Inference (ms) | Size (MB) |
|-------|-------------|-----------|--------|-----|----------------|----------|
| FaceNet TL | 92.84 | 0.9316 | 0.9284 | 0.9276 | 509.6 | 92.7 |
| FaceNet PU | 99.15 | 0.9918 | 0.9915 | 0.9912 | 487.6 | 271.9 |
| FaceNet TLoss | 94.63 | 0.9460 | 0.9463 | 0.9455 | 0.0 | 270.4 |
| EfficientNetB0 (full) | 100.00 | 1.0000 | 1.0000 | 1.0000 | 0.0 | 23.8 |
| EfficientNetB0 (float16) | 100.00 | 1.0000 | 1.0000 | 1.0000 | 0.0 | 9.0 |

## Per-Class Accuracy

| Class | FaceNet TL | FaceNet PU |
|-------|--------|--------|
| Stranger_1 | 93.6% | 100.0% |
| Stranger_10 | 66.7% | 88.9% |
| Stranger_11 | 99.1% | 100.0% |
| Stranger_12 | 100.0% | 100.0% |
| Stranger_14 | 83.3% | 100.0% |
| Stranger_2 | 90.6% | 99.4% |
| Stranger_3 | 90.7% | 99.1% |
| Stranger_4 | 90.5% | 98.4% |
| Stranger_5 | 77.8% | 88.9% |
| Stranger_6 | 55.6% | 66.7% |
| Stranger_7 | 88.9% | 100.0% |
| Stranger_8 | 83.3% | 100.0% |
| Stranger_9 | 55.6% | 100.0% |
| Yurii | 98.1% | 100.0% |

### Notes

- **FaceNet TLoss**: Embedding model — metrics from training evaluation, not live benchmark
- **EfficientNetB0 (full)**: Trained on seccam_2 (15 classes). Accuracy from training report. Full precision.
- **EfficientNetB0 (float16)**: Trained on seccam_2 (15 classes). Accuracy from training report. Float16 quantized.

## Quantization Impact

| Variant | Size (MB) | Compression |
|---------|----------|-------------|
| EfficientNetB0 (full) | 23.8 | — |
| EfficientNetB0 (float16) | 9.0 | — |
| EfficientNetB0 (float16) | 9.0 | 62% reduction |

## LaTeX Table (Recognition)

```latex
\begin{table}[h]
\centering
\caption{Recognition Model Comparison}
\begin{tabular}{lcccccc}
\hline
Model & Accuracy & Precision & Recall & F1 & Time (ms) & Size (MB) \\
\hline
FaceNet TL & 92.84\% & 0.9316 & 0.9284 & 0.9276 & 509.6 & 92.7 \\
FaceNet PU & 99.15\% & 0.9918 & 0.9915 & 0.9912 & 487.6 & 271.9 \\
FaceNet TLoss & 94.63\% & 0.9460 & 0.9463 & 0.9455 & 0.0 & 270.4 \\
EfficientNetB0 (full) & 100.00\% & 1.0000 & 1.0000 & 1.0000 & 0.0 & 23.8 \\
EfficientNetB0 (float16) & 100.00\% & 1.0000 & 1.0000 & 1.0000 & 0.0 & 9.0 \\
\hline
\end{tabular}
\end{table}
```

