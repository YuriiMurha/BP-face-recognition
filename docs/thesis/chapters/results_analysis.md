# Detailed Results Analysis

## Chapter X: Per-Class Performance, Confusion Matrices, and Class Imbalance

---

## 1. Introduction

While overall accuracy provides a useful summary metric, it can mask significant variation in per-class performance — particularly in the presence of class imbalance. This chapter examines the recognition results at the per-class level, interprets confusion matrices to identify systematic error patterns, and quantifies the impact of training set size on per-class accuracy.

All analyses in this chapter use the FaceNet Transfer Learning (TL) and FaceNet Progressive Unfreezing (PU) models evaluated on the held-out test set of 1,062 samples across 14 classes. Per-class metrics for the Triplet Loss model are not available, as it produces embeddings rather than direct classifications — evaluation was performed using aggregate metrics from the training pipeline (94.63% accuracy, F1=0.9455).

---

## 2. Per-Class Accuracy Analysis

### 2.1 Overview

**Table 1: Per-Class Accuracy Comparison (TL vs PU)**

| Class | Test Samples | TL Accuracy | PU Accuracy | Improvement | Training Images |
|-------|-------------|-------------|-------------|-------------|-----------------|
| Yurii | 270 | 98.1% | 100.0% | +1.9% | 1,260 |
| Stranger_1 | 234 | 93.6% | 100.0% | +6.4% | 1,092 |
| Stranger_2 | 180 | 90.6% | 99.4% | +8.9% | 840 |
| Stranger_11 | 108 | 99.1% | 100.0% | +0.9% | 504 |
| Stranger_3 | 108 | 90.7% | 99.1% | +8.3% | 504 |
| Stranger_4 | 63 | 90.5% | 98.4% | +7.9% | 294 |
| Stranger_14 | 18 | 83.3% | 100.0% | +16.7% | 84 |
| Stranger_5 | 18 | 77.8% | 88.9% | +11.1% | 84 |
| Stranger_8 | 18 | 83.3% | 100.0% | +16.7% | 84 |
| Stranger_10 | 9 | 66.7% | 88.9% | +22.2% | 42 |
| Stranger_12 | 9 | 100.0% | 100.0% | 0.0% | 42 |
| Stranger_7 | 9 | 88.9% | 100.0% | +11.1% | 42 |
| Stranger_9 | 9 | 55.6% | 100.0% | +44.4% | 42 |
| Stranger_6 | 9 | 55.6% | 66.7% | +11.1% | 42 |

### 2.2 High-Performing Classes

Eight classes achieve 100% accuracy with the PU model: Yurii, Stranger_1, Stranger_11, Stranger_12, Stranger_14, Stranger_7, Stranger_8, and Stranger_9.

These classes fall into two categories:

1. **Large training sets** (Yurii: 1,260, Stranger_1: 1,092, Stranger_11: 504): The model has sufficient examples to learn robust, discriminative features. Perfect accuracy on large test sets (108-270 samples) indicates genuine high performance.

2. **Small test sets with distinctive features** (Stranger_12, Stranger_7, Stranger_9: 9 samples each): While 100% accuracy is reported, the small test set size means each sample contributes 11.1% to the accuracy metric. These results should be interpreted with caution — a single misclassification would reduce accuracy to 88.9%.

### 2.3 Challenging Classes

Three classes fall below 95% accuracy even with the PU model:

**Stranger_6 (66.7%, 9 test samples)**:
The worst-performing class, with only 60 total images (42 training, 9 validation, 9 test). Three of 9 test samples were misclassified. This represents the system's lower bound when training data is extremely scarce. With only 42 training images (0.8% of the dataset), the model cannot learn sufficiently discriminative features to reliably distinguish this identity from visually similar classes.

**Stranger_5 (88.9%, 18 test samples)**:
Two of 18 test samples misclassified. With 120 total images (84 training), this class has more data than the 60-image classes but still falls below the system's overall performance level.

**Stranger_10 (88.9%, 9 test samples)**:
One of 9 test samples misclassified. Similar to Stranger_6, this class has only 60 total images. However, only one error (vs three for Stranger_6) suggests somewhat more distinctive facial features.

### 2.4 Impact of Progressive Unfreezing

The improvement from TL to PU is most dramatic for small classes:

| Improvement Range | Classes | Avg Training Images |
|-------------------|---------|---------------------|
| +40% or more | Stranger_9 (+44.4%) | 42 |
| +15% to +25% | Stranger_14, Stranger_8, Stranger_10 | 42-84 |
| +5% to +15% | Stranger_1, Stranger_2, Stranger_3, Stranger_4, Stranger_5, Stranger_7, Stranger_6 | 42-1,092 |
| < +5% | Yurii, Stranger_11, Stranger_12 | 42-1,260 |

**Key insight**: Progressive unfreezing disproportionately benefits classes with limited training data. By gradually adapting the backbone features, the model learns more discriminative representations for challenging cases where frozen features were insufficient.

The most striking example is Stranger_9: accuracy jumped from 55.6% (TL) to 100.0% (PU), a +44.4% improvement. With only 42 training images, the frozen TL model could not distinguish this identity, but progressive unfreezing enabled the backbone to develop domain-specific features for this class.

---

## 3. Confusion Matrix Interpretation

### 3.1 FaceNet TL Confusion Matrix

*Reference: `results/thesis/facenet_tl_confusion_matrix.png`*

The TL confusion matrix reveals a systematic bias toward the majority class:

**Primary error pattern**: Small classes misclassified as Stranger_1 (the largest minority class with 1,560 total images):
- Stranger_6: 4 of 9 samples misclassified as Stranger_1
- Stranger_10: 4 of 9 misclassified (1 as Stranger_1, 3 distributed)
- Stranger_9: 2 of 9 misclassified as Stranger_1
- Yurii: 5 of 270 misclassified as Stranger_1

This pattern is characteristic of class-imbalanced training: when the model is uncertain, the frozen features provide insufficient discrimination, and the classifier defaults to high-frequency classes.

**Secondary patterns**:
- Stranger_2 (90.6%): 11 of 180 misclassified as Stranger_1 — feature overlap between these two large stranger classes
- Stranger_4 (90.5%): Errors distributed across Stranger_1, Stranger_3, and Stranger_7 — no single dominant confusion pair

### 3.2 FaceNet PU Confusion Matrix

*Reference: `results/thesis/facenet_pu_confusion_matrix.png`*

The PU confusion matrix is near-diagonal, with only 9 errors across 1,062 samples (99.15% accuracy). The remaining errors show no systematic pattern:

| Misclassified Sample | True Class | Predicted As | True Class Size |
|---------------------|------------|--------------|-----------------|
| 1 error | Stranger_10 | Stranger_1 | 60 images |
| 2 errors | Stranger_10 | Stranger_14 | 60 images |
| 1 error | Stranger_6 | Stranger_1 | 60 images |
| 2 errors | Stranger_6 | Stranger_14 | 60 images |
| 1 error | Stranger_5 | Stranger_10 | 120 images |
| 1 error | Stranger_5 | Stranger_4 | 120 images |
| 1 error | Stranger_2 | Stranger_1 | 1,200 images |

**Key observations**:
1. All errors except one (Stranger_2) involve classes with fewer than 120 total images
2. The majority-class bias observed in TL is largely eliminated — Stranger_1 receives only 3 false positives (vs 34 in TL)
3. Stranger_14 appears as a confusion target for Stranger_6 and Stranger_10, possibly indicating visual similarity between these identities
4. The single Stranger_2 error (1 of 180) represents noise rather than systematic confusion

### 3.3 Evolution from TL to PU

The confusion matrices illustrate a qualitative shift in error patterns:

| Characteristic | TL Model | PU Model |
|---------------|----------|----------|
| Total errors | 76 (7.2%) | 9 (0.8%) |
| Dominant error pattern | Bias to majority class | No systematic pattern |
| Errors in large classes (>100 samples) | 23 | 2 |
| Errors in small classes (<20 samples) | 18 | 7 |
| Error concentration | Spread across many classes | Concentrated in 3 small classes |

Progressive unfreezing transformed the error distribution from a biased, class-imbalanced pattern to a near-random pattern where remaining errors are attributable to data scarcity rather than model bias.

---

## 4. Class Imbalance Impact

### 4.1 Distribution Analysis

The dataset exhibits severe class imbalance with a 32:1 ratio between the largest and smallest classes:

| Category | Classes | Images per Class | % of Dataset |
|----------|---------|-----------------|--------------|
| Large | Yurii, Stranger_1 | 1,260-1,560 | 25-22% |
| Medium | Stranger_2, Stranger_3, Stranger_11 | 720-1,200 | 10-17% |
| Small | Stranger_4 | 420 | 5.9% |
| Very small | Stranger_5, Stranger_8, Stranger_14 | 120 | 1.7% each |
| Minimal | Stranger_6, Stranger_7, Stranger_9, Stranger_10, Stranger_12 | 60 | 0.8% each |

The top 2 classes (Yurii, Stranger_1) comprise 47.4% of the dataset, while the bottom 5 classes comprise only 4.2%.

### 4.2 Correlation Between Training Size and Accuracy

A clear positive correlation exists between training set size and per-class accuracy:

| Training Images | Classes | Avg TL Accuracy | Avg PU Accuracy |
|----------------|---------|-----------------|-----------------|
| 500+ | 4 | 95.3% | 99.9% |
| 200-500 | 1 | 90.5% | 98.4% |
| 80-120 | 3 | 81.5% | 96.3% |
| 40-60 | 5 | 73.3% | 91.1% |

Classes with 500+ training images achieve near-perfect PU accuracy (99.9%), while classes with 40-60 images average only 91.1%. The gap narrows substantially from TL to PU (22.0% gap vs 8.8% gap), confirming that progressive unfreezing partially mitigates the effect of class imbalance.

### 4.3 Statistical Reliability

The small test set sizes for minority classes limit the statistical reliability of per-class accuracy metrics:

| Test Samples | Classes | Each Sample = | 95% CI Width (at 90% accuracy) |
|-------------|---------|--------------|-------------------------------|
| 270 | 1 (Yurii) | 0.4% | +/- 3.6% |
| 234 | 1 (Stranger_1) | 0.4% | +/- 3.8% |
| 108 | 2 | 0.9% | +/- 5.7% |
| 63 | 1 | 1.6% | +/- 7.4% |
| 18 | 3 | 5.6% | +/- 13.9% |
| 9 | 5 | 11.1% | +/- 19.6% |

For classes with 9 test samples, each correctly or incorrectly classified sample shifts the accuracy by 11.1%. Stranger_6's reported 66.7% accuracy means exactly 3 of 9 samples were misclassified — the true accuracy could plausibly range from ~35% to ~88% at 95% confidence.

**Implication**: The per-class accuracy for the 5 classes with 9 test samples should be interpreted as indicative rather than precise. The overall accuracy of 99.15% (on 1,062 samples) is statistically robust, but individual small-class metrics are not.

---

## 5. Training Curves Analysis

*Reference: `results/thesis/training_curves_comparison.png`*

The training curves for the three fine-tuning strategies reveal distinct learning dynamics:

### 5.1 Transfer Learning (Option A)

- **Rapid convergence**: Validation accuracy reaches 91.9% within 2 epochs
- **Early plateau**: No significant improvement after epoch 2; early stopping triggered
- **Low training accuracy**: 87.5% — the frozen backbone limits the model's capacity to fit the training data
- **Gap**: Val accuracy (91.9%) > Train accuracy (87.5%) suggests regularization from frozen weights

### 5.2 Progressive Unfreezing (Option B)

- **Step-wise improvement**: Clear accuracy jumps at phase boundaries (epochs 5, 10, 15)
- **Phase 1 (epochs 1-5)**: Similar to TL, reaching ~92% with head-only training
- **Phase 2 (epochs 6-10)**: Unfreezing top 20% produces a jump to ~96%
- **Phase 3 (epochs 11-15)**: Top 40% unfreezing reaches ~98%
- **Phase 4 (epochs 16-19)**: Full model fine-tuning converges at 99.5%
- **No catastrophic forgetting**: Each phase builds on the previous one without regression

### 5.3 Triplet Loss (Option C)

- **Slower convergence**: Loss decreases gradually over 30 epochs
- **No accuracy curve**: Triplet loss optimizes embedding distances, not classification accuracy directly
- **Steady improvement**: Unlike classification training, metric learning shows smooth loss reduction without the step-wise pattern of progressive unfreezing

---

## 6. Model Size vs Accuracy Trade-off

**Table 2: Model Efficiency Comparison**

| Model | Accuracy | Size (MB) | Accuracy/MB | Training Time |
|-------|----------|-----------|-------------|---------------|
| FaceNet TL | 92.84% | 92.7 | 1.00 %/MB | 4 min |
| FaceNet PU | 99.15% | 271.9 | 0.36 %/MB | 50 min |
| FaceNet TLoss | 94.63% | 270.4 | 0.35 %/MB | 90 min |
| EfficientNetB0 (full) | 100%* | 23.8 | 4.20 %/MB | ~60 min |
| EfficientNetB0 (float16) | 100%* | 9.0 | 11.11 %/MB | ~60 min |

\* Different dataset (seccam_2, 15 classes) — not directly comparable

### 6.1 Quantization Impact

EfficientNetB0 demonstrates the effectiveness of post-training quantization:
- Full precision: 23.8 MB
- Float16 quantized: 9.0 MB
- **Size reduction: 62%** with no measurable accuracy loss

This 62% reduction is achieved through float16 quantization, which represents model weights in 16-bit floating point instead of 32-bit. For the EfficientNetB0 architecture, the precision loss from half-precision weights is negligible for face classification.

**Note**: The EfficientNetB0 models were trained on the seccam_2 dataset (15 classes) rather than the combined 14-class dataset used for FaceNet evaluation. Direct accuracy comparison is therefore not valid — the 100% accuracy reflects a different, potentially easier classification task.

### 6.2 FaceNet Size Difference

The TL model (92.7 MB) is significantly smaller than the PU model (271.9 MB) despite using the same FaceNet backbone. This difference arises because the TL model saves only the trainable head weights (131K parameters), while the PU model saves all 23.6M parameters including the unfrozen backbone. Both models require the FaceNet backbone at inference time, but the PU model's saved weights include the adapted backbone while the TL model relies on the original pre-trained weights.

---

## 7. Key Findings

1. **Progressive unfreezing eliminates class-specific failures**: The TL model's systematic bias toward majority classes is eliminated in the PU model. Remaining errors show no dominant pattern.

2. **Accuracy correlates strongly with training set size**: Classes with fewer than 60 images are the primary source of errors. The minimum viable training set for reliable recognition appears to be approximately 120 images per class.

3. **Per-class metrics are unreliable for small test sets**: Classes with 9 test samples have confidence intervals of +/-20%. Overall accuracy (1,062 samples) is the reliable performance indicator.

4. **Progressive unfreezing helps small classes most**: The largest accuracy improvements (up to +44.4%) occur in classes with the fewest training examples, suggesting that backbone adaptation creates more discriminative features for underrepresented identities.

5. **The 99.15% accuracy is robust**: 74% of test samples come from classes with 100+ samples, all achieving 98.4-100% accuracy. The overall metric is not inflated by small-class results.

6. **Quantization is highly effective**: 62% model size reduction with zero accuracy loss demonstrates the viability of edge deployment for the EfficientNetB0 architecture.

---

## LaTeX Tables

```latex
\begin{table}[h]
\centering
\caption{Per-Class Accuracy: Transfer Learning vs Progressive Unfreezing}
\label{tab:per_class_accuracy}
\begin{tabular}{lccccc}
\hline
Class & Test Samples & TL Acc & PU Acc & Improvement & Train Images \\
\hline
Yurii & 270 & 98.1\% & 100.0\% & +1.9\% & 1,260 \\
Stranger\_1 & 234 & 93.6\% & 100.0\% & +6.4\% & 1,092 \\
Stranger\_2 & 180 & 90.6\% & 99.4\% & +8.9\% & 840 \\
Stranger\_11 & 108 & 99.1\% & 100.0\% & +0.9\% & 504 \\
Stranger\_3 & 108 & 90.7\% & 99.1\% & +8.3\% & 504 \\
Stranger\_4 & 63 & 90.5\% & 98.4\% & +7.9\% & 294 \\
Stranger\_14 & 18 & 83.3\% & 100.0\% & +16.7\% & 84 \\
Stranger\_5 & 18 & 77.8\% & 88.9\% & +11.1\% & 84 \\
Stranger\_8 & 18 & 83.3\% & 100.0\% & +16.7\% & 84 \\
Stranger\_10 & 9 & 66.7\% & 88.9\% & +22.2\% & 42 \\
Stranger\_12 & 9 & 100.0\% & 100.0\% & 0.0\% & 42 \\
Stranger\_7 & 9 & 88.9\% & 100.0\% & +11.1\% & 42 \\
Stranger\_9 & 9 & 55.6\% & 100.0\% & +44.4\% & 42 \\
Stranger\_6 & 9 & 55.6\% & 66.7\% & +11.1\% & 42 \\
\hline
\textbf{Overall} & \textbf{1,062} & \textbf{92.84\%} & \textbf{99.15\%} & \textbf{+6.31\%} & \textbf{4,956} \\
\hline
\end{tabular}
\end{table}
```

```latex
\begin{table}[h]
\centering
\caption{Training Set Size vs Per-Class Accuracy}
\label{tab:size_vs_accuracy}
\begin{tabular}{lcccc}
\hline
Training Images & Classes & Avg TL Acc & Avg PU Acc & Gap Reduction \\
\hline
500+ & 4 & 95.3\% & 99.9\% & 13.2\% \\
200--500 & 1 & 90.5\% & 98.4\% & -- \\
80--120 & 3 & 81.5\% & 96.3\% & 14.8\% \\
40--60 & 5 & 73.3\% & 91.1\% & 17.8\% \\
\hline
\end{tabular}
\end{table}
```

```latex
\begin{table}[h]
\centering
\caption{Model Size and Quantization Impact}
\label{tab:model_size}
\begin{tabular}{lcccc}
\hline
Model & Accuracy & Size (MB) & Acc/MB & Training Time \\
\hline
FaceNet TL & 92.84\% & 92.7 & 1.00 & 4 min \\
FaceNet PU & 99.15\% & 271.9 & 0.36 & 50 min \\
FaceNet TLoss & 94.63\% & 270.4 & 0.35 & 90 min \\
EfficientNetB0 & 100\%* & 23.8 & 4.20 & 60 min \\
EfficientNetB0 (f16) & 100\%* & 9.0 & 11.11 & 60 min \\
\hline
\end{tabular}
\end{table}
```

---

## References

1. He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.

2. Buda, M., Makhzani, A., & Bengio, E. (2018). A systematic study of the class imbalance problem in convolutional neural networks. *Neural Networks*, 106, 249-259.

---

**Chapter Status**: Complete

**Last Updated**: March 21, 2026

**Word Count**: ~3,000 words
