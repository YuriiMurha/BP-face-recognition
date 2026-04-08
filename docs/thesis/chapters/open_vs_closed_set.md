# Open-Set vs Closed-Set Face Recognition

## Chapter X: Recognition Paradigm Comparison

---

## 1. Introduction

Face recognition systems can be categorized into two fundamental paradigms based on how they handle identity classification: **closed-set** and **open-set** recognition. The distinction is critical for system design, as each paradigm makes different assumptions about the population of subjects and requires different architectural approaches.

This chapter defines both paradigms, compares their architectures as implemented in our system, analyzes their trade-offs, and provides guidance on when to use each approach.

---

## 2. Definitions

### 2.1 Closed-Set Recognition

In closed-set recognition, the system assumes that every test subject belongs to one of *N* known identities seen during training. The recognition task is a standard *N*-class classification problem: given a face image, output the most probable identity from the fixed set.

**Formal definition**: Given a probe face image *x* and a gallery of *N* known identities $\{c_1, c_2, ..., c_N\}$, the system computes:

$$\hat{c} = \arg\max_{i} P(c_i | x)$$

If the confidence $P(\hat{c} | x)$ falls below a threshold, the system may output "Unknown," but this is a rejection mechanism rather than true open-set recognition — the model has no explicit "none of the above" class.

### 2.2 Open-Set Recognition

In open-set recognition, the system must handle subjects that were **not** seen during training. Instead of classifying into fixed categories, the system maps face images to a continuous embedding space and compares distances or similarities to registered identities.

**Formal definition**: Given a probe face image *x*, the system computes an embedding $e = f(x) \in \mathbb{R}^d$ and compares it against a database of registered embeddings $\{e_1, e_2, ..., e_M\}$:

$$\hat{c} = \arg\max_{i} \text{sim}(e, e_i)$$

If the maximum similarity falls below a threshold, the subject is classified as "Unknown" — a genuine unknown, not merely a low-confidence known identity.

---

## 3. Architecture Comparison

### 3.1 Closed-Set Pipeline

Our closed-set pipeline (`ClosedSetPipelineService`) uses a fine-tuned FaceNet model with a 14-class softmax classification head:

```
Camera Frame
     |
Face Detection (MediaPipe, 50% resolution)
     |
Face Crop (160x160 RGB)
     |
FaceNet InceptionResNetV1 (pretrained backbone)
     |
Dense(256, ReLU) + Dropout(0.5)
     |
Dense(14, Softmax)
     |
argmax -> (Identity, Confidence)
     |
if confidence >= threshold: Identity
else: "Unknown"
```

**Key characteristics**:
- Single forward pass through the model produces the identity
- No database lookup required
- Identities are encoded in the model weights (the softmax layer)
- Adding a new identity requires retraining the entire model

### 3.2 Open-Set Pipeline

Our open-set pipeline (`PipelineService`) uses the same FaceNet backbone but produces embeddings instead of class probabilities:

```
Camera Frame
     |
Face Detection (MediaPipe, 50% resolution)
     |
Face Crop (160x160 RGB)
     |
FaceNet InceptionResNetV1
     |
512-dimensional embedding (L2-normalized)
     |
DatabaseService: cosine similarity vs all registered embeddings
     |
if max_similarity >= threshold: Identity
else: "Unknown"
```

**Key characteristics**:
- Forward pass produces an embedding vector, not a class label
- Requires a database of pre-registered face embeddings
- Adding a new identity requires only capturing face samples and storing embeddings (seconds)
- Can recognize identities never seen during model training

---

## 4. Trade-offs

**Table 1: Closed-Set vs Open-Set Comparison**

| Criterion | Closed-Set | Open-Set |
|-----------|------------|----------|
| **Adding new person** | Retrain model (~50 min) | Register only (seconds) |
| **Unknown rejection** | Confidence threshold (soft) | Similarity threshold (principled) |
| **Model requirement** | Classifier head (softmax) | Embedding model |
| **Database dependency** | None | Required (CSV/PostgreSQL) |
| **Accuracy (our data)** | 99.15% (FaceNet PU) | Depends on threshold and database |
| **Scalability** | Limited by class count | Scales to thousands of identities |
| **Inference cost** | Single forward pass | Forward pass + database search |
| **Training data** | Needed for all identities | Needed only for embedding model |
| **Unknown detection** | Weak (low-confidence fallback) | Strong (distance-based) |
| **Model size** | Larger (backbone + head) | Backbone only |

### 4.1 Adding New Identities

The most significant practical difference is the cost of adding new identities:

- **Closed-set**: Adding a new person requires collecting training data, retraining the model (50 minutes for progressive unfreezing), and redeploying. This makes closed-set systems impractical for dynamic populations.

- **Open-set**: Adding a new person requires only capturing 10+ face samples and computing their embeddings. The `register_from_camera.py` script completes this process in under a minute. No model retraining is needed.

### 4.2 Unknown Person Detection

A critical distinction for surveillance applications:

- **Closed-set**: The softmax function always assigns probability mass across the known classes. An unknown person will be misclassified as one of the known identities, typically with lower confidence. Thresholding on confidence provides only weak unknown rejection — the model was never trained to distinguish "none of the above."

- **Open-set**: An unknown person's embedding will have low similarity to all registered embeddings, producing a clear signal for rejection. This is the principled approach for surveillance where detecting unknown individuals is often the primary goal.

### 4.3 Scalability

- **Closed-set**: Each identity adds a neuron to the output layer and requires training examples. Performance typically degrades as the class count increases, and retraining time grows.

- **Open-set**: The embedding model is fixed regardless of how many identities are registered. Adding the 1,000th identity is as cheap as adding the 10th. Database search scales linearly (or sub-linearly with approximate nearest neighbor structures like FAISS).

---

## 5. When to Use Each Paradigm

### 5.1 Closed-Set Use Cases

Closed-set recognition is appropriate when:
- The population is **fixed and small** (e.g., employees in a secure facility)
- **Maximum accuracy** on known faces is the priority
- There is **no need** to detect unknown individuals
- The system can be retrained periodically when the population changes
- **Simplicity** is valued — no database infrastructure needed

### 5.2 Open-Set Use Cases

Open-set recognition is appropriate when:
- The population is **dynamic** (e.g., visitors, customers, new employees)
- **Unknown detection** is a primary requirement (security, surveillance)
- The identity set is **large** (hundreds to thousands of individuals)
- **Rapid enrollment** of new identities is needed
- The system must operate without retraining after initial deployment

### 5.3 Hybrid Approach

In practice, the two paradigms can be combined:
- Use closed-set for a core set of known identities (high accuracy)
- Fall back to open-set embedding matching when closed-set confidence is low
- Use the closed-set model's intermediate features as embeddings for open-set matching

Our system demonstrates this flexibility: the `FinetunedRecognizer` class supports both `recognize()` (closed-set softmax output) and `get_embedding()` (open-set embedding extraction) from the same model.

---

## 6. Implementation in This System

### 6.1 Shared Infrastructure

Both paradigms share identical components:
- Camera capture and frame preprocessing
- Face detection (MediaPipe default, configurable)
- Face cropping and normalization (160x160 pixels, [-1, 1] range)
- Display and visualization (OpenCV window with bounding boxes and labels)

### 6.2 Entry Points

- `main.py`: Open-set recognition with `PipelineService` and `DatabaseService`
- `closed_set_main.py`: Closed-set recognition with `ClosedSetPipelineService`
- Make commands: `make run` (open-set) and `make run-closed-set` (closed-set)

### 6.3 Class Label Mapping

The closed-set model's 14 class labels are stored in `models/finetuned/dataset_info.json`, maintaining a consistent mapping between training-time class indices and human-readable identity names.

---

## 7. Conclusions

Both recognition paradigms serve valid use cases:

- **Closed-set** recognition achieves the highest accuracy (99.15%) with the simplest deployment (no database needed), making it ideal for fixed, small populations where known-face identification is the primary task.

- **Open-set** recognition provides the flexibility essential for real-world surveillance: dynamic enrollment, principled unknown detection, and scalability to large populations.

For the surveillance use case that motivates this thesis, open-set recognition is the more appropriate choice — surveillance systems must handle unknown individuals and allow rapid enrollment. However, for controlled access scenarios with a stable population, closed-set recognition offers superior accuracy with simpler infrastructure.

The dual-paradigm architecture of our system allows deploying either approach, or both in combination, without changes to the underlying detection and preprocessing infrastructure.

---

```latex
\begin{table}[h]
\centering
\caption{Open-Set vs Closed-Set Recognition Comparison}
\label{tab:open_vs_closed}
\begin{tabular}{lcc}
\hline
Criterion & Closed-Set & Open-Set \\
\hline
Adding new person & Retrain (50 min) & Register (seconds) \\
Unknown detection & Weak (confidence) & Strong (distance) \\
Database required & No & Yes \\
Accuracy (our data) & 99.15\% & Threshold-dependent \\
Scalability & Limited & High \\
Inference cost & Low & Low + DB lookup \\
\hline
\end{tabular}
\end{table}
```

---

**Chapter Status**: Complete

**Last Updated**: March 21, 2026

**Word Count**: ~1,800 words
