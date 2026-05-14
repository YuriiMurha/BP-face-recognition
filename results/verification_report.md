# Open-Set Verification Results

Verification protocol: 5,000 positive + 5,000 negative pairs sampled from the 1,062-sample held-out test set (RNG seed = 42). L2-normalized 512D FaceNet-backbone embeddings, cosine similarity, threshold swept from -1 to +1 in 500 steps. EER, TAR@FAR, and AUC reported with linear interpolation between adjacent threshold steps.

| Model | EER | EER threshold | TAR @ FAR=1% | TAR @ FAR=0.1% | AUC | # Positive pairs | # Negative pairs |
|---|---|---|---|---|---|---|---|
| Transfer Learning | 0.2958 | +0.2144 | 0.1104 | 0.0472 | 0.7710 | 5000 | 5000 |
| Progressive Unfreezing | 0.0903 | +0.1531 | 0.6322 | 0.3072 | 0.9710 | 5000 | 5000 |
| Triplet Loss | 0.1787 | +0.5266 | 0.3421 | 0.2124 | 0.9096 | 5000 | 5000 |

Lower EER is better. Higher TAR and AUC are better.
