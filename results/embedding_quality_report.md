# Embedding Quality Comparison

All three models evaluated at the FaceNet backbone output (512D), on the 1,062-sample held-out test set. Distances are L2; silhouette uses cosine. Transfer Learning's backbone is frozen during training, so its row also serves as the vanilla pre-trained FaceNet baseline.

| Metric | Transfer Learning | Progressive Unfreezing | Triplet Loss |
|---|---|---|---|
| Avg Intra-class Distance (L2, lower is better) | 0.651 | 0.575 | 0.337 |
| Avg Inter-class Distance (L2, higher is better) | 0.866 | 1.092 | 1.254 |
| Silhouette Score (cosine, higher is better) | 0.1111 | 0.3203 | 0.1700 |
| Separation Ratio (inter / intra) | 1.330 | 1.901 | 3.724 |
