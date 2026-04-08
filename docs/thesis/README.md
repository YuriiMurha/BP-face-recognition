# Thesis Documentation Index

Bachelor Thesis: **Face Recognition Using Surveillance Systems**
Author: Yurii Murha (TUKE)

## Chapters (Recommended Reading Order)

1. **[System Architecture](chapters/system_architecture.md)** — Software design, plugin system, service layer, design decisions
2. **[Detection Methods](chapters/detection_methods.md)** — MediaPipe, MTCNN, Haar Cascade, Dlib HOG comparison and benchmarks
3. **[FaceNet Fine-Tuning](chapters/facenet_finetuning.md)** — Transfer Learning vs Progressive Unfreezing vs Triplet Loss (main study)
4. **[Open-Set vs Closed-Set](chapters/open_vs_closed_set.md)** — Recognition paradigm comparison, trade-offs, use cases
5. **[Results Analysis](chapters/results_analysis.md)** — Per-class accuracy, confusion matrices, class imbalance impact

## Reference Tables

- **[FaceNet Results Tables](tables/facenet_results_tables.md)** — 13 publication-ready tables (Markdown + LaTeX conversion guide)

## Generated Benchmark Data

All in `results/thesis/`:
- `thesis_benchmark_report.md` — Summary tables with LaTeX versions
- `detection_results.json` — Raw detection benchmark data
- `recognition_results.json` — Raw recognition benchmark data (per-class metrics, confusion matrices)
- `facenet_tl_confusion_matrix.png` — Transfer Learning confusion matrix
- `facenet_pu_confusion_matrix.png` — Progressive Unfreezing confusion matrix
- `facenet_tloss_confusion_matrix.png` — Triplet Loss confusion matrix
- `training_curves_comparison.png` — Training curves for all 3 strategies

## Reproducing Benchmarks

```bash
make thesis-benchmark
```

Results are saved to `results/thesis/`.
