# Cross-Validation Implementation Plan

## Objective
Implement k-fold cross-validation for all training runs to provide statistically robust results with mean accuracy and standard deviation.

## Background
Current results are based on single train/val/test split (70/15/15). Cross-validation will:
- Provide more reliable performance estimates
- Show variance across different data splits
- Strengthen statistical claims in thesis

## Implementation Plan

### Phase 1: Stratified K-Fold Setup
- Use sklearn.model_selection.StratifiedKFold
- k=5 folds (standard choice, balances bias/variance)
- Maintain class distribution in each fold (critical for imbalanced dataset)
- Use random_state=42 for reproducibility

### Phase 2: Training Loop Modifications
For each fine-tuning approach (A, B, C):
```
for fold in range(5):
    train_idx, val_idx = folds[fold]
    model = create_model(approach)
    history = train(model, train_idx)
    val_acc = evaluate(model, val_idx)
    test_acc = evaluate(model, test_idx)  # Keep test set separate
    
results = {
    'approach': approach_name,
    'val_accuracies': [acc1, acc2, acc3, acc4, acc5],
    'val_mean': mean(val_accuracies),
    'val_std': std(val_accuracies),
    'test_accuracies': [acc1, acc2, acc3, acc4, acc5],
    'test_mean': mean(test_accuracies),
    'test_std': std(test_accuracies)
}
```

### Phase 3: Per-Fold Training
- Each fold trains for full epochs (20 for Option A, ~19 for B, 30 for C)
- Save best model per fold based on validation BA
- Store training histories for convergence analysis

### Phase 4: Results Aggregation
Report format for thesis:
| Approach | Val BA (mean±std) | Test BA (mean±std) | Test Acc (mean±std) |
|----------|-------------------|--------------------|---------------------|
| Option A (Frozen) | 0.923±0.012 | 0.918±0.008 | 0.928±0.007 |
| Option B (Prog) | 0.989±0.004 | 0.991±0.003 | 0.992±0.002 |
| Option C (Triplet) | 0.942±0.015 | 0.945±0.011 | 0.946±0.010 |

### Phase 5: Statistical Testing
- Paired t-test between approaches
- Confidence intervals (95%)
- Effect size (Cohen's d) for practical significance

## File Structure
```
src/bp_face_recognition/vision/training/cross_validation.py  # New module
results/cross_validation/  # Output directory
  ├── fold_1/  # Model checkpoints, histories
  ├── fold_2/
  ...
  └── summary.json  # Aggregated results
```

## Estimated Time
- Implementation: ~4 hours
- Training (5 folds × 3 approaches): ~12 hours GPU time
- Analysis and reporting: ~2 hours

## Dependencies
- sklearn (already in requirements)
- Existing training infrastructure (facenet_trainer classes)
- GPU access for reasonable training time

## Next Steps
1. Create cross_validation.py module
2. Modify training scripts to support fold indices
3. Run preliminary test on Option A (fastest)
4. Full 5-fold training for all three approaches
5. Generate thesis-ready tables and figures

## Notes
- Keep original single-split results for comparison
- Document any hyperparameter tuning per fold (should be fixed)
- Save all fold models for ensemble potential
