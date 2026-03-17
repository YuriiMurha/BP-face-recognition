#!/bin/bash
# Run all FaceNet evaluations
# This script evaluates all 3 models and generates comparison report

set -e

echo "=================================="
echo "FaceNet Model Evaluation Suite"
echo "=================================="
echo ""

# Create output directory
mkdir -p results/evaluation

echo "Note: Using embedding-based evaluation (KNN classifier)"
echo "This provides fair comparison across all models"
echo ""

# Evaluate Progressive Unfreezing (PU) - Best model
echo "1/3: Evaluating Progressive Unfreezing (PU) - 99.15% target..."
uv run python src/bp_face_recognition/evaluation/evaluate_simple.py \
    --model src/bp_face_recognition/models/finetuned/facenet_progressive_v1.0.keras \
    --output results/evaluation/facenet_pu_eval.json \
    --quiet

# Evaluate Transfer Learning (TL)
echo "2/3: Evaluating Transfer Learning (TL) - 92.84% target..."
uv run python src/bp_face_recognition/evaluation/evaluate_simple.py \
    --model src/bp_face_recognition/models/finetuned/facenet_transfer_v1.0.keras \
    --output results/evaluation/facenet_tl_eval.json \
    --quiet

# Evaluate Triplet Loss (TLoss)
echo "3/3: Evaluating Triplet Loss (TLoss) - 94.63% target..."
uv run python src/bp_face_recognition/evaluation/evaluate_simple.py \
    --model src/bp_face_recognition/models/finetuned/facenet_triplet_best.keras \
    --output results/evaluation/facenet_tloss_eval.json \
    --quiet

# Generate comparison report
echo ""
echo "Generating comparison report..."
uv run python src/bp_face_recognition/evaluation/generate_comparison_report.py

echo ""
echo "=================================="
echo "Evaluation Complete!"
echo "=================================="
echo ""
echo "Results:"
echo "  - Individual results: results/evaluation/*.json"
echo "  - Comparison report: results/evaluation/comparison_report.md"
echo ""
