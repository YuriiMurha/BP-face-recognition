"""Model Comparison Framework for FaceNet Fine-Tuning Study

Evaluates and compares multiple fine-tuned models across various metrics.
Generates publication-ready visualizations and reports.

Usage:
    python compare_models.py --models facenet_transfer facenet_full facenet_triplet baseline
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparisonFramework:
    """
    Comprehensive framework for comparing face recognition models.

    Generates:
    - Accuracy comparison charts
    - Training curves comparison
    - t-SNE embedding visualizations
    - Similarity distribution analysis
    - LaTeX tables for papers
    - Comprehensive comparison report
    """

    def __init__(self, output_dir: str = "results/comparison"):
        """
        Initialize comparison framework.

        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

        self.models_data = {}

        logger.info(f"Comparison framework initialized. Output: {self.output_dir}")

    def load_model_results(self, model_name: str, results_path: str):
        """
        Load training results for a model.

        Args:
            model_name: Name identifier for the model
            results_path: Path to results JSON file
        """
        results_path = Path(results_path)

        if not results_path.exists():
            logger.warning(f"Results file not found: {results_path}")
            return

        with open(results_path, "r") as f:
            data = json.load(f)

        self.models_data[model_name] = data
        logger.info(f"Loaded results for {model_name}")

    def compare_accuracy(self) -> Dict[str, Any]:
        """
        Compare test and validation accuracy across models.

        Returns:
            Dictionary with comparison metrics
        """
        comparison = {}

        for model_name, data in self.models_data.items():
            comparison[model_name] = {
                "test_accuracy": data.get("results", {}).get("test_accuracy"),
                "val_accuracy": data.get("results", {}).get("final_val_accuracy"),
                "best_val_accuracy": data.get("results", {}).get("best_val_accuracy"),
            }

        return comparison

    def plot_training_curves(self):
        """
        Generate training curves comparison plot.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        for model_name, data in self.models_data.items():
            history = data.get("history", {})
            if not history:
                continue

            epochs = range(1, len(history.get("accuracy", [])) + 1)

            # Accuracy
            axes[0, 0].plot(
                epochs, history.get("accuracy", []), label=model_name, marker="o"
            )
            axes[0, 0].set_title("Training Accuracy")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Accuracy")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Validation Accuracy
            if "val_accuracy" in history:
                axes[0, 1].plot(
                    epochs, history["val_accuracy"], label=model_name, marker="s"
                )
                axes[0, 1].set_title("Validation Accuracy")
                axes[0, 1].set_xlabel("Epoch")
                axes[0, 1].set_ylabel("Accuracy")
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

            # Loss
            axes[1, 0].plot(
                epochs, history.get("loss", []), label=model_name, marker="o"
            )
            axes[1, 0].set_title("Training Loss")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Validation Loss
            if "val_loss" in history:
                axes[1, 1].plot(
                    epochs, history["val_loss"], label=model_name, marker="s"
                )
                axes[1, 1].set_title("Validation Loss")
                axes[1, 1].set_xlabel("Epoch")
                axes[1, 1].set_ylabel("Loss")
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "figures" / "training_curves_comparison.png", dpi=300
        )
        plt.close()

        logger.info("Saved training curves comparison")

    def plot_accuracy_comparison(self):
        """
        Generate bar chart comparing final accuracies.
        """
        comparison = self.compare_accuracy()

        models = list(comparison.keys())
        test_accs = [comparison[m].get("test_accuracy", 0) for m in models]
        val_accs = [comparison[m].get("best_val_accuracy", 0) for m in models]

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(
            x - width / 2, test_accs, width, label="Test Accuracy", color="skyblue"
        )
        bars2 = ax.bar(
            x + width / 2,
            val_accs,
            width,
            label="Best Val Accuracy",
            color="lightcoral",
        )

        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "accuracy_comparison.png", dpi=300)
        plt.close()

        logger.info("Saved accuracy comparison chart")

    def generate_latex_table(self):
        """
        Generate LaTeX table for academic paper.
        """
        comparison = self.compare_accuracy()

        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{FaceNet Fine-Tuning Comparison Results}")
        latex.append("\\label{tab:facenet_comparison}")
        latex.append("\\begin{tabular}{lcccc}")
        latex.append("\\toprule")
        latex.append(
            "\\textbf{Model} & \\textbf{Test Acc} & \\textbf{Val Acc} & \\textbf{Epochs} & \\textbf{Training Time} \\\\"
        )
        latex.append("\\midrule")

        for model_name, data in self.models_data.items():
            results = data.get("results", {})
            test_acc = results.get("test_accuracy", "N/A")
            val_acc = results.get("best_val_accuracy", "N/A")
            epochs = len(data.get("history", {}).get("accuracy", []))

            if isinstance(test_acc, float):
                test_acc = f"{test_acc:.4f}"
            if isinstance(val_acc, float):
                val_acc = f"{val_acc:.4f}"

            latex.append(f"{model_name} & {test_acc} & {val_acc} & {epochs} & TBD \\\\")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        table_path = self.output_dir / "tables" / "comparison_table.tex"
        with open(table_path, "w") as f:
            f.write("\n".join(latex))

        logger.info(f"Saved LaTeX table: {table_path}")
        return "\n".join(latex)

    def generate_comparison_report(self):
        """
        Generate comprehensive comparison report.
        """
        report = []
        report.append("# FaceNet Fine-Tuning: Comprehensive Model Comparison")
        report.append(f"\nGenerated: {datetime.now().isoformat()}")
        report.append("\n## Summary")

        comparison = self.compare_accuracy()

        # Rank models by test accuracy
        ranked = sorted(
            comparison.items(),
            key=lambda x: x[1].get("test_accuracy", 0) or 0,
            reverse=True,
        )

        report.append(f"\n### Model Ranking (by Test Accuracy):")
        for i, (model, metrics) in enumerate(ranked, 1):
            test_acc = metrics.get("test_accuracy", "N/A")
            val_acc = metrics.get("best_val_accuracy", "N/A")
            report.append(f"{i}. **{model}**: Test={test_acc}, Val={val_acc}")

        report.append("\n## Detailed Results")

        for model_name, data in self.models_data.items():
            report.append(f"\n### {model_name}")
            report.append(
                f"- **Test Accuracy**: {data.get('results', {}).get('test_accuracy', 'N/A')}"
            )
            report.append(
                f"- **Best Val Accuracy**: {data.get('results', {}).get('best_val_accuracy', 'N/A')}"
            )
            report.append(
                f"- **Epochs Trained**: {len(data.get('history', {}).get('accuracy', []))}"
            )
            report.append(
                f"- **Architecture**: {data.get('architecture', {}).get('base_model', 'N/A')}"
            )

        report.append("\n## Files Generated")
        report.append("- `figures/training_curves_comparison.png` - Training curves")
        report.append("- `figures/accuracy_comparison.png` - Accuracy bar chart")
        report.append("- `tables/comparison_table.tex` - LaTeX table")
        report.append("- `comparison_report.md` - This report")

        report_path = self.output_dir / "reports" / "comparison_report.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report))

        logger.info(f"Saved comparison report: {report_path}")

    def run_full_comparison(self):
        """
        Run complete comparison pipeline.
        """
        logger.info("Starting full comparison...")

        if len(self.models_data) == 0:
            logger.warning("No model data loaded. Cannot perform comparison.")
            return

        # Generate all comparisons
        self.plot_training_curves()
        self.plot_accuracy_comparison()
        self.generate_latex_table()
        self.generate_comparison_report()

        logger.info("Comparison complete!")
        logger.info(f"Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare FaceNet Fine-Tuning Models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["facenet_transfer", "facenet_full", "facenet_triplet"],
        help="Model names to compare",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="src/bp_face_recognition/models/finetuned",
        help="Directory containing model results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison",
        help="Output directory for comparison results",
    )

    args = parser.parse_args()

    # Initialize framework
    framework = ModelComparisonFramework(output_dir=args.output_dir)

    # Load results for each model
    results_dir = Path(args.results_dir)

    for model_name in args.models:
        # Try to find results file
        possible_paths = [
            results_dir / f"{model_name}_report.json",
            results_dir / f"{model_name}_history.json",
        ]

        for path in possible_paths:
            if path.exists():
                framework.load_model_results(model_name, str(path))
                break

    # Run comparison
    framework.run_full_comparison()


if __name__ == "__main__":
    main()
