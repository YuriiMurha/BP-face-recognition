"""Preliminary Comparison Visualization

Generates visualizations comparing FaceNet fine-tuning approaches.
Can work with partial data (as experiments complete).

Usage:
    python visualize_preliminary_results.py
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


def load_results(model_dir: str, model_name: str) -> dict:
    """Load results for a model if available."""
    report_path = Path(model_dir) / f"{model_name}_report.json"
    history_path = Path(model_dir) / f"{model_name}_history.json"

    results = {"name": model_name, "available": False, "report": None, "history": None}

    if report_path.exists():
        with open(report_path, "r") as f:
            results["report"] = json.load(f)
        results["available"] = True

    if history_path.exists():
        with open(history_path, "r") as f:
            results["history"] = json.load(f)

    return results


def plot_accuracy_comparison(models_data: dict, output_dir: Path):
    """
    Generate bar chart comparing model accuracies.
    """
    available_models = {k: v for k, v in models_data.items() if v["available"]}

    if not available_models:
        print("No model data available yet!")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = []
    test_accs = []
    val_accs = []
    colors = []

    color_map = {
        "facenet_transfer": "#3498db",  # Blue
        "facenet_progressive": "#2ecc71",  # Green
        "facenet_triplet": "#e74c3c",  # Red
    }

    for model_name, data in available_models.items():
        report = data["report"]
        results = report.get("results", {})

        model_names.append(model_name.replace("facenet_", "").replace("_", " ").title())
        test_accs.append(results.get("test_accuracy", 0) * 100)
        val_accs.append(
            results.get("best_val_accuracy", results.get("final_val_accuracy", 0)) * 100
        )
        colors.append(color_map.get(model_name, "#95a5a6"))

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, test_accs, width, label="Test Accuracy", alpha=0.8)
    bars2 = ax.bar(
        x + width / 2, val_accs, width, label="Validation Accuracy", alpha=0.8
    )

    # Color the bars
    for bar, color in zip(bars1, colors):
        bar.set_color(color)
        bar.set_edgecolor("black")
        bar.set_linewidth(1.5)

    for bar, color in zip(bars2, colors):
        bar.set_color(color)
        bar.set_alpha(0.5)
        bar.set_edgecolor("black")
        bar.set_linewidth(1.5)

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "FaceNet Fine-Tuning: Accuracy Comparison\n(Preliminary Results)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 105])

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.savefig(
        output_dir / "accuracy_comparison_preliminary.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"[OK] Saved: accuracy_comparison_preliminary.png")


def plot_training_curves(models_data: dict, output_dir: Path):
    """
    Generate training curves comparison.
    """
    available_models = {
        k: v for k, v in models_data.items() if v["available"] and v["history"]
    }

    if not available_models:
        print("No training history available yet!")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    color_map = {
        "facenet_transfer": "#3498db",
        "facenet_progressive": "#2ecc71",
        "facenet_triplet": "#e74c3c",
    }

    for model_name, data in available_models.items():
        history = data["history"]
        color = color_map.get(model_name, "#95a5a6")
        label = model_name.replace("facenet_", "").replace("_", " ").title()

        epochs = range(1, len(history.get("accuracy", [])) + 1)

        # Training Accuracy
        if "accuracy" in history:
            axes[0, 0].plot(
                epochs,
                history["accuracy"],
                label=label,
                color=color,
                linewidth=2,
                marker="o",
                markersize=4,
            )

        # Validation Accuracy
        if "val_accuracy" in history:
            axes[0, 1].plot(
                epochs,
                history["val_accuracy"],
                label=label,
                color=color,
                linewidth=2,
                marker="s",
                markersize=4,
            )

        # Training Loss
        if "loss" in history:
            axes[1, 0].plot(
                epochs,
                history["loss"],
                label=label,
                color=color,
                linewidth=2,
                marker="o",
                markersize=4,
            )

        # Validation Loss
        if "val_loss" in history:
            axes[1, 1].plot(
                epochs,
                history["val_loss"],
                label=label,
                color=color,
                linewidth=2,
                marker="s",
                markersize=4,
            )

    # Configure subplots
    axes[0, 0].set_title("Training Accuracy", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Validation Accuracy", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Training Loss", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Validation Loss", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        "FaceNet Fine-Tuning: Training Curves Comparison\n(Preliminary Results)",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "training_curves_preliminary.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"[OK] Saved: training_curves_preliminary.png")


def plot_summary_table(models_data: dict, output_dir: Path):
    """
    Generate a summary table visualization.
    """
    available_models = {k: v for k, v in models_data.items() if v["available"]}

    if not available_models:
        print("No model data available for table!")
        return

    # Prepare data
    data_rows = []

    for model_name, data in available_models.items():
        report = data["report"]
        results = report.get("results", {})
        config = report.get("training_config", {})

        row = {
            "Model": model_name.replace("facenet_", "").replace("_", " ").title(),
            "Test Acc (%)": f"{results.get('test_accuracy', 0) * 100:.2f}",
            "Val Acc (%)": f"{results.get('best_val_accuracy', results.get('final_val_accuracy', 0)) * 100:.2f}",
            "Epochs": str(config.get("epochs_trained", "N/A")),
            "Status": "[OK] Complete",
        }
        data_rows.append(row)

    # Add placeholder rows for incomplete models
    for model_name in ["facenet_progressive", "facenet_triplet"]:
        if model_name not in available_models:
            data_rows.append(
                {
                    "Model": model_name.replace("facenet_", "")
                    .replace("_", " ")
                    .title(),
                    "Test Acc (%)": "Pending",
                    "Val Acc (%)": "Pending",
                    "Epochs": "N/A",
                    "Status": "[WAIT] Running",
                }
            )

    # Create table figure
    fig, ax = plt.subplots(figsize=(12, len(data_rows) * 0.8 + 2))
    ax.axis("tight")
    ax.axis("off")

    # Table data
    table_data = [
        [
            row[col]
            for col in ["Model", "Test Acc (%)", "Val Acc (%)", "Epochs", "Status"]
        ]
        for row in data_rows
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=["Model", "Test Accuracy", "Validation Accuracy", "Epochs", "Status"],
        cellLoc="center",
        loc="center",
        colColours=["#3498db"] * 5,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header
    for i in range(5):
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    # Color status cells
    for i, row in enumerate(data_rows, 1):
        if row["Status"] == "[OK] Complete":
            table[(i, 4)].set_facecolor("#d5f5e3")
        else:
            table[(i, 4)].set_facecolor("#fef9e7")

    plt.title(
        "FaceNet Fine-Tuning: Results Summary\n(Preliminary)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.savefig(output_dir / "results_summary_table.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved: results_summary_table.png")


def generate_latex_table(models_data: dict, output_dir: Path):
    """
    Generate LaTeX table code.
    """
    available_models = {k: v for k, v in models_data.items() if v["available"]}

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(
        "\\caption{FaceNet Fine-Tuning Strategy Comparison (Preliminary Results)}"
    )
    latex.append("\\label{tab:preliminary_comparison}")
    latex.append("\\begin{tabular}{@{}lcccc@{}}")
    latex.append("\\toprule")
    latex.append(
        "\\textbf{Approach} & \\textbf{Test Acc.} & \\textbf{Val Acc.} & \\textbf{Epochs} & \\textbf{Status} \\\\"
    )
    latex.append("\\midrule")

    for model_name, data in available_models.items():
        report = data["report"]
        results = report.get("results", {})
        config = report.get("training_config", {})

        name = model_name.replace("facenet_", "").replace("_", " ").title()
        test_acc = f"{results.get('test_accuracy', 0):.4f}"
        val_acc = f"{results.get('best_val_accuracy', results.get('final_val_accuracy', 0)):.4f}"
        epochs = str(config.get("epochs_trained", "N/A"))

        latex.append(f"{name} & {test_acc} & {val_acc} & {epochs} & \\checkmark \\\\")

    # Add pending rows
    for model_name in ["facenet_progressive", "facenet_triplet"]:
        if model_name not in available_models:
            name = model_name.replace("facenet_", "").replace("_", " ").title()
            latex.append(f"{name} & TBD & TBD & TBD & $\\dagger$ \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")
    latex.append("\\noindent $\\dagger$ Training in progress")

    # Save to file
    latex_path = output_dir / "comparison_table.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex))

    print(f"[OK] Saved: comparison_table.tex")
    print("\nLaTeX Table Code:")
    print("=" * 60)
    print("\n".join(latex))
    print("=" * 60)


def generate_summary_report(models_data: dict, output_dir: Path):
    """
    Generate a summary report.
    """
    available_models = {k: v for k, v in models_data.items() if v["available"]}

    report = []
    report.append("# FaceNet Fine-Tuning: Preliminary Results Summary")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## Overview")
    report.append(f"\nCompleted Models: {len(available_models)}/3")

    if available_models:
        report.append("\n## Completed Results")

        for model_name, data in available_models.items():
            report.append(
                f"\n### {model_name.replace('facenet_', '').replace('_', ' ').title()}"
            )

            report_data = data["report"]
            results = report_data.get("results", {})
            config = report_data.get("training_config", {})

            report.append(f"- **Test Accuracy**: {results.get('test_accuracy', 'N/A')}")
            report.append(
                f"- **Validation Accuracy**: {results.get('best_val_accuracy', results.get('final_val_accuracy', 'N/A'))}"
            )
            report.append(f"- **Epochs**: {config.get('epochs_trained', 'N/A')}")
            report.append(f"- **Training Time**: {config.get('training_time', 'N/A')}")

    report.append("\n## Pending")
    pending = [
        m
        for m in ["facenet_progressive", "facenet_triplet"]
        if m not in available_models
    ]
    if pending:
        report.append(f"\nModels still training or pending:")
        for model in pending:
            report.append(
                f"- {model.replace('facenet_', '').replace('_', ' ').title()}"
            )

    report.append("\n## Files Generated")
    report.append("- `accuracy_comparison_preliminary.png` - Accuracy bar chart")
    report.append("- `training_curves_preliminary.png` - Training curves")
    report.append("- `results_summary_table.png` - Summary table")
    report.append("- `comparison_table.tex` - LaTeX table code")

    report_path = output_dir / "preliminary_summary.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))

    print(f"[OK] Saved: preliminary_summary.md")


def main():
    print("=" * 60)
    print("FACE NET FINE-TUNING: PRELIMINARY VISUALIZATIONS")
    print("=" * 60)
    print()

    # Setup paths
    model_dir = Path("src/bp_face_recognition/models/finetuned")
    output_dir = Path("results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model directory: {model_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load data for all models
    models = ["facenet_transfer", "facenet_progressive", "facenet_triplet"]
    models_data = {}

    print("Loading model data...")
    for model in models:
        data = load_results(model_dir, model)
        models_data[model] = data
        status = "[OK] Available" if data["available"] else "[WAIT] Pending"
        print(f"  {model}: {status}")

    print()

    # Generate visualizations
    print("Generating visualizations...")
    print("-" * 60)

    plot_accuracy_comparison(models_data, output_dir)
    plot_training_curves(models_data, output_dir)
    plot_summary_table(models_data, output_dir)
    generate_latex_table(models_data, output_dir)
    generate_summary_report(models_data, output_dir)

    print("-" * 60)
    print()
    print("[OK] All visualizations generated successfully!")
    print(f"\nOutput location: {output_dir}")
    print()
    print("Generated files:")
    for f in output_dir.iterdir():
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
