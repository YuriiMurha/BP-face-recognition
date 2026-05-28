"""Generate confusion-matrix heatmap for the Progressive Unfreezing model.

Reads:
    results/recognition_results.json

Writes (1200x1000 px, 150 dpi):
    thesis/figures/confusion_matrix_pu.png

The PU confusion matrix is the most informative of the three because it has
the highest classification accuracy (99.15%) and the off-diagonal entries
that remain are the most illustrative misclassifications. We also render a
TL companion if the user wants to embed a comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "recognition_results.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "thesis" / "figures"


def _find_model(records: list[dict], name_substr: str) -> dict | None:
    for m in records:
        if name_substr.lower() in m.get("name", "").lower():
            if m.get("confusion_matrix"):
                return m
    return None


def plot_confusion(model: dict, out_path: Path) -> None:
    cm = np.array(model["confusion_matrix"], dtype=float)
    n = cm.shape[0]

    # Per-row normalisation gives the diagonal as recall, off-diagonals as
    # the fraction of true-class examples mistaken for another class. The
    # normalisation makes the magnitude scale comparable across classes
    # with different support sizes.
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums > 0)

    class_names = model.get("class_names") or [str(i) for i in range(n)]
    if len(class_names) != n:
        class_names = [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    # Overlay the raw counts on each cell (white text on dark cells, black
    # on light) so the figure conveys both magnitude and proportion.
    for i in range(n):
        for j in range(n):
            val = int(cm[i, j])
            if val == 0:
                continue
            colour = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j, i, str(val), ha="center", va="center", fontsize=8, color=colour
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix: {model['name']} ({model.get('accuracy_percent', 0):.2f}%)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Per-row recall", rotation=270, labelpad=15)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"recognition_results.json (default: {DEFAULT_INPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found.", file=sys.stderr)
        return 2

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        print("ERROR: recognition_results.json is not a list at top level.", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)

    # Progressive Unfreezing is the headline model — generate that one.
    pu_model = _find_model(records, "Progressive Unfreezing") or _find_model(records, " PU")
    if pu_model:
        plot_confusion(pu_model, args.output / "confusion_matrix_pu.png")
    else:
        print("WARN: Progressive Unfreezing model not found in results", file=sys.stderr)

    # Also Transfer Learning for the comparison view.
    tl_model = _find_model(records, "Transfer Learning") or _find_model(records, " TL")
    if tl_model:
        plot_confusion(tl_model, args.output / "confusion_matrix_tl.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
