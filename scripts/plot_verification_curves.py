"""Plot ROC and DET curves for the three FaceNet fine-tuning strategies.

Reads:
    results/verification_results.json

Writes (1200 x 600 px @ 150 dpi):
    thesis/figures/verification_roc.png
    thesis/figures/verification_det.png

The input file has a top-level `models` list. Each element exposes
`threshold_sweep` -- a list of `{threshold, tpr, fpr}` triples that
defines the ROC curve. AUC and EER are also present and annotated on
the legend.

The DET curve is the same data plotted on log-log axes with miss rate
(1-TPR) against FAR; it spreads the low-error region for visual
comparison between strong systems.

Usage:
    python scripts/plot_verification_curves.py
    python scripts/plot_verification_curves.py --output thesis/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "verification_results.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "thesis" / "figures"

# Match the styling of plot_training_curves.py for visual consistency.
PALETTE = {
    "TL": "#1f77b4",
    "PU": "#2ca02c",
    "TLoss": "#d62728",
    "Triplet": "#d62728",
}


def _colour_for(key: str, name: str) -> str:
    return PALETTE.get(key) or PALETTE.get(name) or "#666666"


def _legend_label(model: dict) -> str:
    name = model.get("name", model.get("key", "model"))
    eer = model.get("EER")
    auc = model.get("AUC")
    if eer is not None and auc is not None:
        return f"{name}  (EER={eer:.3f}, AUC={auc:.3f})"
    return name


def _sweep_points(model: dict) -> tuple[list[float], list[float]]:
    """Return (fpr, tpr) lists from `threshold_sweep`, sorted by fpr."""
    sweep = model.get("threshold_sweep") or []
    pts = sorted(((p["fpr"], p["tpr"]) for p in sweep), key=lambda x: x[0])
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_roc(models: list[dict], out_path: Path) -> None:
    # Slightly wider canvas; legend placed below the axes so long labels
    # like "Progressive Unfreezing (EER=0.090, AUC=0.972)" never clip when
    # \includegraphics scales the PNG to \linewidth in LaTeX.
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
    for m in models:
        fpr, tpr = _sweep_points(m)
        ax.plot(
            fpr,
            tpr,
            "-",
            color=_colour_for(m.get("key", ""), m.get("name", "")),
            linewidth=2,
            label=_legend_label(m),
        )
    ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1, label="Random")
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("True Acceptance Rate (TAR)")
    ax.set_title("ROC curves: 5,000 positive vs 5,000 negative pairs")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        framealpha=0.95,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_det(models: list[dict], out_path: Path) -> None:
    """DET: miss rate (1-TPR) vs FAR on log-log axes.

    A small floor is applied to avoid log(0). Points with FAR=0 or
    miss=0 are clipped to 1e-4 for plotting purposes.
    """
    floor = 1e-4
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
    for m in models:
        fpr, tpr = _sweep_points(m)
        miss = [max(1.0 - t, floor) for t in tpr]
        far = [max(f, floor) for f in fpr]
        ax.plot(
            far,
            miss,
            "-",
            color=_colour_for(m.get("key", ""), m.get("name", "")),
            linewidth=2,
            label=_legend_label(m),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("False Acceptance Rate (log)")
    ax.set_ylabel("Miss Rate = 1 - TAR (log)")
    ax.set_title("DET curves: log-log scaling of the low-error region")
    ax.set_xlim(floor, 1.0)
    ax.set_ylim(floor, 1.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        framealpha=0.95,
        fontsize=9,
    )
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
        help=f"verification_results.json (default: {DEFAULT_INPUT.relative_to(PROJECT_ROOT)})",
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

    data = json.loads(args.input.read_text(encoding="utf-8"))
    models = data.get("models") or []
    if not models:
        print("ERROR: no `models` array in input.", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    plot_roc(models, args.output / "verification_roc.png")
    plot_det(models, args.output / "verification_det.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
