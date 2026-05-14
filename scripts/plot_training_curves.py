"""Generate publication-quality per-approach training-curve figures.

Reads:
    src/bp_face_recognition/models/finetuned/facenet_transfer_history.json
    src/bp_face_recognition/models/finetuned/facenet_progressive_history.json
    src/bp_face_recognition/models/finetuned/facenet_triplet_history.json

Writes (1200 x 500 px @ 150 dpi):
    thesis/figures/facenet_transfer_curves.png
    thesis/figures/facenet_progressive_curves.png
    thesis/figures/facenet_triplet_curves.png

Transfer Learning was early-stopped at epoch 2 — annotated on its figure.
Progressive Unfreezing has phase boundaries at epochs 5, 10, 15 — drawn as vlines.
Triplet Loss has loss only (no per-epoch accuracy metric in the triplet
training objective).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINETUNED_DIR = PROJECT_ROOT / "src" / "bp_face_recognition" / "models" / "finetuned"
FIGURES_DIR = PROJECT_ROOT / "thesis" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_history(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [warn] missing {path.name}")
        return None
    with open(path) as f:
        return json.load(f)


def plot_two_panel(
    history: dict,
    *,
    title: str,
    out_path: Path,
    annotate_early_stop: bool = False,
    phase_boundaries: list[int] | None = None,
):
    """Plot (loss, accuracy) side-by-side for a classifier-style history."""
    epochs = list(range(1, len(history["loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Loss panel
    axes[0].plot(epochs, history["loss"], "-o", label="Train", color="#1f77b4", linewidth=2, markersize=5)
    axes[0].plot(epochs, history["val_loss"], "-s", label="Validation", color="#ff7f0e", linewidth=2, markersize=5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (categorical cross-entropy)")
    axes[0].set_title(f"{title} — Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    # Accuracy panel
    axes[1].plot(epochs, [a * 100 for a in history["accuracy"]], "-o", label="Train", color="#1f77b4", linewidth=2, markersize=5)
    axes[1].plot(epochs, [a * 100 for a in history["val_accuracy"]], "-s", label="Validation", color="#ff7f0e", linewidth=2, markersize=5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    if annotate_early_stop:
        for ax in axes:
            ax.axvline(epochs[-1], color="red", linestyle="--", alpha=0.6, label="Early stop")
            ax.annotate(
                "Early stopping\ntriggered",
                xy=(epochs[-1], ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                xytext=(-90, 30),
                textcoords="offset points",
                fontsize=9,
                color="red",
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.6),
            )

    if phase_boundaries:
        # Draw phase boundary lines on both panels (visual cue), but only label
        # them on the loss panel to avoid collision with the accuracy panel's
        # legend in the top-left.
        for ax in axes:
            for x in phase_boundaries:
                if x < len(epochs):
                    ax.axvline(x + 0.5, color="gray", linestyle=":", alpha=0.5)
        loss_ax = axes[0]
        ymin, ymax = loss_ax.get_ylim()
        label_y = ymax - 0.04 * (ymax - ymin)
        for x_mid, label in [(3, "Phase 1"), (8, "Phase 2"), (13, "Phase 3"), (17, "Phase 4")]:
            loss_ax.text(x_mid, label_y, label, ha="center", va="top", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_loss_only(history: dict, *, title: str, out_path: Path):
    """Plot just train and val loss (for triplet loss training)."""
    epochs = list(range(1, len(history["loss"]) + 1))

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=150)
    ax.plot(epochs, history["loss"], "-o", label="Train", color="#1f77b4", linewidth=2, markersize=5)
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], "-s", label="Validation", color="#ff7f0e", linewidth=2, markersize=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Triplet loss")
    ax.set_title(f"{title} — Training (lower loss is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.text(
        0.99, 0.95,
        "No per-epoch accuracy: triplet loss does not produce class probabilities.",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color="dimgray",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgray"),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    print("Generating training-curve figures...")

    tl = load_history(FINETUNED_DIR / "facenet_transfer_history.json")
    if tl is not None:
        plot_two_panel(
            tl,
            title="Transfer Learning",
            out_path=FIGURES_DIR / "facenet_transfer_curves.png",
            annotate_early_stop=True,
        )

    pu = load_history(FINETUNED_DIR / "facenet_progressive_history.json")
    if pu is not None:
        plot_two_panel(
            pu,
            title="Progressive Unfreezing",
            out_path=FIGURES_DIR / "facenet_progressive_curves.png",
            phase_boundaries=[5, 10, 15],
        )

    tloss = load_history(FINETUNED_DIR / "facenet_triplet_history.json")
    if tloss is not None:
        plot_loss_only(
            tloss,
            title="Triplet Loss",
            out_path=FIGURES_DIR / "facenet_triplet_curves.png",
        )
    else:
        print("  [info] facenet_triplet_history.json not present yet — skipping TLoss plot.")

    print("Done.")


if __name__ == "__main__":
    main()
