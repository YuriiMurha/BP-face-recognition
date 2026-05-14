"""Open-set verification protocol evaluation for the 3 FaceNet fine-tuning approaches.

For each of Transfer Learning (TL), Progressive Unfreezing (PU), and Triplet
Loss (TLoss), this script:

1. Extracts L2-normalized 512D embeddings from the FaceNet backbone for all
   1,062 test samples.
2. Samples 5,000 positive pairs (same identity) and 5,000 negative pairs
   (different identity) using a fixed RNG seed for reproducibility.
3. Computes cosine similarity for each pair.
4. Sweeps a similarity threshold from -1 to +1 in 500 steps and computes
   TAR (= TPR) and FAR (= FPR) at every threshold.
5. Reports the equal error rate (EER), TAR at FAR = 1%, TAR at FAR = 0.1%,
   and ROC AUC, with linear interpolation between adjacent threshold steps.
6. Saves the per-threshold sweep, the summary metrics, and a markdown table to
   ``results/`` and renders ROC + DET curves to ``thesis/figures/``.

This converts the qualitative "TLoss is best for open-set" claim in §7.4.2 of
the thesis into hard verification numbers reported at standard biometric
operating points.

Output:
    results/verification_results.json
    results/verification_report.md
    thesis/figures/verification_roc.png
    thesis/figures/verification_det.png
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

# Force CPU inference before TF import (another job has the GPU).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow import keras

from bp_face_recognition.config.settings import settings
from bp_face_recognition.utils.facenet_loader import load_finetuned_facenet_robust
from bp_face_recognition.vision.training.finetune.dataset_loader import (
    create_combined_dataset,
)


PROJECT_ROOT = settings.ROOT_DIR
FINETUNED_DIR = PROJECT_ROOT / "src" / "bp_face_recognition" / "models" / "finetuned"
OUTPUT_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "thesis" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


MODELS = [
    {
        "key": "TL",
        "name": "Transfer Learning",
        "path": FINETUNED_DIR / "facenet_transfer_v1.0.keras",
        "kind": "classifier",
    },
    {
        "key": "PU",
        "name": "Progressive Unfreezing",
        "path": FINETUNED_DIR / "facenet_progressive_v1.0.keras",
        "kind": "classifier",
    },
    {
        "key": "TLoss",
        "name": "Triplet Loss",
        "path_weights": FINETUNED_DIR / "facenet_triplet_v1.0.weights.h5",
        "path_best_weights": FINETUNED_DIR / "facenet_triplet_best.weights.h5",
        "kind": "triplet",
    },
]


# ---------------------------------------------------------------------------
# Embedder construction (mirrors embedding_quality.py but standalone)
# ---------------------------------------------------------------------------


def build_classifier_embedder(model_path: Path) -> keras.Model:
    """Return the inner FaceNet base (Input → 512D) from a TL/PU classifier."""
    classifier = load_finetuned_facenet_robust(str(model_path))
    return classifier.layers[1]


def build_triplet_embedder(path_weights: Path, path_best_weights: Path) -> keras.Model:
    """Rebuild FaceNet backbone and load the triplet-trained weights into it."""
    from keras_facenet import FaceNet

    chosen = path_weights if path_weights.exists() else path_best_weights
    if not chosen.exists():
        raise FileNotFoundError(
            f"No TLoss weights found. Expected {path_weights} or {path_best_weights}."
        )
    facenet = FaceNet()
    base_model = facenet.model
    base_model.load_weights(str(chosen))
    return base_model


# ---------------------------------------------------------------------------
# Test set + embeddings
# ---------------------------------------------------------------------------


def collect_test_set() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    _, _, test_ds, info = create_combined_dataset(batch_size=32, augmentation=False)
    images_list, labels_list = [], []
    for imgs, lbls in test_ds:
        images_list.append(imgs.numpy())
        labels_list.append(np.argmax(lbls.numpy(), axis=1))
    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return images, labels, info["class_names"]


def extract_l2_embeddings(
    embedder: keras.Model, images: np.ndarray, batch_size: int = 32
) -> np.ndarray:
    embs = embedder.predict(images, batch_size=batch_size, verbose=0)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return embs / norms


# ---------------------------------------------------------------------------
# Pair sampling
# ---------------------------------------------------------------------------


def sample_pairs(
    labels: np.ndarray,
    num_pos: int = 5000,
    num_neg: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample positive and negative index pairs without replacement.

    Returns:
        pos_pairs: (M, 2) array of indices with labels[i] == labels[j]
        neg_pairs: (N, 2) array of indices with labels[i] != labels[j]
    """
    rng = np.random.default_rng(seed)
    n = len(labels)

    # Group indices by class.
    by_class: Dict[int, np.ndarray] = {}
    for cls in np.unique(labels):
        by_class[int(cls)] = np.where(labels == cls)[0]

    # ----- POSITIVE PAIRS -----
    # Build the complete pool of (i, j) with i<j and same class. If the pool is
    # smaller than num_pos we use the entire pool (no replacement).
    pos_pool: List[Tuple[int, int]] = []
    for cls, idxs in by_class.items():
        if len(idxs) < 2:
            continue
        # All unordered pairs within this class.
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pos_pool.append((int(idxs[a]), int(idxs[b])))

    pos_pool_arr = np.array(pos_pool, dtype=np.int64)
    if len(pos_pool_arr) >= num_pos:
        sel = rng.choice(len(pos_pool_arr), size=num_pos, replace=False)
        pos_pairs = pos_pool_arr[sel]
    else:
        pos_pairs = pos_pool_arr  # use all available

    # ----- NEGATIVE PAIRS -----
    # The cross-class pool is enormous (~5e5+), so sample by rejection rather
    # than materialize it. Sample (i, j) with i != j, reject if same class or
    # already chosen.
    neg_set = set()
    neg_pairs_list: List[Tuple[int, int]] = []
    attempts = 0
    max_attempts = num_neg * 200
    while len(neg_pairs_list) < num_neg and attempts < max_attempts:
        # Draw a batch to amortize call overhead.
        batch = max(1, num_neg - len(neg_pairs_list)) * 4
        i_arr = rng.integers(0, n, size=batch)
        j_arr = rng.integers(0, n, size=batch)
        for i, j in zip(i_arr, j_arr):
            attempts += 1
            if i == j:
                continue
            if labels[i] == labels[j]:
                continue
            key = (int(min(i, j)), int(max(i, j)))
            if key in neg_set:
                continue
            neg_set.add(key)
            neg_pairs_list.append(key)
            if len(neg_pairs_list) >= num_neg:
                break

    neg_pairs = np.array(neg_pairs_list, dtype=np.int64)
    return pos_pairs, neg_pairs


# ---------------------------------------------------------------------------
# Verification metrics
# ---------------------------------------------------------------------------


def cosine_similarity_pairs(embs: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Cosine sim for each (i, j) pair. Embeddings must be L2-normalized."""
    a = embs[pairs[:, 0]]
    b = embs[pairs[:, 1]]
    return np.sum(a * b, axis=1)


def threshold_sweep(
    pos_sims: np.ndarray, neg_sims: np.ndarray, num_steps: int = 500
) -> Dict[str, np.ndarray]:
    """Sweep similarity threshold from -1 to 1 over ``num_steps`` points.

    Returns dict with keys: thresholds, tpr (= TAR), fpr (= FAR), fnr.
    """
    thresholds = np.linspace(-1.0, 1.0, num_steps)
    tpr = np.empty(num_steps)
    fpr = np.empty(num_steps)
    for k, t in enumerate(thresholds):
        # Sample is "same" if sim >= threshold.
        tpr[k] = np.mean(pos_sims >= t)  # accept-as-same when truly same
        fpr[k] = np.mean(neg_sims >= t)  # accept-as-same when truly different
    fnr = 1.0 - tpr
    return {"thresholds": thresholds, "tpr": tpr, "fpr": fpr, "fnr": fnr}


def find_eer(sweep: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """Find EER + threshold by locating FAR vs FNR crossover with interpolation.

    As threshold increases, FAR (=fpr) decreases and FNR increases. The EER is
    the point where they cross.
    """
    fpr = sweep["fpr"]
    fnr = sweep["fnr"]
    thresholds = sweep["thresholds"]

    diff = fpr - fnr  # decreases monotonically (roughly) as threshold rises
    # Find the index where diff changes sign (or the first index where diff <= 0).
    sign = np.sign(diff)
    cross = np.where(np.diff(sign) != 0)[0]
    if len(cross) == 0:
        # No crossing — return the closest point.
        k = int(np.argmin(np.abs(diff)))
        return float((fpr[k] + fnr[k]) / 2.0), float(thresholds[k])

    k = int(cross[0])
    # Linear interpolation between k and k+1.
    d0, d1 = diff[k], diff[k + 1]
    if d1 == d0:
        alpha = 0.0
    else:
        alpha = d0 / (d0 - d1)
    eer = float(fpr[k] + alpha * (fpr[k + 1] - fpr[k]))
    fnr_at = float(fnr[k] + alpha * (fnr[k + 1] - fnr[k]))
    eer_t = float(thresholds[k] + alpha * (thresholds[k + 1] - thresholds[k]))
    return float((eer + fnr_at) / 2.0), eer_t


def tar_at_far(sweep: Dict[str, np.ndarray], far_target: float) -> float:
    """Linear-interpolate TAR (=TPR) at the threshold where FAR (=FPR) == far_target.

    As threshold rises, FAR decreases monotonically (roughly). We look for the
    largest threshold for which FAR >= far_target, then interpolate against the
    next step.
    """
    fpr = sweep["fpr"]
    tpr = sweep["tpr"]

    # Sort by threshold ascending → fpr decreasing → reverse for monotone-ascending
    # search by FAR.
    order = np.argsort(-fpr)  # descending FPR → ascending threshold direction
    fpr_s = fpr[order]
    tpr_s = tpr[order]

    # fpr_s is monotone non-increasing. Want largest index k where fpr_s[k] >= target.
    # Use np.searchsorted: since fpr_s is decreasing, work on -fpr_s.
    neg = -fpr_s
    k = int(np.searchsorted(neg, -far_target, side="right"))
    if k == 0:
        return float(tpr_s[0])
    if k >= len(fpr_s):
        return float(tpr_s[-1])
    # Interpolate between k-1 and k.
    x0, x1 = fpr_s[k - 1], fpr_s[k]
    y0, y1 = tpr_s[k - 1], tpr_s[k]
    if x0 == x1:
        return float(y0)
    alpha = (x0 - far_target) / (x0 - x1)
    return float(y0 + alpha * (y1 - y0))


def trapezoidal_auc(sweep: Dict[str, np.ndarray]) -> float:
    """ROC AUC via trapezoidal integration of TPR over FPR."""
    fpr = sweep["fpr"]
    tpr = sweep["tpr"]
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


COLOR_MAP = {
    "TL": "#1f77b4",
    "PU": "#2ca02c",
    "TLoss": "#d62728",
}


def plot_roc(results: List[Dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for row in results:
        if "sweep" not in row:
            continue
        fpr = np.array(row["sweep"]["fpr"])
        tpr = np.array(row["sweep"]["tpr"])
        order = np.argsort(fpr)
        ax.plot(
            fpr[order],
            tpr[order],
            label=f"{row['name']} (AUC={row['AUC']:.4f})",
            color=COLOR_MAP.get(row["key"], "black"),
            linewidth=2,
        )
        # Mark EER point.
        eer = row["EER"]
        ax.plot(
            [eer], [1 - eer], "o", color=COLOR_MAP.get(row["key"], "black"), markersize=7
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="Chance")
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("True Acceptance Rate (TAR)")
    ax.set_title("ROC Curves — Open-Set Verification (1062-sample test set)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_det(results: List[Dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    for row in results:
        if "sweep" not in row:
            continue
        fpr = np.array(row["sweep"]["fpr"])
        fnr = np.array(row["sweep"]["fnr"])
        order = np.argsort(fpr)
        fpr_s = fpr[order]
        fnr_s = fnr[order]
        mask = (fpr_s > 0) & (fnr_s > 0)
        ax.plot(
            fpr_s[mask],
            fnr_s[mask],
            label=f"{row['name']} (EER={row['EER']:.4f})",
            color=COLOR_MAP.get(row["key"], "black"),
            linewidth=2,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("False Acceptance Rate (FAR), log scale")
    ax.set_ylabel("False Non-Match Rate (FNR), log scale")
    ax.set_title("DET Curves — Open-Set Verification (log-log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_markdown(rows: List[Dict]) -> str:
    out = ["# Open-Set Verification Results\n"]
    out.append(
        "Verification protocol: 5,000 positive + 5,000 negative pairs sampled from "
        "the 1,062-sample held-out test set (RNG seed = 42). L2-normalized 512D "
        "FaceNet-backbone embeddings, cosine similarity, threshold swept from -1 to "
        "+1 in 500 steps. EER, TAR@FAR, and AUC reported with linear interpolation "
        "between adjacent threshold steps.\n"
    )
    out.append("| Model | EER | EER threshold | TAR @ FAR=1% | TAR @ FAR=0.1% | AUC | # Positive pairs | # Negative pairs |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            out.append(f"| {r['name']} | ERROR: {r['error']} | | | | | | |")
            continue
        out.append(
            "| {name} | {eer:.4f} | {eer_t:+.4f} | {tar1:.4f} | {tar01:.4f} | "
            "{auc:.4f} | {npos} | {nneg} |".format(
                name=r["name"],
                eer=r["EER"],
                eer_t=r["EER_threshold"],
                tar1=r["TAR_at_FAR_1pct"],
                tar01=r["TAR_at_FAR_0p1pct"],
                auc=r["AUC"],
                npos=r["num_positive_pairs"],
                nneg=r["num_negative_pairs"],
            )
        )
    out.append("")
    out.append("Lower EER is better. Higher TAR and AUC are better.")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def evaluate_model(
    spec: Dict, images: np.ndarray, pos_pairs: np.ndarray, neg_pairs: np.ndarray
) -> Dict:
    print(f"\n  [{spec['key']}] {spec['name']}")
    if spec["kind"] == "classifier":
        embedder = build_classifier_embedder(spec["path"])
    else:
        embedder = build_triplet_embedder(spec["path_weights"], spec["path_best_weights"])

    print(f"    Embedder output shape: {embedder.output_shape}")
    print(f"    Extracting L2-normalized embeddings for {len(images)} samples...")
    embs = extract_l2_embeddings(embedder, images)
    print(f"    Embedding matrix: {embs.shape}")

    pos_sims = cosine_similarity_pairs(embs, pos_pairs)
    neg_sims = cosine_similarity_pairs(embs, neg_pairs)
    print(
        f"    Positive sim: mean={pos_sims.mean():.4f} std={pos_sims.std():.4f}; "
        f"Negative sim: mean={neg_sims.mean():.4f} std={neg_sims.std():.4f}"
    )

    sweep = threshold_sweep(pos_sims, neg_sims, num_steps=500)
    eer, eer_t = find_eer(sweep)
    tar1 = tar_at_far(sweep, 0.01)
    tar01 = tar_at_far(sweep, 0.001)

    # AUC via sklearn (robust) and trapezoid (sanity check).
    y_true = np.concatenate([np.ones(len(pos_sims)), np.zeros(len(neg_sims))])
    y_score = np.concatenate([pos_sims, neg_sims])
    auc_sklearn = float(roc_auc_score(y_true, y_score))
    auc_trap = trapezoidal_auc(sweep)

    print(
        f"    EER={eer:.4f} @ thr={eer_t:+.4f}, TAR@1%FAR={tar1:.4f}, "
        f"TAR@0.1%FAR={tar01:.4f}, AUC={auc_sklearn:.4f} (trap={auc_trap:.4f})"
    )

    # Compact serialisable sweep (keep all 500 points).
    sweep_serial = [
        {
            "threshold": float(sweep["thresholds"][k]),
            "tpr": float(sweep["tpr"][k]),
            "fpr": float(sweep["fpr"][k]),
        }
        for k in range(len(sweep["thresholds"]))
    ]

    return {
        "key": spec["key"],
        "name": spec["name"],
        "embedding_dim": int(embs.shape[1]),
        "num_positive_pairs": int(len(pos_pairs)),
        "num_negative_pairs": int(len(neg_pairs)),
        "EER": round(eer, 6),
        "EER_threshold": round(eer_t, 6),
        "TAR_at_FAR_1pct": round(tar1, 6),
        "TAR_at_FAR_0p1pct": round(tar01, 6),
        "AUC": round(auc_sklearn, 6),
        "AUC_trapezoidal": round(auc_trap, 6),
        "pos_sim_mean": round(float(pos_sims.mean()), 6),
        "neg_sim_mean": round(float(neg_sims.mean()), 6),
        "threshold_sweep": sweep_serial,
        # In-memory only for plotting (stripped on JSON dump).
        "sweep": {
            "thresholds": sweep["thresholds"].tolist(),
            "tpr": sweep["tpr"].tolist(),
            "fpr": sweep["fpr"].tolist(),
            "fnr": sweep["fnr"].tolist(),
        },
    }


def main():
    print("=" * 60)
    print("OPEN-SET VERIFICATION EVALUATION (FaceNet backbone, 512D)")
    print("=" * 60)
    print(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"  TensorFlow {tf.__version__}, GPU devices: {tf.config.list_physical_devices('GPU')}")

    print("\nLoading test split...")
    images, labels, class_names = collect_test_set()
    print(f"  Test samples: {len(images)}, classes: {len(class_names)}")

    print("\nSampling pairs (seed=42)...")
    pos_pairs, neg_pairs = sample_pairs(labels, num_pos=5000, num_neg=5000, seed=42)
    print(f"  Positive pairs: {len(pos_pairs)} (target 5000)")
    print(f"  Negative pairs: {len(neg_pairs)} (target 5000)")

    rows = []
    for spec in MODELS:
        try:
            rows.append(evaluate_model(spec, images, pos_pairs, neg_pairs))
        except Exception as exc:
            import traceback

            traceback.print_exc()
            print(f"  [error] {spec['key']} failed: {exc}")
            rows.append({"key": spec["key"], "name": spec["name"], "error": str(exc)})

    # Render figures (uses in-memory "sweep" arrays).
    print("\nRendering figures...")
    roc_path = FIGURES_DIR / "verification_roc.png"
    det_path = FIGURES_DIR / "verification_det.png"
    plot_roc([r for r in rows if "sweep" in r], roc_path)
    plot_det([r for r in rows if "sweep" in r], det_path)
    print(f"  Wrote {roc_path}")
    print(f"  Wrote {det_path}")

    # Strip the in-memory "sweep" before JSON dump (the structured
    # threshold_sweep field is the canonical serialisable form).
    for r in rows:
        r.pop("sweep", None)

    out_payload = {
        "protocol": "Open-set verification, 5000 positive + 5000 negative pairs",
        "rng_seed": 42,
        "num_test_samples": int(len(images)),
        "num_classes": int(len(class_names)),
        "class_names": class_names,
        "similarity": "cosine on L2-normalized 512D FaceNet-backbone embeddings",
        "threshold_sweep_steps": 500,
        "threshold_sweep_range": [-1.0, 1.0],
        "models": rows,
    }
    json_path = OUTPUT_DIR / "verification_results.json"
    with open(json_path, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nWrote {json_path}")

    md_path = OUTPUT_DIR / "verification_report.md"
    with open(md_path, "w") as f:
        f.write(render_markdown(rows))
    print(f"Wrote {md_path}")

    # Final summary table.
    print("\nFINAL SUMMARY")
    print("-" * 60)
    print(f"{'Model':<25} {'EER':>8} {'TAR@1%':>10} {'TAR@0.1%':>10} {'AUC':>8}")
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<25} ERROR: {r['error']}")
            continue
        print(
            f"{r['name']:<25} {r['EER']:>8.4f} {r['TAR_at_FAR_1pct']:>10.4f} "
            f"{r['TAR_at_FAR_0p1pct']:>10.4f} {r['AUC']:>8.4f}"
        )


if __name__ == "__main__":
    main()
