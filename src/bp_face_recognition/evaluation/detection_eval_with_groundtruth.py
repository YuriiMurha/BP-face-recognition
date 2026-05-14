"""Detection evaluation with ground-truth bounding boxes.

Loads LabelMe-format JSON annotations from
    data/datasets/raw/{seccam,seccam_2,webcam}/test/labels/
and computes proper Precision / Recall / F1 against each detector's output
using greedy IoU matching at threshold 0.5.

Output:
    results/detection_results_groundtruth.json
    results/detection_eval_report.md
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

import cv2
import numpy as np

from bp_face_recognition.config.settings import settings
from bp_face_recognition.vision.factory import RecognizerFactory


PROJECT_ROOT = settings.ROOT_DIR
RAW_DATASETS = PROJECT_ROOT / "data" / "datasets" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Same constraints used by thesis_benchmark.py so numbers are comparable
MAX_DIM = 800
IOU_THRESHOLD = 0.5

DETECTORS: List[Tuple[str, str]] = [
    ("MediaPipe", "mediapipe_v1"),
    ("MTCNN", "mtcnn_v1"),
    ("Haar Cascade", "haar_v1"),
    ("Dlib HOG", "dlib_hog_v1"),
]


@dataclass
class GTSample:
    image_path: Path
    boxes_xywh: List[Tuple[int, int, int, int]]
    dataset: str

    @property
    def num_faces(self) -> int:
        return len(self.boxes_xywh)


@dataclass
class DetectorScore:
    name: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0
    iou_count: int = 0
    times_ms: List[float] = field(default_factory=list)
    per_image: List[Dict] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def mean_iou(self) -> float:
        return self.iou_sum / self.iou_count if self.iou_count else 0.0


def load_labelme_box(points: List[List[float]]) -> Tuple[int, int, int, int]:
    """Convert LabelMe rectangle (two corner points, any order) to (x, y, w, h)."""
    (x1, y1), (x2, y2) = points
    x_min = int(min(x1, x2))
    y_min = int(min(y1, y2))
    x_max = int(max(x1, x2))
    y_max = int(max(y1, y2))
    return x_min, y_min, x_max - x_min, y_max - y_min


def collect_ground_truth() -> List[GTSample]:
    samples: List[GTSample] = []
    for dataset in ("seccam", "seccam_2", "webcam"):
        labels_dir = RAW_DATASETS / dataset / "test" / "labels"
        images_dir = RAW_DATASETS / dataset / "test" / "images"
        if not labels_dir.exists():
            continue

        for label_file in sorted(labels_dir.glob("*.json")):
            with open(label_file, "r") as f:
                data = json.load(f)

            image_name = data.get("imagePath") or f"{label_file.stem}.jpg"
            image_path = images_dir / image_name
            if not image_path.exists():
                # Fall back to matching stem with any common extension
                for ext in (".jpg", ".jpeg", ".png"):
                    candidate = images_dir / f"{label_file.stem}{ext}"
                    if candidate.exists():
                        image_path = candidate
                        break
                else:
                    print(f"  [warn] image for {label_file.name} not found")
                    continue

            boxes = [
                load_labelme_box(shape["points"])
                for shape in data.get("shapes", [])
                if shape.get("shape_type") == "rectangle"
            ]
            samples.append(GTSample(image_path=image_path, boxes_xywh=boxes, dataset=dataset))

    return samples


def resize_with_scale(image: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image, scale


def scale_box(box: Tuple[int, int, int, int], scale: float) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    return (
        int(round(x * scale)),
        int(round(y * scale)),
        int(round(w * scale)),
        int(round(h * scale)),
    )


def iou_xywh(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def greedy_match(
    gt: List[Tuple[int, int, int, int]],
    pred: List[Tuple[int, int, int, int]],
    iou_thresh: float = IOU_THRESHOLD,
) -> Tuple[int, int, int, List[float]]:
    """Greedy IoU matching. Returns (tp, fp, fn, matched_ious)."""
    if not gt and not pred:
        return 0, 0, 0, []
    if not gt:
        return 0, len(pred), 0, []
    if not pred:
        return 0, 0, len(gt), []

    # Compute IoU matrix
    iou_mat = np.zeros((len(gt), len(pred)))
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            iou_mat[i, j] = iou_xywh(g, p)

    matched_gt = set()
    matched_pred = set()
    matched_ious: List[float] = []

    # Sort all (gt, pred) pairs by IoU desc, greedy assign
    pairs = [
        (iou_mat[i, j], i, j)
        for i in range(len(gt))
        for j in range(len(pred))
        if iou_mat[i, j] >= iou_thresh
    ]
    pairs.sort(reverse=True)

    for iou, i, j in pairs:
        if i in matched_gt or j in matched_pred:
            continue
        matched_gt.add(i)
        matched_pred.add(j)
        matched_ious.append(float(iou))

    tp = len(matched_ious)
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn, matched_ious


def evaluate_detector(name: str, detector_type: str, samples: List[GTSample]) -> DetectorScore:
    print(f"\n  [{name}] loading...")
    detector = RecognizerFactory.get_detector(detector_type)
    score = DetectorScore(name=name)

    # Warmup
    if samples:
        warm = cv2.imread(str(samples[0].image_path))
        if warm is not None:
            warm, _ = resize_with_scale(warm, MAX_DIM)
            try:
                detector.detect(warm)
            except Exception:
                pass

    for sample in samples:
        img = cv2.imread(str(sample.image_path))
        if img is None:
            print(f"    [warn] failed to read {sample.image_path.name}")
            continue
        img_resized, scale = resize_with_scale(img, MAX_DIM)
        gt_scaled = [scale_box(b, scale) for b in sample.boxes_xywh]

        start = time.time()
        try:
            preds_raw = detector.detect(img_resized)
        except Exception as exc:
            print(f"    [warn] {name} failed on {sample.image_path.name}: {exc}")
            preds_raw = []
        elapsed_ms = (time.time() - start) * 1000.0
        score.times_ms.append(elapsed_ms)

        preds = [tuple(int(v) for v in p[:4]) for p in preds_raw]

        tp, fp, fn, matched_ious = greedy_match(gt_scaled, preds)
        score.tp += tp
        score.fp += fp
        score.fn += fn
        score.iou_sum += sum(matched_ious)
        score.iou_count += len(matched_ious)

        score.per_image.append(
            {
                "image": sample.image_path.name,
                "dataset": sample.dataset,
                "gt_boxes": len(gt_scaled),
                "pred_boxes": len(preds),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "mean_iou_matched": float(np.mean(matched_ious)) if matched_ious else 0.0,
                "time_ms": elapsed_ms,
            }
        )

    print(
        f"    TP={score.tp} FP={score.fp} FN={score.fn}  "
        f"P={score.precision:.3f}  R={score.recall:.3f}  F1={score.f1:.3f}  "
        f"meanIoU={score.mean_iou:.3f}  avg={np.mean(score.times_ms):.1f}ms"
    )
    return score


def render_markdown(scores: List[DetectorScore], total_gt: int, num_images: int) -> str:
    lines = []
    lines.append("# Detection Evaluation with Ground Truth\n")
    lines.append(
        f"**Test set:** {num_images} surveillance frames "
        f"({total_gt} annotated face bounding boxes).\n"
    )
    lines.append(
        f"**Matching:** greedy IoU at threshold {IOU_THRESHOLD}. "
        f"Images resized so max(h,w) ≤ {MAX_DIM}.\n"
    )

    lines.append("## Detection Quality (P / R / F1)\n")
    lines.append("| Method | TP | FP | FN | Precision | Recall | F1 | Mean IoU |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in scores:
        lines.append(
            f"| {s.name} | {s.tp} | {s.fp} | {s.fn} | "
            f"{s.precision:.3f} | {s.recall:.3f} | {s.f1:.3f} | {s.mean_iou:.3f} |"
        )

    lines.append("\n## Detection Speed\n")
    lines.append("| Method | Avg (ms) | Min (ms) | Max (ms) | FPS |")
    lines.append("|---|---|---|---|---|")
    for s in scores:
        if not s.times_ms:
            continue
        avg = float(np.mean(s.times_ms))
        fps = 1000.0 / avg if avg else 0.0
        lines.append(
            f"| {s.name} | {avg:.1f} | {min(s.times_ms):.1f} | "
            f"{max(s.times_ms):.1f} | {fps:.1f} |"
        )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        default=None,
        help="Run only one detector (matches DETECTORS list display name, case-insensitive). "
        "Output goes to a per-detector JSON shard.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DETECTION EVAL WITH GROUND TRUTH")
    print("=" * 60)

    samples = collect_ground_truth()
    total_gt_boxes = sum(s.num_faces for s in samples)
    print(f"Loaded {len(samples)} test images, {total_gt_boxes} GT boxes total")
    by_dataset: Dict[str, int] = {}
    for s in samples:
        by_dataset[s.dataset] = by_dataset.get(s.dataset, 0) + 1
    for k, v in by_dataset.items():
        print(f"  {k}: {v} images")

    detectors_to_run = DETECTORS
    if args.only:
        wanted = args.only.lower()
        detectors_to_run = [(n, t) for n, t in DETECTORS if n.lower() == wanted]
        if not detectors_to_run:
            print(f"  [error] no detector matches --only={args.only!r}")
            return

    scores = [evaluate_detector(name, dtype, samples) for name, dtype in detectors_to_run]

    # Per-detector shard mode: write a small JSON and exit; the merge happens
    # in the caller (run_detection_eval.py).
    if args.only:
        shard_path = OUTPUT_DIR / f"detection_shard_{args.only.lower().replace(' ', '_')}.json"
        with open(shard_path, "w") as f:
            json.dump(
                {
                    "detector": scores[0].name,
                    "tp": scores[0].tp,
                    "fp": scores[0].fp,
                    "fn": scores[0].fn,
                    "precision": round(scores[0].precision, 4),
                    "recall": round(scores[0].recall, 4),
                    "f1": round(scores[0].f1, 4),
                    "mean_iou_matched": round(scores[0].mean_iou, 4),
                    "avg_time_ms": round(float(sum(scores[0].times_ms) / len(scores[0].times_ms)) if scores[0].times_ms else 0.0, 2),
                    "min_time_ms": round(float(min(scores[0].times_ms)) if scores[0].times_ms else 0.0, 2),
                    "max_time_ms": round(float(max(scores[0].times_ms)) if scores[0].times_ms else 0.0, 2),
                    "fps": round(1000.0 / (sum(scores[0].times_ms) / len(scores[0].times_ms)), 2) if scores[0].times_ms else 0.0,
                    "num_test_images": len(samples),
                    "total_gt_boxes": total_gt_boxes,
                    "per_image": scores[0].per_image,
                },
                f,
                indent=2,
            )
        print(f"\nWrote shard {shard_path}")
        return

    # JSON output
    payload = {
        "iou_threshold": IOU_THRESHOLD,
        "max_dim": MAX_DIM,
        "num_test_images": len(samples),
        "total_gt_boxes": total_gt_boxes,
        "samples_by_dataset": by_dataset,
        "detectors": [
            {
                "name": s.name,
                "tp": s.tp,
                "fp": s.fp,
                "fn": s.fn,
                "precision": round(s.precision, 4),
                "recall": round(s.recall, 4),
                "f1": round(s.f1, 4),
                "mean_iou_matched": round(s.mean_iou, 4),
                "avg_time_ms": round(float(np.mean(s.times_ms)) if s.times_ms else 0.0, 2),
                "min_time_ms": round(float(min(s.times_ms)) if s.times_ms else 0.0, 2),
                "max_time_ms": round(float(max(s.times_ms)) if s.times_ms else 0.0, 2),
                "fps": round(1000.0 / float(np.mean(s.times_ms)), 2) if s.times_ms else 0.0,
                "per_image": s.per_image,
            }
            for s in scores
        ],
    }
    json_path = OUTPUT_DIR / "detection_results_groundtruth.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")

    md_path = OUTPUT_DIR / "detection_eval_report.md"
    with open(md_path, "w") as f:
        f.write(render_markdown(scores, total_gt_boxes, len(samples)))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
