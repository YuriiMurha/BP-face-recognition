"""Run the GT-based detection eval one detector at a time in fresh subprocesses,
then merge the per-detector shards into the canonical thesis outputs.

Why subprocesses? When multiple detectors load TF/keras backends in the same
Python process, MTCNN's library has been observed to hang during init (likely a
state collision with already-loaded TFLite/XNNPACK from MediaPipe). Running each
detector in its own interpreter sidesteps the issue and bounds the blast radius:
a single detector that hangs only kills its own subprocess.

Output:
    results/detection_results_groundtruth.json
    results/detection_eval_report.md
"""

from __future__ import annotations

import json
import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results"

DETECTORS = ["MediaPipe", "Haar Cascade", "Dlib HOG", "MTCNN"]
PER_DETECTOR_TIMEOUT_S = 180  # seconds


def run_one(name: str) -> dict | None:
    print(f"\n[run] {name} ...")
    cmd = [
        sys.executable,
        "-u",
        "src/bp_face_recognition/evaluation/detection_eval_with_groundtruth.py",
        "--only",
        name,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU; avoids GPU contention with retrains

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            timeout=PER_DETECTOR_TIMEOUT_S,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.TimeoutExpired:
        print(f"  [timeout] {name} exceeded {PER_DETECTOR_TIMEOUT_S}s — skipped.")
        return None

    if result.returncode != 0:
        print(f"  [fail] {name} exit={result.returncode}\n{result.stdout[-600:]}")
        return None

    shard_path = OUTPUT_DIR / f"detection_shard_{name.lower().replace(' ', '_')}.json"
    if not shard_path.exists():
        print(f"  [warn] {name} finished but shard missing: {shard_path}")
        return None
    with open(shard_path) as f:
        return json.load(f)


def render_markdown(rows: list[dict], total_gt: int, num_images: int) -> str:
    lines = ["# Detection Evaluation with Ground Truth\n"]
    lines.append(
        f"**Test set:** {num_images} surveillance frames "
        f"({total_gt} annotated face bounding boxes).\n"
    )
    lines.append(
        "**Matching:** greedy IoU at threshold 0.5. Images resized so max(h,w) ≤ 800. "
        "Each detector evaluated in an isolated subprocess (CPU-only) to avoid cross-detector state collisions.\n"
    )

    lines.append("## Detection Quality (P / R / F1)\n")
    lines.append("| Method | TP | FP | FN | Precision | Recall | F1 | Mean IoU |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if not r:
            continue
        lines.append(
            f"| {r['detector']} | {r['tp']} | {r['fp']} | {r['fn']} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['mean_iou_matched']:.3f} |"
        )

    lines.append("\n## Detection Speed (CPU-only)\n")
    lines.append("| Method | Avg (ms) | Min (ms) | Max (ms) | FPS |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        if not r:
            continue
        lines.append(
            f"| {r['detector']} | {r['avg_time_ms']:.1f} | {r['min_time_ms']:.1f} | "
            f"{r['max_time_ms']:.1f} | {r['fps']:.1f} |"
        )

    return "\n".join(lines) + "\n"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in DETECTORS:
        r = run_one(name)
        rows.append(r)

    valid = [r for r in rows if r]
    if not valid:
        print("\n[fatal] no detector produced output.")
        sys.exit(1)

    # Use any successful shard for the run-wide metadata
    sample = valid[0]
    payload = {
        "iou_threshold": 0.5,
        "max_dim": 800,
        "num_test_images": sample["num_test_images"],
        "total_gt_boxes": sample["total_gt_boxes"],
        "detectors": [
            {k: v for k, v in r.items() if k != "per_image"}
            for r in valid
        ],
        "skipped": [name for name, r in zip(DETECTORS, rows) if r is None],
    }
    out_json = OUTPUT_DIR / "detection_results_groundtruth.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_json}")

    out_md = OUTPUT_DIR / "detection_eval_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(render_markdown(rows, sample["total_gt_boxes"], sample["num_test_images"]))
    print(f"Wrote {out_md}")

    if payload["skipped"]:
        print(f"\n[note] Skipped: {payload['skipped']}")


if __name__ == "__main__":
    main()
