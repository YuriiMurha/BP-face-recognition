"""5-seed cross-validation across FaceNet fine-tuning approaches.

Runs each of the three trainers (Transfer Learning, Progressive Unfreezing,
Triplet Loss) across 5 random seeds and aggregates the test accuracies into a
JSON + Markdown report. Each per-seed run is a separate subprocess (so Keras /
TensorFlow state is fully isolated between runs), launched via WSL2 against the
`.venv-wsl` virtual environment so training runs on the GPU.

Outputs:
  - results/cross_validation.json       — machine-readable aggregate
  - results/cross_validation_report.md  — human-readable summary
  - src/bp_face_recognition/models/finetuned/cv/seed_{seed}/  — per-seed artifacts
  - data/logs/cv_progress.log            — incremental progress log

This script is run from the Windows host. Each WSL invocation cd's into the
project root inside WSL and activates the WSL venv. We choose subprocess
isolation over in-process `clear_session()` calls so that an OOM or hung TF
process in one approach cannot poison the others.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WSL_DISTRO = "Ubuntu-22.04"
WSL_PROJECT_ROOT = "/mnt/d/Coding/Personal/BP-face-recognition"

CV_SEEDS: List[int] = [42, 123, 456, 789, 1024]

CV_OUTPUT_DIR = PROJECT_ROOT / "src" / "bp_face_recognition" / "models" / "finetuned" / "cv"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "data" / "logs"

CV_JSON_PATH = RESULTS_DIR / "cross_validation.json"
CV_REPORT_PATH = RESULTS_DIR / "cross_validation_report.md"
PROGRESS_LOG_PATH = LOG_DIR / "cv_progress.log"

# Approach definitions. Each entry encodes how to invoke its trainer and how to
# pull the test accuracy back out afterwards. The `relative_model_dir` is the
# path passed via --model-dir, **relative to the WSL project root** since the
# trainers expect POSIX paths. We use the same path on the Windows side because
# `/mnt/d/.../BP-face-recognition/foo` resolves to `D:\...\BP-face-recognition\foo`.
APPROACHES: List[Dict] = [
    {
        "name": "TL",
        "label": "Transfer Learning (Option A)",
        "module": "bp_face_recognition.vision.training.finetune.facenet_transfer_trainer",
        "extra_args": [],
        "fallback_args": [],  # nothing to fall back to: TL is already small batch
        "report_file": "facenet_transfer_report.json",
        "accuracy_key": ("results", "test_accuracy"),
        "needs_eval_step": False,
    },
    {
        "name": "TLoss",
        "label": "Triplet Loss (Option C)",
        "module": "bp_face_recognition.vision.training.finetune.facenet_triplet_trainer",
        "extra_args": ["--batch-size", "8"],
        "fallback_args": ["--batch-size", "4"],
        "report_file": "facenet_triplet_evaluation.json",
        "accuracy_key": ("test_accuracy",),
        "needs_eval_step": True,
        "eval_module": "bp_face_recognition.vision.training.finetune.evaluate_triplet_model",
        "eval_weights_file": "facenet_triplet_v1.0.weights.h5",
        "eval_output_file": "facenet_triplet_evaluation.json",
    },
    {
        "name": "PU",
        "label": "Progressive Unfreezing (Option B)",
        "module": "bp_face_recognition.vision.training.finetune.facenet_progressive_trainer",
        "extra_args": [],
        "fallback_args": ["--batch-size", "16"],
        "report_file": "facenet_progressive_report.json",
        "accuracy_key": ("results", "test_accuracy"),
        "needs_eval_step": False,
    },
]

# Sanity-check ranges for seed=42. These are deliberately loose because the
# canonical numbers in the thesis (TL=92.84%, PU=99.15%, TLoss=94.63%) were
# produced under slightly different RNG conditions; for example the canonical
# TL run early-stopped at 2 epochs while seeded runs train to ~15-20 epochs and
# routinely exceed 95%. The sanity check only flags catastrophic deviation
# (broken seed plumbing, dataset bug, etc.).
SEED_42_EXPECTED: Dict[str, Tuple[float, float]] = {
    # name -> (lower_bound, upper_bound)
    "TL": (0.85, 0.99),
    "PU": (0.96, 1.0001),
    "TLoss": (0.88, 0.99),
}


# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------

LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(PROGRESS_LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("cv")


# ----------------------------------------------------------------------------
# WSL plumbing
# ----------------------------------------------------------------------------

def build_wsl_command(python_args: List[str], cwd: str = WSL_PROJECT_ROOT) -> List[str]:
    """Return a Windows-side command list that runs `python ...` inside WSL.

    We compose a single bash command that cd's into the project root, activates
    the WSL venv, sets PYTHONPATH=src, and execs the python invocation.
    """
    quoted_inner = " ".join(shlex.quote(a) for a in python_args)
    bash_line = (
        f"cd {shlex.quote(cwd)} && "
        f"source .venv-wsl/bin/activate && "
        f"PYTHONPATH=src python {quoted_inner}"
    )
    return ["wsl", "-d", WSL_DISTRO, "bash", "-c", bash_line]


def run_subprocess(
    python_args: List[str],
    log_path: Path,
    timeout_seconds: Optional[int] = None,
) -> Tuple[int, float]:
    """Run a python command inside WSL, streaming combined output to log_path.

    Returns (exit_code, wall_seconds).
    """
    cmd = build_wsl_command(python_args)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Launching: %s", " ".join(python_args))
    logger.info("WSL command: %s", cmd)
    start = time.time()

    with open(log_path, "w", encoding="utf-8", errors="replace") as log_fh:
        log_fh.write(f"# Command: {python_args}\n")
        log_fh.write(f"# Started: {datetime.now(timezone.utc).isoformat()}\n\n")
        log_fh.flush()
        try:
            proc = subprocess.run(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            log_fh.write(f"\n# TIMEOUT after {timeout_seconds}s\n")
            exit_code = -1

    elapsed = time.time() - start
    logger.info("Finished in %.1fs (exit=%d). Log: %s", elapsed, exit_code, log_path)
    return exit_code, elapsed


# ----------------------------------------------------------------------------
# Result parsing helpers
# ----------------------------------------------------------------------------

def extract_accuracy(report_path: Path, accuracy_key: Tuple[str, ...]) -> Optional[float]:
    """Drill into the JSON report using the dotted key path and return a float.

    Returns None if the file is missing, malformed, or the key path is absent.
    """
    if not report_path.exists():
        logger.warning("Report not found: %s", report_path)
        return None

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse JSON %s: %s", report_path, exc)
        return None

    cur = data
    for key in accuracy_key:
        if not isinstance(cur, dict) or key not in cur:
            logger.warning("Key %s missing in %s", accuracy_key, report_path)
            return None
        cur = cur[key]

    if cur is None:
        return None
    try:
        return float(cur)
    except (TypeError, ValueError):
        logger.warning("Accuracy value %r is not numeric in %s", cur, report_path)
        return None


# ----------------------------------------------------------------------------
# Per-approach runner
# ----------------------------------------------------------------------------

def run_one(
    approach: Dict,
    seed: int,
    seed_dir_posix: str,
    seed_dir_windows: Path,
) -> Dict:
    """Train one approach for one seed (with one retry on failure) and return a
    structured result dict including test accuracy or failure info.
    """
    name = approach["name"]
    log_path = LOG_DIR / f"cv_seed_{seed}_{name}.log"

    # Build the training command
    base_args = [
        "-m",
        approach["module"],
        "--seed",
        str(seed),
        "--model-dir",
        seed_dir_posix,
    ]
    base_args.extend(approach.get("extra_args", []))

    train_exit, train_elapsed = run_subprocess(base_args, log_path)

    retried = False
    if train_exit != 0 and approach.get("fallback_args"):
        retried = True
        logger.warning(
            "[%s seed=%d] First attempt failed (exit=%d). Retrying with fallback args %s",
            name,
            seed,
            train_exit,
            approach["fallback_args"],
        )
        fallback_args = [
            "-m",
            approach["module"],
            "--seed",
            str(seed),
            "--model-dir",
            seed_dir_posix,
        ]
        fallback_args.extend(approach["fallback_args"])
        retry_log = LOG_DIR / f"cv_seed_{seed}_{name}_retry.log"
        train_exit, retry_elapsed = run_subprocess(fallback_args, retry_log)
        train_elapsed += retry_elapsed

    # Optional eval step (Triplet)
    eval_exit = 0
    eval_elapsed = 0.0
    if approach.get("needs_eval_step") and train_exit == 0:
        weights_posix = f"{seed_dir_posix}/{approach['eval_weights_file']}"
        eval_output_posix = f"{seed_dir_posix}/{approach['eval_output_file']}"
        eval_args = [
            "-m",
            approach["eval_module"],
            "--seed",
            str(seed),
            "--weights",
            weights_posix,
            "--output",
            eval_output_posix,
        ]
        eval_log = LOG_DIR / f"cv_seed_{seed}_{name}_eval.log"
        eval_exit, eval_elapsed = run_subprocess(eval_args, eval_log)

    # Parse accuracy
    report_path = seed_dir_windows / approach["report_file"]
    accuracy: Optional[float] = None
    if train_exit == 0 and eval_exit == 0:
        accuracy = extract_accuracy(report_path, approach["accuracy_key"])

    return {
        "approach": name,
        "label": approach["label"],
        "seed": seed,
        "train_exit": train_exit,
        "eval_exit": eval_exit,
        "retried_with_fallback": retried,
        "elapsed_seconds": train_elapsed + eval_elapsed,
        "test_accuracy": accuracy,
        "report_path": str(report_path),
        "log_path": str(log_path),
    }


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def aggregate(
    seeds: List[int],
    raw_results: List[Dict],
    total_runtime_seconds: float,
) -> Dict:
    """Aggregate raw per-(approach, seed) results into the final dict."""
    aggregate: Dict = {
        "seeds": seeds,
        "approaches": {},
        "total_runtime_seconds": total_runtime_seconds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "failed_runs": [],
        "raw_results": raw_results,
    }

    for approach in APPROACHES:
        name = approach["name"]
        accs = []
        for r in raw_results:
            if r["approach"] != name:
                continue
            if r["test_accuracy"] is not None:
                accs.append(r["test_accuracy"])
            else:
                aggregate["failed_runs"].append(
                    {
                        "approach": name,
                        "seed": r["seed"],
                        "train_exit": r["train_exit"],
                        "eval_exit": r["eval_exit"],
                        "retried_with_fallback": r["retried_with_fallback"],
                        "log_path": r["log_path"],
                    }
                )

        if accs:
            mean = statistics.fmean(accs)
            std = statistics.pstdev(accs) if len(accs) > 1 else 0.0
            sample_std = statistics.stdev(accs) if len(accs) > 1 else 0.0
            aggregate["approaches"][name] = {
                "label": approach["label"],
                "mean": mean,
                "std": std,  # population std (matches NumPy default for ddof=0)
                "sample_std": sample_std,  # ddof=1 for completeness
                "min": min(accs),
                "max": max(accs),
                "individual": accs,
                "n_successful": len(accs),
                "n_total": len(seeds),
            }
        else:
            aggregate["approaches"][name] = {
                "label": approach["label"],
                "mean": None,
                "std": None,
                "sample_std": None,
                "min": None,
                "max": None,
                "individual": [],
                "n_successful": 0,
                "n_total": len(seeds),
            }

    return aggregate


def write_markdown(report: Dict, path: Path) -> None:
    """Write a human-readable Markdown report."""
    lines: List[str] = []
    lines.append("# 5-Seed Cross-Validation: FaceNet Fine-Tuning Approaches\n")
    lines.append(f"_Generated: {report['timestamp']}_\n")
    lines.append(f"_Total wallclock: {report['total_runtime_seconds'] / 60:.1f} min_\n")
    lines.append(f"_Seeds: {report['seeds']}_\n")
    lines.append("\n## Summary\n")
    lines.append("| Approach | Mean | Std (pop) | Min | Max | n |")
    lines.append("|---|---|---|---|---|---|")
    for approach in APPROACHES:
        ap = report["approaches"][approach["name"]]
        if ap["mean"] is None:
            lines.append(
                f"| **{approach['name']}** ({approach['label']}) | FAILED | - | - | - | 0/{ap['n_total']} |"
            )
        else:
            lines.append(
                f"| **{approach['name']}** ({approach['label']}) | "
                f"{ap['mean'] * 100:.2f}% | "
                f"±{ap['std'] * 100:.2f}% | "
                f"{ap['min'] * 100:.2f}% | "
                f"{ap['max'] * 100:.2f}% | "
                f"{ap['n_successful']}/{ap['n_total']} |"
            )

    lines.append("\n## Per-seed accuracies\n")
    lines.append("| Approach | " + " | ".join(f"seed={s}" for s in report["seeds"]) + " |")
    lines.append("|---" * (1 + len(report["seeds"])) + "|")
    for approach in APPROACHES:
        ap = report["approaches"][approach["name"]]
        row = [f"**{approach['name']}**"]
        for s in report["seeds"]:
            match = [
                r["test_accuracy"]
                for r in report["raw_results"]
                if r["approach"] == approach["name"] and r["seed"] == s
            ]
            if match and match[0] is not None:
                row.append(f"{match[0] * 100:.2f}%")
            else:
                row.append("FAILED")
        lines.append("| " + " | ".join(row) + " |")

    if report["failed_runs"]:
        lines.append("\n## Failed runs\n")
        for f in report["failed_runs"]:
            lines.append(
                f"- **{f['approach']}** seed={f['seed']} train_exit={f['train_exit']} "
                f"eval_exit={f['eval_exit']} retried={f['retried_with_fallback']} "
                f"log=`{f['log_path']}`"
            )

    lines.append("\n## Configuration\n")
    lines.append("Each per-seed run was a separate WSL2 subprocess. Approaches:\n")
    for approach in APPROACHES:
        extras = " ".join(approach.get("extra_args", []))
        lines.append(f"- **{approach['name']}**: `{approach['module']}` {extras}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------

def sanity_check_seed_42(raw_results: List[Dict]) -> List[str]:
    """Check that seed=42 numbers are within roughly 1% of canonical values.

    Returns a list of warning messages. Does NOT abort.
    """
    warnings: List[str] = []
    for approach in APPROACHES:
        name = approach["name"]
        match = [
            r for r in raw_results if r["approach"] == name and r["seed"] == 42
        ]
        if not match or match[0]["test_accuracy"] is None:
            warnings.append(f"[{name}] seed=42 has no test_accuracy — cannot sanity-check")
            continue
        acc = match[0]["test_accuracy"]
        lo, hi = SEED_42_EXPECTED[name]
        if not (lo <= acc <= hi):
            warnings.append(
                f"[{name}] seed=42 accuracy {acc * 100:.2f}% is outside expected range "
                f"[{lo * 100:.1f}%, {hi * 100:.1f}%] — seed plumbing may be broken"
            )
        else:
            logger.info("[%s] seed=42 sanity OK: %.2f%%", name, acc * 100)
    return warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=CV_SEEDS,
        help="Seeds to run (default: 42 123 456 789 1024)",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        choices=["TL", "PU", "TLoss"],
        default=None,
        help="Restrict to a subset of approaches (debugging)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="If a per-seed report already exists for an approach, skip retraining and just parse it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds: List[int] = list(args.seeds)
    approaches = APPROACHES
    if args.only:
        approaches = [a for a in APPROACHES if a["name"] in args.only]

    logger.info("=" * 70)
    logger.info("5-SEED CROSS-VALIDATION")
    logger.info("=" * 70)
    logger.info("Seeds: %s", seeds)
    logger.info("Approaches: %s", [a["name"] for a in approaches])
    logger.info("Output dir: %s", CV_OUTPUT_DIR)
    logger.info("Skip existing: %s", args.skip_existing)
    logger.info("=" * 70)

    overall_start = time.time()
    raw_results: List[Dict] = []

    # Order: run cheap approaches first (TL, TLoss), then PU. Per seed: TL -> TLoss -> PU.
    # That way an early sanity-check failure on seed=42 surfaces quickly.
    for seed in seeds:
        seed_dir_windows = CV_OUTPUT_DIR / f"seed_{seed}"
        seed_dir_windows.mkdir(parents=True, exist_ok=True)
        seed_dir_posix = (
            f"{WSL_PROJECT_ROOT}/src/bp_face_recognition/models/finetuned/cv/seed_{seed}"
        )
        logger.info("\n===== seed=%d =====", seed)

        for approach in approaches:
            report_path = seed_dir_windows / approach["report_file"]
            if args.skip_existing and report_path.exists():
                acc = extract_accuracy(report_path, approach["accuracy_key"])
                logger.info(
                    "[%s seed=%d] SKIP (report exists, accuracy=%s)",
                    approach["name"],
                    seed,
                    f"{acc * 100:.2f}%" if acc is not None else "unparseable",
                )
                raw_results.append(
                    {
                        "approach": approach["name"],
                        "label": approach["label"],
                        "seed": seed,
                        "train_exit": 0,
                        "eval_exit": 0,
                        "retried_with_fallback": False,
                        "elapsed_seconds": 0.0,
                        "test_accuracy": acc,
                        "report_path": str(report_path),
                        "log_path": "(skipped)",
                        "skipped": True,
                    }
                )
                continue

            result = run_one(approach, seed, seed_dir_posix, seed_dir_windows)
            raw_results.append(result)

            # Incremental save after every single run — so a crash halfway never
            # loses prior results.
            elapsed = time.time() - overall_start
            partial = aggregate(seeds, raw_results, elapsed)
            CV_JSON_PATH.write_text(json.dumps(partial, indent=2), encoding="utf-8")
            write_markdown(partial, CV_REPORT_PATH)

            # Sanity-check seed=42 as soon as it's complete for any approach
            if seed == 42:
                warnings = sanity_check_seed_42([result])
                for w in warnings:
                    logger.warning(w)

    total_runtime = time.time() - overall_start
    final = aggregate(seeds, raw_results, total_runtime)
    CV_JSON_PATH.write_text(json.dumps(final, indent=2), encoding="utf-8")
    write_markdown(final, CV_REPORT_PATH)

    logger.info("\n" + "=" * 70)
    logger.info("CROSS-VALIDATION COMPLETE")
    logger.info("=" * 70)
    logger.info("Total wallclock: %.1f min", total_runtime / 60)
    for approach in approaches:
        ap = final["approaches"][approach["name"]]
        if ap["mean"] is not None:
            logger.info(
                "[%s] mean=%.4f std=%.4f n=%d/%d",
                approach["name"],
                ap["mean"],
                ap["std"],
                ap["n_successful"],
                ap["n_total"],
            )
        else:
            logger.info("[%s] ALL FAILED", approach["name"])

    if final["failed_runs"]:
        logger.warning("Failed runs: %d", len(final["failed_runs"]))
        for f in final["failed_runs"]:
            logger.warning(
                "  - %s seed=%d (train_exit=%d eval_exit=%d, log=%s)",
                f["approach"],
                f["seed"],
                f["train_exit"],
                f["eval_exit"],
                f["log_path"],
            )

    logger.info("Outputs:")
    logger.info("  JSON: %s", CV_JSON_PATH)
    logger.info("  Markdown: %s", CV_REPORT_PATH)
    logger.info("  Per-seed dirs: %s", CV_OUTPUT_DIR)

    # Final sanity check across all seed=42 results
    warnings = sanity_check_seed_42(raw_results)
    if warnings:
        logger.warning("Sanity check warnings:")
        for w in warnings:
            logger.warning("  - %s", w)

    return 0


if __name__ == "__main__":
    sys.exit(main())
