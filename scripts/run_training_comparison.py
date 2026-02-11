#!/usr/bin/env python3
"""
Comprehensive Training Comparison Script
Executes all 4 training runs: EfficientNetB0/MobileNetV3 × GPU/CPU
"""

import subprocess
import sys
import time
import json
import os
from pathlib import Path
from datetime import datetime
from bp_face_recognition.config.settings import settings


def run_training(
    backbone, dataset="seccam_2", epochs=20, batch_size=32, force_cpu=False
):
    """
    Run a single training configuration and return results.
    """
    # Determine platform
    platform = "CPU" if force_cpu else "GPU"

    print(f"\n{'='*60}")
    print(f"🚀 Starting Training: {backbone} on {platform}")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Build command
    cmd = [
        sys.executable,
        "experiments/train.py",
        "--dataset",
        dataset,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--backbone",
        backbone,
        "--fine-tune",  # Always enable fine-tuning for production models
    ]

    # Force CPU if specified
    cmd_env = None
    if force_cpu:
        # Set environment variable to disable GPU
        cmd_env = dict(os.environ)
        cmd_env["CUDA_VISIBLE_DEVICES"] = "-1"

    start_time = time.time()

    try:
        # Run training
        result = subprocess.run(
            cmd, cwd=settings.ROOT_DIR, capture_output=True, text=True, env=cmd_env
        )

        execution_time = time.time() - start_time

        # Parse results from output
        output_lines = result.stdout.split("\n")
        val_accuracy = 0.0
        model_path = None

        for line in output_lines:
            if "Final validation accuracy:" in line:
                val_accuracy = float(line.split(":")[-1].strip())
            elif "Model saved to:" in line:
                model_path = line.split(":")[-1].strip()

        # Record training summary
        training_summary = {
            "backbone": backbone,
            "platform": platform,
            "execution_time_seconds": execution_time,
            "execution_time_minutes": execution_time / 60,
            "final_val_accuracy": val_accuracy,
            "model_path": model_path,
            "success": result.returncode == 0,
            "timestamp": datetime.now().isoformat(),
            "stdout": result.stdout,
            "stderr": result.stderr if result.stderr else None,
        }

        if training_summary["success"]:
            print(f"✅ Training completed successfully!")
            print(f"📊 Final Val Accuracy: {val_accuracy:.4f}")
            print(f"⏱️  Total Time: {execution_time:.2f}s ({execution_time/60:.2f}m)")
            print(f"💾 Model saved to: {model_path}")
        else:
            print(f"❌ Training failed!")
            print(f"📋 Return code: {result.returncode}")
            if result.stderr:
                print(f"🚨 Error: {result.stderr}")

        return training_summary

    except Exception as e:
        error_summary = {
            "backbone": backbone,
            "platform": platform,
            "execution_time_seconds": time.time() - start_time,
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }
        print(f"❌ Training failed with exception: {e}")
        return error_summary


def run_training_matrix():
    """
    Execute the complete 4-run training matrix.
    """
    print("🔥 Starting Comprehensive Training Matrix")
    print("📊 Matrix: EfficientNetB0/MobileNetV3 × GPU/CPU")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Training configurations
    training_configs = [
        ("EfficientNetB0", False),  # GPU
        ("MobileNetV3Small", False),  # GPU
        ("EfficientNetB0", True),  # CPU
        ("MobileNetV3Small", True),  # CPU
    ]

    results = []

    for backbone, force_cpu in training_configs:
        # Run individual training
        result = run_training(
            backbone=backbone, epochs=20, batch_size=32, force_cpu=force_cpu
        )
        results.append(result)

        # Brief pause between runs
        time.sleep(5)

        # Check if we should continue with CPU runs
        if not force_cpu and not result["success"]:
            print("⚠️ GPU training failed, but continuing with comparison...")

    # Save comprehensive results
    comparison_results = {
        "summary": {
            "total_runs": len(results),
            "successful_runs": sum(1 for r in results if r["success"]),
            "failed_runs": sum(1 for r in results if not r["success"]),
            "timestamp": datetime.now().isoformat(),
        },
        "detailed_results": results,
        "analysis": analyze_results(results) if results else {},
    }

    # Save results
    results_path = settings.LOGS_DIR / "training_comparison_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(comparison_results, f, indent=2)

    print(f"\n{'='*60}")
    print("📊 TRAINING MATRIX COMPLETE")
    print(f"{'='*60}")
    print_comparison_summary(comparison_results)
    print(f"📄 Full results saved to: {results_path}")

    return comparison_results


def analyze_results(results):
    """
    Analyze training results and generate insights.
    """
    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        return {"error": "No successful training runs to analyze"}

    # Find best results by accuracy and speed
    best_accuracy = max(successful_results, key=lambda x: x["final_val_accuracy"])
    fastest_training = min(
        successful_results, key=lambda x: x["execution_time_seconds"]
    )

    # GPU vs CPU comparison
    gpu_results = [r for r in successful_results if r["platform"] == "GPU"]
    cpu_results = [r for r in successful_results if r["platform"] == "CPU"]

    # Backbone comparison
    enet_results = [r for r in successful_results if "EfficientNet" in r["backbone"]]
    mobile_results = [r for r in successful_results if "MobileNet" in r["backbone"]]

    analysis = {
        "best_overall_accuracy": {
            "backbone": best_accuracy["backbone"],
            "platform": best_accuracy["platform"],
            "accuracy": best_accuracy["final_val_accuracy"],
            "time_minutes": best_accuracy["execution_time_minutes"],
        },
        "fastest_training": {
            "backbone": fastest_training["backbone"],
            "platform": fastest_training["platform"],
            "time_minutes": fastest_training["execution_time_minutes"],
            "accuracy": fastest_training["final_val_accuracy"],
        },
        "gpu_vs_cpu": {
            "avg_gpu_time": sum(r["execution_time_minutes"] for r in gpu_results)
            / len(gpu_results)
            if gpu_results
            else 0,
            "avg_cpu_time": sum(r["execution_time_minutes"] for r in cpu_results)
            / len(cpu_results)
            if cpu_results
            else 0,
            "gpu_speedup": (
                sum(r["execution_time_minutes"] for r in cpu_results) / len(cpu_results)
            )
            / (sum(r["execution_time_minutes"] for r in gpu_results) / len(gpu_results))
            if gpu_results and cpu_results
            else 0,
        },
        "backbone_comparison": {
            "avg_enet_accuracy": sum(r["final_val_accuracy"] for r in enet_results)
            / len(enet_results)
            if enet_results
            else 0,
            "avg_mobile_accuracy": sum(r["final_val_accuracy"] for r in mobile_results)
            / len(mobile_results)
            if mobile_results
            else 0,
        },
    }

    return analysis


def print_comparison_summary(results):
    """
    Print a formatted summary of the training comparison results.
    """
    print(f"\n📈 COMPARISON SUMMARY:")
    print(
        f"✅ Successful runs: {results['summary']['successful_runs']}/{results['summary']['total_runs']}"
    )

    if "analysis" in results and "best_overall_accuracy" in results["analysis"]:
        best = results["analysis"]["best_overall_accuracy"]
        fastest = results["analysis"]["fastest_training"]

        print(f"\n🏆 Best Overall Accuracy:")
        print(f"   📊 {best['backbone']} on {best['platform']}: {best['accuracy']:.4f}")
        print(f"   ⏱️  Time: {best['time_minutes']:.2f} minutes")

        print(f"\n⚡ Fastest Training:")
        print(
            f"   🚀 {fastest['backbone']} on {fastest['platform']}: {fastest['time_minutes']:.2f} minutes"
        )
        print(f"   📊 Accuracy: {fastest['accuracy']:.4f}")

        if results["analysis"]["gpu_vs_cpu"]["gpu_speedup"] > 0:
            speedup = results["analysis"]["gpu_vs_cpu"]["gpu_speedup"]
            print(f"\n🔄 GPU vs CPU Performance:")
            print(
                f"   🐌 Avg CPU time: {results['analysis']['gpu_vs_cpu']['avg_cpu_time']:.2f} minutes"
            )
            print(
                f"   🚀 Avg GPU time: {results['analysis']['gpu_vs_cpu']['avg_gpu_time']:.2f} minutes"
            )
            print(f"   ⚡ GPU speedup: {speedup:.2f}x faster")

        enet_acc = results["analysis"]["backbone_comparison"]["avg_enet_accuracy"]
        mobile_acc = results["analysis"]["backbone_comparison"]["avg_mobile_accuracy"]
        if enet_acc > 0 and mobile_acc > 0:
            print(f"\n🤖 Backbone Comparison:")
            print(f"   📊 EfficientNetB0 avg accuracy: {enet_acc:.4f}")
            print(f"   📊 MobileNetV3 avg accuracy: {mobile_acc:.4f}")


if __name__ == "__main__":
    # Ensure we're in the right directory
    import os

    os.chdir(settings.ROOT_DIR)

    print("🔥 Comprehensive Training Comparison Starting")
    print("⚠️  This will run 4 training sessions (GPU + CPU for both backbones)")
    print("⏱️  Total estimated time: 6-27 hours depending on hardware")

    response = input(
        "\n🤔 Do you want to proceed with the full training matrix? (y/N): "
    )
    if response.lower() in ["y", "yes"]:
        run_training_matrix()
    else:
        print("❌ Training comparison cancelled")

        # Allow single run
        print("\n🔧 Or run a single configuration:")
        print("1. EfficientNetB0 GPU")
        print("2. MobileNetV3 GPU")
        print("3. EfficientNetB0 CPU")
        print("4. MobileNetV3 CPU")

        choice = input("🎯 Enter choice (1-4): ")
        configs = [
            ("EfficientNetB0", False),
            ("MobileNetV3Small", False),
            ("EfficientNetB0", True),
            ("MobileNetV3Small", True),
        ]

        if choice in ["1", "2", "3", "4"]:
            backbone, force_cpu = configs[int(choice) - 1]
            result = run_training(backbone=backbone, force_cpu=force_cpu)
            print(f"\n📊 Single run complete: {result['success']}")
        else:
            print("❌ Invalid choice")
