#!/usr/bin/env python3
"""
Comprehensive GPU vs CPU Performance Benchmark for MediaPipe Face Detection

This script measures and compares performance between CPU and GPU configurations
across different image sizes and batch sizes. It's designed to work in both
Windows (CPU) and WSL (GPU) environments.
"""

import time
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from bp_face_recognition.models.methods.mediapipe_detector import MediaPipeDetector
from bp_face_recognition.utils.gpu import get_gpu_info, print_gpu_diagnostics

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Comprehensive performance benchmark for MediaPipe face detection."""

    def __init__(self):
        self.results = {}
        self.test_configs = [
            {"name": "Small (320x240)", "width": 320, "height": 240},
            {"name": "Medium (640x480)", "width": 640, "height": 480},
            {"name": "Large (1280x720)", "width": 1280, "height": 720},
            {"name": "HD (1920x1080)", "width": 1920, "height": 1080},
        ]
        self.batch_sizes = [1, 10, 50, 100]

    def generate_test_image(self, width: int, height: int) -> np.ndarray:
        """Generate a test image with face-like patterns."""
        # Create base image
        img = np.random.randint(0, 100, (height, width, 3), dtype=np.uint8)

        # Add some face-like rectangles to ensure detections
        for i in range(min(3, max(1, (width * height) // 200000))):
            x = np.random.randint(width // 4, 3 * width // 4)
            y = np.random.randint(height // 4, 3 * height // 4)
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)

            # Ensure bounds
            x = min(x, width - w)
            y = min(y, height - h)

            # Add face-like pattern
            img[y : y + h, x : x + w] = np.random.randint(
                150, 255, (h, w, 3), dtype=np.uint8
            )

        return img

    def benchmark_detector(
        self, detector: MediaPipeDetector, config: Dict, batch_size: int
    ) -> float:
        """Benchmark detector performance for given configuration."""
        images = [
            self.generate_test_image(config["width"], config["height"])
            for _ in range(batch_size)
        ]

        # Warm up
        for _ in range(3):
            detector.detect(images[0])

        # Benchmark
        start_time = time.time()
        for img in images:
            boxes = detector.detect(img)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / batch_size
        fps = 1.0 / avg_time

        return {
            "total_time": total_time,
            "avg_time_ms": avg_time * 1000,
            "fps": fps,
            "detections": len(boxes) if batch_size > 0 else 0,
        }

    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark across all configurations."""
        logger.info("Starting comprehensive performance benchmark...")

        # Print GPU diagnostics
        print_gpu_diagnostics()

        # Test both CPU and GPU configurations
        configs = [
            {
                "name": "Auto Detection (CPU)",
                "auto_gpu_detection": True,
                "use_gpu": False,
            },
            {"name": "Force CPU", "auto_gpu_detection": False, "use_gpu": False},
        ]

        # Add GPU test only if GPU is available
        gpu_info = get_gpu_info()
        if gpu_info["mediapipe_gpu_compatible"]:
            configs.append(
                {"name": "Force GPU", "auto_gpu_detection": False, "use_gpu": True}
            )
            configs.append(
                {
                    "name": "Auto Detection (GPU)",
                    "auto_gpu_detection": True,
                    "use_gpu": True,
                }
            )

        for config in configs:
            logger.info(f"Testing configuration: {config['name']}")

            try:
                detector = MediaPipeDetector(
                    auto_gpu_detection=config["auto_gpu_detection"],
                    use_gpu=config["use_gpu"],
                )

                # Check actual initialization method
                gpu_status = detector.get_gpu_status()
                actual_method = gpu_status["initialization_method"]

                self.results[config["name"]] = {
                    "method": actual_method,
                    "gpu_info": gpu_status["gpu_info"],
                    "performance": {},
                }

                # Run benchmarks for all image sizes
                for test_config in self.test_configs:
                    logger.info(f"  Testing {test_config['name']}...")

                    # Run benchmarks for different batch sizes
                    for batch_size in self.batch_sizes:
                        result = self.benchmark_detector(
                            detector, test_config, batch_size
                        )

                        key = f"{test_config['name']}_batch_{batch_size}"
                        self.results[config["name"]]["performance"][key] = result

                        logger.info(
                            f"    Batch {batch_size}: {result['avg_time_ms']:.2f}ms per image "
                            f"({result['fps']:.1f} FPS)"
                        )

                logger.info(f"  Completed {config['name']}")

            except Exception as e:
                logger.error(f"Failed to benchmark {config['name']}: {e}")
                self.results[config["name"]] = {"error": str(e), "method": "Failed"}

    def generate_report(self) -> str:
        """Generate comprehensive performance report."""
        report = []
        report.append("# MediaPipe GPU vs CPU Performance Report")
        report.append("=" * 60)
        report.append("")

        # Summary table
        report.append("## Performance Summary")
        report.append("")
        report.append(
            "| Configuration | Image Size | Batch | Avg Time (ms) | FPS | Detections |"
        )
        report.append(
            "|---------------|-------------|--------|---------------|-----|------------|"
        )

        for config_name, config_data in self.results.items():
            if "error" in config_data:
                report.append(
                    f"| {config_name} | ERROR | - | - | - | {config_data['error'][:50]} |"
                )
                continue

            for perf_key, perf_data in config_data["performance"].items():
                parts = perf_key.split("_")
                image_size = " ".join(parts[:-2])
                batch_size = parts[-1]

                report.append(
                    f"| {config_name} | {image_size} | {batch_size} | "
                    f"{perf_data['avg_time_ms']:.2f} | "
                    f"{perf_data['fps']:.1f} | {perf_data['detections']} |"
                )

        report.append("")

        # Performance improvement analysis
        report.append("## Performance Analysis")
        report.append("")

        cpu_results = {}
        gpu_results = {}

        for config_name, config_data in self.results.items():
            if "error" in config_data:
                continue

            if "GPU" in config_name:
                gpu_results = config_data["performance"]
            elif "CPU" in config_name:
                cpu_results = config_data["performance"]

        if cpu_results and gpu_results:
            report.append("### GPU vs CPU Speedup")
            report.append("")

            for key in cpu_results:
                if key in gpu_results:
                    cpu_time = cpu_results[key]["avg_time_ms"]
                    gpu_time = gpu_results[key]["avg_time_ms"]
                    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")

                    report.append(
                        f"- {key}: **{speedup:.2f}x faster** "
                        f"(CPU: {cpu_time:.2f}ms, GPU: {gpu_time:.2f}ms)"
                    )

        report.append("")

        # System information
        report.append("## System Information")
        report.append("")

        if self.results:
            first_result = next(iter(self.results.values()))
            if "gpu_info" in first_result:
                gpu_info = first_result["gpu_info"]

                report.append(f"- **Platform**: {gpu_info['platform']}")
                report.append(
                    f"- **TensorFlow GPUs**: {gpu_info['tensorflow_gpu_count']}"
                )
                report.append(f"- **OpenGL Supported**: {gpu_info['opengl_supported']}")
                report.append(f"- **CUDA Available**: {gpu_info['cuda_available']}")
                report.append(
                    f"- **MediaPipe GPU Compatible**: {gpu_info['mediapipe_gpu_compatible']}"
                )
                report.append(
                    f"- **Recommended Delegate**: {gpu_info['recommended_delegate']}"
                )

        report.append("")
        report.append("## Recommendations")
        report.append("")

        if gpu_results:
            report.append(
                "✅ **GPU acceleration is working** - Use GPU configuration for production:"
            )
            report.append("```python")
            report.append("detector = MediaPipeDetector(auto_gpu_detection=True)")
            report.append("```")
        else:
            report.append(
                "⚠️ **GPU not available** - Consider setting up WSL2 with GPU support:"
            )
            report.append(
                "1. Follow the setup guide in `.maintenance/WSL_GPU_SETUP.md`"
            )
            report.append("2. Run `scripts/setup_wsl_gpu.sh` in WSL")
            report.append("3. Verify with `~/gpu_verification.py`")

        report.append("")

        return "\n".join(report)

    def save_results(self, filename: str = None):
        """Save benchmark results to files."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"gpu_benchmark_{timestamp}"

        # Save JSON data
        json_file = project_root / f"{filename}.json"
        with open(json_file, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {json_file}")

        # Save markdown report
        report_file = project_root / f"{filename}.md"
        with open(report_file, "w") as f:
            f.write(self.generate_report())
        logger.info(f"Report saved to {report_file}")

        return json_file, report_file


def main():
    """Run the comprehensive benchmark."""
    print("MediaPipe GPU vs CPU Performance Benchmark")
    print("=" * 50)

    benchmark = PerformanceBenchmark()

    try:
        benchmark.run_comprehensive_benchmark()
        json_file, report_file = benchmark.save_results()

        print(f"\nBenchmark complete!")
        print(f"JSON results: {json_file}")
        print(f"Report: {report_file}")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
