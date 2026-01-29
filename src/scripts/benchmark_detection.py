"""
Benchmark script to compare MediaPipe vs MTCNN detection performance.
Run this to validate optimization gains before integration.
"""

import time
import cv2
import numpy as np
from typing import Dict, Any
from bp_face_recognition.models.factory import RecognizerFactory


class DetectionBenchmark:
    """Benchmark different face detection methods."""

    def __init__(self, test_images_path: str | None = None):
        self.detectors = {
            "mtcnn": RecognizerFactory.get_detector("mtcnn"),
            "haar": RecognizerFactory.get_detector("haar"),
            "dlib_hog": RecognizerFactory.get_detector("dlib_hog"),
            "face_recognition": RecognizerFactory.get_detector("face_recognition"),
        }

        # Try to add MediaPipe if available
        try:
            self.detectors["mediapipe"] = RecognizerFactory.get_detector("mediapipe")
        except Exception as e:
            print(f"MediaPipe not available: {e}")

    def test_on_synthetic_data(
        self, iterations: int = 100
    ) -> Dict[str, Dict[str, Any]]:
        """Test detectors on synthetic face-like patterns."""
        results = {}

        for detector_name, detector in self.detectors.items():
            print(f"\nTesting {detector_name}...")

            times = []
            face_counts = []

            for i in range(iterations):
                # Create test image with face-like pattern
                test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                # Add a face-like rectangle
                test_image[100:200, 200:300] = [150, 150, 150]  # Gray rectangle
                test_image[130:140, 230:240] = [0, 0, 0]  # Eyes
                test_image[130:140, 260:270] = [0, 0, 0]
                test_image[170:180, 245:255] = [100, 50, 50]  # Mouth

                start_time = time.time()

                try:
                    if hasattr(detector, "detect_with_confidence"):
                        detections = detector.detect_with_confidence(test_image)
                        faces = [box for box, conf in detections]
                    else:
                        faces = detector.detect(test_image)

                    detection_time = time.time() - start_time
                    times.append(detection_time)
                    face_counts.append(len(faces))

                except Exception as e:
                    print(f"Error in {detector_name}: {e}")
                    times.append(float("inf"))
                    face_counts.append(0)

            # Calculate statistics
            valid_times = [t for t in times if t != float("inf")]
            results[detector_name] = {
                "avg_time": np.mean(valid_times) if valid_times else float("inf"),
                "min_time": np.min(valid_times) if valid_times else float("inf"),
                "max_time": np.max(valid_times) if valid_times else float("inf"),
                "avg_faces": np.mean(face_counts),
                "success_rate": len(valid_times) / iterations,
                "fps_estimate": 1.0 / np.mean(valid_times) if valid_times else 0,
            }

        return results

    def print_results(self, results: Dict[str, Dict[str, Any]]):
        """Print benchmark results in a readable format."""
        print("\n" + "=" * 80)
        print("FACE DETECTION BENCHMARK RESULTS")
        print("=" * 80)

        print("Detector        Avg Time (ms)  FPS     Success Rate  Avg Faces")
        print("-" * 65)

        for detector_name, stats in results.items():
            if stats["avg_time"] != float("inf"):
                avg_time_ms = stats["avg_time"] * 1000
                fps = stats["fps_estimate"]
                success_rate = f"{stats['success_rate']*100:.1f}%"
                avg_faces = f"{stats['avg_faces']:.1f}"

                print(
                    f"{detector_name:<15} {avg_time_ms:<12.2f} {fps:<8.1f} {success_rate:<12} {avg_faces:<10}"
                )
            else:
                print(
                    f"{detector_name:<15} {'FAILED':<12} {'-':<8} {'0%':<12} {'-':<10}"
                )

        # Performance comparison
        print("\n" + "-" * 65)
        if "mtcnn" in results and "mediapipe" in results:
            mtcnn_fps = results["mtcnn"]["fps_estimate"]
            mediapipe_fps = results["mediapipe"]["fps_estimate"]

            if mtcnn_fps > 0 and mediapipe_fps > 0:
                speedup = mediapipe_fps / mtcnn_fps
                print(f"MediaPipe speedup over MTCNN: {speedup:.1f}x")

        # Recommendation
        best_detector = max(
            results.keys(),
            key=lambda k: results[k]["fps_estimate"]
            if results[k]["fps_estimate"] > 0
            else 0,
        )
        if results[best_detector]["fps_estimate"] > 0:
            print(f"\nRecommended detector: {best_detector}")
            print(f"Performance: {results[best_detector]['fps_estimate']:.1f} FPS")


def main():
    """Run benchmark."""
    print("Starting face detection benchmark...")
    print("Testing synthetic data for reliability comparison...\n")

    benchmark = DetectionBenchmark()
    results = benchmark.test_on_synthetic_data(iterations=50)
    benchmark.print_results(results)

    print(f"\nBenchmark completed!")
    print(
        "Results show relative performance - real-world performance may vary based on image content."
    )


if __name__ == "__main__":
    main()
