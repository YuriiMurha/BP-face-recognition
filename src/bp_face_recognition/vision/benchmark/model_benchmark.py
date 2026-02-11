"""
Comprehensive model benchmark tracking system for BP Face Recognition.
Tracks all model variants with detailed metrics for comparison.
"""

import json
import time
import psutil
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import tensorflow as tf
from dataclasses import dataclass, asdict
import subprocess
import platform


@dataclass
class ModelMetrics:
    """Complete model performance metrics."""

    model_name: str
    model_path: str
    architecture: str
    training_epochs: int
    quantized: bool
    platform: str

    # Performance metrics
    inference_time_ms: float
    memory_usage_mb: float
    model_size_mb: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    # Dataset info
    dataset_name: str
    test_samples: int
    classes: int

    # System info
    cpu_info: str
    gpu_info: str
    timestamp: str
    batch_size: int
    input_shape: Tuple[int, ...]


class ModelBenchmarkTracker:
    """Tracks and compares model performance across all variants."""

    def __init__(self, results_dir: str = "data/benchmarks"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.results_dir / "model_benchmarks.json"
        self.current_results = self._load_existing_results()

    def _load_existing_results(self) -> List[Dict]:
        """Load existing benchmark results."""
        if self.results_file.exists():
            with open(self.results_file, "r") as f:
                return json.load(f)
        return []

    def get_system_info(self) -> Tuple[str, str]:
        """Get CPU and GPU information."""
        try:
            cpu_info = f"{platform.processor()} ({psutil.cpu_count()} cores)"

            gpu_info = "None"
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    gpu_info = result.stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            return cpu_info, gpu_info
        except Exception:
            return "Unknown", "Unknown"

    def benchmark_model(
        self,
        model_path: str,
        test_data_path: str,
        model_name: str,
        architecture: str,
        training_epochs: int,
        quantized: bool = False,
        batch_size: int = 32,
    ) -> ModelMetrics:
        """
        Comprehensive model benchmarking.

        Args:
            model_path: Path to model file
            test_data_path: Path to test dataset
            model_name: Model identifier
            architecture: Model architecture (EfficientNetB0, MobileNetV3, etc.)
            training_epochs: Number of training epochs
            quantized: Whether model is quantized
            batch_size: Batch size for inference

        Returns:
            ModelMetrics: Complete performance metrics
        """
        print(f"[BENCHMARK] Testing {model_name} (Quantized: {quantized})")

        # Get system info
        cpu_info, gpu_info = self.get_system_info()

        # Load model (handle both Keras and TFLite)
        if model_path.endswith(".tflite"):
            import tensorflow as tf

            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()

            # Get input/output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            model = {
                "type": "tflite",
                "interpreter": interpreter,
                "input_details": input_details,
                "output_details": output_details,
                "input_shape": input_details[0]["shape"][1:],  # Remove batch dimension
                "output_shape": output_details[0]["shape"][1:],
            }
        else:
            model = tf.keras.models.load_model(model_path)
            model["type"] = "keras"
            model["input_shape"] = model.input_shape[1:]  # Remove batch dimension
        input_shape = model["input_shape"]  # Remove batch dimension

        # Get model size
        model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)

        # Determine input shape based on model type
        if model_path.endswith(".tflite"):
            import tensorflow as tf

            # Load TFLite model to get input shape
            interpreter = tf.lite.Interpreter(model_path=model_path)
            input_details = interpreter.get_input_details()
            input_shape = input_details[0]["shape"][1:]  # Remove batch dimension
        else:
            # Load Keras model to get input shape
            temp_model = tf.keras.models.load_model(model_path)
            input_shape = temp_model.input_shape[1:]  # Remove batch dimension

        # Load test data (assume existing data loading logic)
        test_dataset = self._load_test_data("cropped/seccam_2", batch_size, input_shape)

        # Measure inference performance
        inference_times = []
        memory_usage = []

        # Memory baseline
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / (1024 * 1024)

        # Initialize tracking variables
        all_predictions = []
        all_labels = []
        correct_predictions = 0
        total_samples = 0

        # Benchmark inference
        for batch_images, batch_labels in test_dataset:
            batch_start = time.time()

            if model["type"] == "tflite":
                # TFLite inference
                interpreter = model["interpreter"]
                input_details = model["input_details"]
                output_details = model["output_details"]

                # Set input tensor - need to process each sample individually for TFLite
                batch_predictions = []
                for i in range(batch_images.shape[0]):
                    sample = batch_images[i : i + 1]  # Keep batch dimension
                    interpreter.set_tensor(input_details[0]["index"], sample.numpy())
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_details[0]["index"])
                    batch_predictions.append(pred)
                predictions = np.concatenate(batch_predictions, axis=0)
            else:
                # Keras inference
                keras_model = model["keras_model"]
                predictions = keras_model.predict(batch_images, verbose=0)

            batch_end = time.time()

            # Store metrics
            inference_times.extend(
                [(batch_end - batch_start) / len(batch_images) * 1000]
            )  # ms per image
            current_memory = process.memory_info().rss / (1024 * 1024)
            memory_usage.append(current_memory - baseline_memory)

            # Calculate accuracy
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(batch_labels, axis=1)

            correct_predictions += np.sum(predicted_classes == true_classes)
            total_samples += len(batch_images)

            all_predictions.extend(predicted_classes)
            all_labels.extend(true_classes)

        # Calculate comprehensive metrics
        accuracy = correct_predictions / total_samples
        precision, recall, f1 = self._calculate_precision_recall_f1(
            all_labels, all_predictions
        )

        # Create metrics object - convert numpy types to Python types
        metrics = ModelMetrics(
            model_name=model_name,
            model_path=model_path,
            architecture=architecture,
            training_epochs=training_epochs,
            quantized=quantized,
            platform="GPU" if gpu_info != "None" else "CPU",
            inference_time_ms=float(np.mean(inference_times)),
            memory_usage_mb=float(np.mean(memory_usage)),
            model_size_mb=model_size_mb,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            dataset_name="seccam_2",  # hardcoded for now
            test_samples=total_samples,
            classes=int(len(np.unique(all_labels))),
            cpu_info=cpu_info,
            gpu_info=gpu_info,
            timestamp=datetime.now().isoformat(),
            batch_size=batch_size,
            input_shape=tuple(input_shape),
        )

        # Create metrics object
        metrics = ModelMetrics(
            model_name=model_name,
            model_path=model_path,
            architecture=architecture,
            training_epochs=training_epochs,
            quantized=quantized,
            platform="GPU" if gpu_info != "None" else "CPU",
            inference_time_ms=np.mean(inference_times),
            memory_usage_mb=np.mean(memory_usage),
            model_size_mb=model_size_mb,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            dataset_name="seccam_2",  # hardcoded for now
            test_samples=total_samples,
            classes=len(np.unique(all_labels)),
            cpu_info=cpu_info,
            gpu_info=gpu_info,
            timestamp=datetime.now().isoformat(),
            batch_size=batch_size,
            input_shape=input_shape,
        )

        # Save results
        self._save_results(metrics)

        print(
            f"[BENCHMARK] Completed: Acc={accuracy:.4f}, Inf={metrics.inference_time_ms:.2f}ms, Size={model_size_mb:.1f}MB"
        )
        return metrics

    def _load_test_data(
        self, data_path: str, batch_size: int, input_shape: Tuple[int, ...]
    ):
        """Load test dataset for benchmarking."""
        # Use existing data loading logic
        try:
            from bp_face_recognition.vision.data.preprocessing import load_face_dataset

            train_ds, val_ds, test_ds = load_face_dataset(
                f"data/datasets/{data_path}",
                batch_size=batch_size,
                image_size=(input_shape[0], input_shape[1]),  # (height, width)
                validation_split=0.2,
                test_split=0.2,
            )
            return test_ds
        except ImportError:
            # Fallback: create dummy dataset for testing
            print(
                f"[BENCHMARK] Warning: Could not import preprocessing, using dummy dataset"
            )
            import tensorflow as tf
            import numpy as np

            # Create dummy test data
            dummy_images = np.random.random((32, *input_shape)).astype(np.float32)
            dummy_labels = tf.keras.utils.to_categorical(
                np.random.randint(0, 15, 32), 15
            )

            dataset = tf.data.Dataset.from_tensor_slices(
                (dummy_images, dummy_labels)
            ).batch(batch_size)
            return dataset
        return test_ds

    def _calculate_precision_recall_f1(
        self, y_true: List[int], y_pred: List[int]
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        from sklearn.metrics import precision_score, recall_score, f1_score

        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        return precision, recall, f1

    def _save_results(self, metrics: ModelMetrics):
        """Save benchmark results to JSON file."""
        result_dict = asdict(metrics)
        self.current_results.append(result_dict)

        with open(self.results_file, "w") as f:
            json.dump(self.current_results, f, indent=2)

    def compare_models(self, model_names: Optional[List[str]] = None) -> Dict:
        """Compare benchmarked models."""
        results = self.current_results

        if model_names:
            results = [r for r in results if r["model_name"] in model_names]

        if not results:
            return {"error": "No benchmark results found"}

        comparison = {
            "summary": {
                "total_models": len(results),
                "architectures": list(set(r["architecture"] for r in results)),
                "quantized_models": len([r for r in results if r["quantized"]]),
                "date_range": {
                    "earliest": min(r["timestamp"] for r in results),
                    "latest": max(r["timestamp"] for r in results),
                },
            },
            "performance_comparison": {},
        }

        # Performance metrics comparison
        for metric in [
            "accuracy",
            "inference_time_ms",
            "model_size_mb",
            "memory_usage_mb",
        ]:
            values = [(r["model_name"], r[metric]) for r in results]
            comparison["performance_comparison"][metric] = {
                "best": min(values, key=lambda x: x[1])
                if metric != "accuracy"
                else max(values, key=lambda x: x[1]),
                "worst": max(values, key=lambda x: x[1])
                if metric != "accuracy"
                else min(values, key=lambda x: x[1]),
                "average": np.mean([v for _, v in values]),
                "all_values": values,
            }

        return comparison

    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        comparison = self.compare_models()
        results = self.current_results

        report = f"""
# BP Face Recognition Model Benchmark Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Models Tested**: {len(results)}
- **Architectures**: {', '.join(comparison['summary']['architectures'])}
- **Quantized Models**: {comparison['summary']['quantized_models']}

## Performance Comparison

### Accuracy Ranking
"""

        # Sort by accuracy
        sorted_by_accuracy = sorted(results, key=lambda x: x["accuracy"], reverse=True)
        for i, model in enumerate(sorted_by_accuracy, 1):
            report += f"{i}. {model['model_name']}: {model['accuracy']:.4f} ({model['architecture']}, {'Quantized' if model['quantized'] else 'Original'})\n"

        report += "\n### Inference Speed Ranking (ms per image)\n"
        sorted_by_speed = sorted(results, key=lambda x: x["inference_time_ms"])
        for i, model in enumerate(sorted_by_speed, 1):
            report += (
                f"{i}. {model['model_name']}: {model['inference_time_ms']:.2f}ms\n"
            )

        report += "\n### Model Size Ranking (MB)\n"
        sorted_by_size = sorted(results, key=lambda x: x["model_size_mb"])
        for i, model in enumerate(sorted_by_size, 1):
            report += f"{i}. {model['model_name']}: {model['model_size_mb']:.1f}MB\n"

        report += "\n## Detailed Results\n"
        for model in results:
            report += f"""
### {model['model_name']}
- **Architecture**: {model['architecture']}
- **Training Epochs**: {model['training_epochs']}
- **Quantized**: {model['quantized']}
- **Platform**: {model['platform']}

**Performance:**
- **Accuracy**: {model['accuracy']:.4f}
- **Precision**: {model['precision']:.4f}
- **Recall**: {model['recall']:.4f}
- **F1 Score**: {model['f1_score']:.4f}
- **Inference Time**: {model['inference_time_ms']:.2f}ms
- **Memory Usage**: {model['memory_usage_mb']:.1f}MB
- **Model Size**: {model['model_size_mb']:.1f}MB

**System:**
- **CPU**: {model['cpu_info']}
- **GPU**: {model['gpu_info']}
- **Tested**: {model['timestamp']}
"""

        # Save report
        report_file = (
            self.results_dir
            / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        with open(report_file, "w") as f:
            f.write(report)

        return str(report_file)


# Convenience function for quick benchmarking
def benchmark_model_variant(
    model_path: str,
    model_name: str,
    architecture: str,
    training_epochs: int,
    quantized: bool = False,
) -> ModelMetrics:
    """Quick benchmark function."""
    tracker = ModelBenchmarkTracker()
    return tracker.benchmark_model(
        model_path=model_path,
        test_data_path="data/processed/seccam_2",
        model_name=model_name,
        architecture=architecture,
        training_epochs=training_epochs,
        quantized=quantized,
    )
