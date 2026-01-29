"""Test quantization and MediaPipe integration (Integration Tests)."""

import pytest
import numpy as np
import tempfile
import os
import time
import tensorflow as tf
from PIL import Image

from bp_face_recognition.models.factory import RecognizerFactory
from bp_face_recognition.models.methods.tflite_recognizer import TFLiteRecognizer


class TestQuantizationMediaPipeIntegration:
    """Test integration between quantization and MediaPipe detection."""

    @pytest.fixture
    def quantized_model(self):
        """Create a quantized model for integration testing."""
        # Create a face recognition model
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer"),
                tf.keras.layers.Rescaling(1.0 / 255),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(
                    512, activation=None, name="embedding_layer"
                ),  # Face embedding
            ]
        )

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
            model_path = f.name

        model.save(model_path)

        # Quantize with different strategies
        from scripts.quantize_model import quantize_model

        quantized_paths = {}

        # Float16 quantization
        quantized_paths["float16"] = quantize_model(model_path, "float16")

        # Dynamic range quantization
        quantized_paths["dynamic"] = quantize_model(model_path, "dynamic")

        # Cleanup original model
        os.unlink(model_path)

        yield quantized_paths

        # Cleanup quantized models
        for path in quantized_paths.values():
            if os.path.exists(path):
                os.unlink(path)

    @pytest.fixture
    def test_faces_dataset(self):
        """Create a test dataset with face images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            face_images = []

            # Generate synthetic face images
            for i in range(5):
                # Create a face-like pattern
                img_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)

                # Add face-like features (simplified)
                # Eyes
                cv2 = __import__("cv2")
                cv2.circle(img_array, (80, 80), 15, (255, 255, 255), -1)
                cv2.circle(img_array, (144, 80), 15, (255, 255, 255), -1)
                # Mouth
                cv2.ellipse(
                    img_array, (112, 140), (20, 10), 0, 0, 180, (255, 255, 255), -1
                )

                img_path = os.path.join(temp_dir, f"face_{i}.jpg")
                Image.fromarray(img_array).save(img_path)
                face_images.append(img_path)

            yield temp_dir, face_images

    def test_end_to_end_pipeline_performance(self, quantized_model, test_faces_dataset):
        """Test end-to-end pipeline performance with quantized models."""
        temp_dir, face_images = test_faces_dataset

        # Test different quantization strategies
        performance_results = {}

        for quant_type, model_path in quantized_model.items():
            print(f"\nTesting {quant_type} quantization...")

            # Initialize TFLite recognizer
            recognizer = TFLiteRecognizer(model_path)

            # Initialize MediaPipe detector
            try:
                detector = RecognizerFactory.get_detector("mediapipe")
            except Exception as e:
                pytest.skip(f"MediaPipe not available: {e}")

            # Measure pipeline performance
            start_time = time.time()
            processed_faces = 0

            for img_path in face_images[:3]:  # Test with 3 images
                # Load image
                img = cv2 = __import__("cv2")
                cv2_img = cv2.imread(img_path)

                if cv2_img is None:
                    continue

                # Detect faces
                try:
                    boxes = detector.detect(cv2_img)

                    for box in boxes:
                        x, y, w, h = box
                        face = cv2_img[y : y + h, x : x + w]

                        if face.size > 0:
                            # Extract embedding
                            embedding = recognizer.get_embedding(face)

                            if embedding is not None and len(embedding) > 0:
                                processed_faces += 1

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

            end_time = time.time()
            total_time = end_time - start_time

            performance_results[quant_type] = {
                "total_time": total_time,
                "processed_faces": processed_faces,
                "avg_time_per_face": total_time / max(processed_faces, 1),
            }

            print(f"{quant_type}: {total_time:.3f}s total, {processed_faces} faces")

        # Verify all quantization strategies work
        assert len(performance_results) == 2
        for quant_type, results in performance_results.items():
            assert (
                results["processed_faces"] > 0
            ), f"{quant_type} failed to process any faces"
            assert (
                results["avg_time_per_face"] < 1.0
            ), f"{quant_type} too slow: {results['avg_time_per_face']:.3f}s"

    def test_model_accuracy_comparison(self, quantized_model, test_faces_dataset):
        """Compare accuracy between different quantization strategies."""
        temp_dir, face_images = test_faces_dataset

        # Load original (non-quantized) model for baseline
        baseline_embeddings = {}
        quantized_embeddings = {}

        # Test with first face image
        img_path = face_images[0]
        cv2 = __import__("cv2")
        cv2_img = cv2.imread(img_path)

        if cv2_img is None:
            pytest.skip("Could not load test image")

        # Extract face using MediaPipe
        try:
            detector = RecognizerFactory.get_detector("mediapipe")
            boxes = detector.detect(cv2_img)

            if not boxes:
                pytest.skip("No faces detected in test image")

            x, y, w, h = boxes[0]
            face = cv2_img[y : y + h, x : x + w]

            if face.size == 0:
                pytest.skip("Extracted face is empty")

        except Exception as e:
            pytest.skip(f"Face detection failed: {e}")

        # Create baseline model (non-quantized)
        baseline_model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer"),
                tf.keras.layers.Rescaling(1.0 / 255),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(512, activation=None, name="embedding_layer"),
            ]
        )

        # Get baseline embedding
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_batch = np.expand_dims(face_normalized, axis=0)

        baseline_embedding = (
            baseline_model(face_batch, training=False).numpy().flatten()
        )

        # Test quantized models
        for quant_type, model_path in quantized_model.items():
            recognizer = TFLiteRecognizer(model_path)
            quantized_embedding = recognizer.get_embedding(face)

            # Calculate similarity with baseline
            similarity = np.dot(baseline_embedding, quantized_embedding) / (
                np.linalg.norm(baseline_embedding) * np.linalg.norm(quantized_embedding)
            )

            print(f"{quant_type} similarity to baseline: {similarity:.4f}")

            # Quantized models should have high similarity (>0.9) to baseline
            assert (
                similarity > 0.9
            ), f"{quant_type} quantization significantly reduced accuracy: {similarity}"

    def test_speed_improvement_benchmark(self, quantized_model, test_faces_dataset):
        """Benchmark speed improvements with quantization."""
        temp_dir, face_images = test_faces_dataset

        # Create larger model for more realistic benchmarking
        large_model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer"),
                tf.keras.layers.Rescaling(1.0 / 255),
                tf.keras.layers.Conv2D(64, 3, activation="relu"),
                tf.keras.layers.Conv2D(64, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(128, 3, activation="relu"),
                tf.keras.layers.Conv2D(128, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(256, 3, activation="relu"),
                tf.keras.layers.Conv2D(256, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(512, activation=None, name="embedding_layer"),
            ]
        )

        # Save and quantize large model
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
            large_model_path = f.name

        large_model.save(large_model_path)

        from scripts.quantize_model import quantize_model

        large_quantized_paths = {
            "float16": quantize_model(large_model_path, "float16"),
            "dynamic": quantize_model(large_model_path, "dynamic"),
        }

        # Cleanup original
        os.unlink(large_model_path)

        try:
            # Benchmark performance
            cv2 = __import__("cv2")
            test_face = cv2.resize(cv2.imread(face_images[0]), (224, 224))

            # Test baseline model
            start_time = time.time()
            for _ in range(50):
                face_batch = np.expand_dims(
                    test_face.astype(np.float32) / 255.0, axis=0
                )
                embedding = large_model(face_batch, training=False).numpy()
            baseline_time = time.time() - start_time

            # Test quantized models
            speed_results = {"baseline": baseline_time / 50}

            for quant_type, model_path in large_quantized_paths.items():
                recognizer = TFLiteRecognizer(model_path)

                start_time = time.time()
                for _ in range(50):
                    embedding = recognizer.get_embedding(test_face)
                quantized_time = time.time() - start_time

                speed_results[quant_type] = quantized_time / 50

                # Calculate speedup
                speedup = baseline_time / quantized_time
                print(f"{quant_type} speedup: {speedup:.2f}x")

                # Should show some speed improvement
                assert speedup > 1.0, f"{quant_type} quantization did not improve speed"

            print(f"Baseline: {speed_results['baseline']:.4f}s per inference")
            for quant_type, avg_time in speed_results.items():
                if quant_type != "baseline":
                    print(f"{quant_type}: {avg_time:.4f}s per inference")

        finally:
            # Cleanup quantized models
            for path in large_quantized_paths.values():
                if os.path.exists(path):
                    os.unlink(path)

    def test_memory_usage_comparison(self, quantized_model):
        """Test memory usage of different quantization strategies."""
        import psutil
        import os

        # Get current process
        process = psutil.Process(os.getpid())

        memory_results = {}

        for quant_type, model_path in quantized_model.items():
            # Measure memory before loading model
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Load quantized model
            recognizer = TFLiteRecognizer(model_path)

            # Measure memory after loading model
            memory_after = process.memory_info().rss / 1024 / 1024  # MB

            memory_usage = memory_after - memory_before
            memory_results[quant_type] = memory_usage

            print(f"{quant_type} memory usage: {memory_usage:.2f} MB")

            # Cleanup
            del recognizer

            # Force garbage collection
            import gc

            gc.collect()

        # All quantization strategies should use reasonable memory
        for quant_type, memory_mb in memory_results.items():
            assert (
                memory_mb < 100
            ), f"{quant_type} uses too much memory: {memory_mb:.2f} MB"
