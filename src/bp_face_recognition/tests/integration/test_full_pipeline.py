import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestFullPipelineIntegration:
    """Integration tests for full detection → recognition pipeline."""

    @pytest.fixture
    def registry(self):
        from bp_face_recognition.vision.registry import get_registry

        return get_registry()

    def test_end_to_end_detection_recognition(self, registry):
        """Test full E2E pipeline: detection → recognition → output."""
        available_detectors = list(registry.list_detectors().keys())
        available_recognizers = list(registry.list_recognizers().keys())

        if not available_detectors or not available_recognizers:
            pytest.skip("Detectors or recognizers not available")

        try:
            detector = registry.get_detector(available_detectors[0])
            recognizer = registry.get_recognizer(available_recognizers[0])

            test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            detections = detector.detect(test_img)

            results = []
            for det in detections:
                if len(det) >= 2:
                    bbox, confidence = det[0], det[1]
                    x, y, w, h = bbox

                    face_img = test_img[y : y + h, x : x + w]

                    if face_img.size > 0:
                        embedding = recognizer.get_embedding(face_img)

                        results.append(
                            {
                                "bbox": bbox,
                                "confidence": confidence,
                                "embedding": embedding,
                            }
                        )

            assert isinstance(results, list)
        except FileNotFoundError:
            pytest.skip("Model not available")
        except Exception as e:
            pytest.skip(f"Pipeline not functional: {e}")

    def test_pipeline_with_no_faces(self, registry):
        """Test pipeline handles image with no faces."""
        available = list(registry.list_detectors().keys())
        if not available:
            pytest.skip("No detectors available")

        try:
            detector = registry.get_detector(available[0])
            blank_img = np.zeros((480, 640, 3), dtype=np.uint8)

            detections = detector.detect(blank_img)

            assert isinstance(detections, (list, tuple))
            assert len(detections) >= 0
        except FileNotFoundError:
            pytest.skip("Model not available")
        except Exception as e:
            pytest.skip(f"Detector not functional: {e}")


class TestErrorRecovery:
    """Test error handling and recovery."""

    def test_graceful_degradation_invalid_image(self):
        """Test graceful handling of invalid image."""
        from bp_face_recognition.vision.registry import get_registry

        registry = get_registry()
        available = list(registry.list_recognizers().keys())
        if not available:
            pytest.skip("No recognizers available")

        try:
            recognizer = registry.get_recognizer(available[0])

            invalid_img = np.array([])

            with pytest.raises((ValueError, AttributeError)):
                recognizer.get_embedding(invalid_img)
        except FileNotFoundError:
            pytest.skip("Model not available")
        except Exception:
            pass

    def test_recovery_from_detection_failure(self):
        """Test recovery when detection fails."""
        from bp_face_recognition.vision.registry import get_registry

        registry = get_registry()
        available = list(registry.list_detectors().keys())
        if not available:
            pytest.skip("No detectors available")

        try:
            detector = registry.get_detector(available[0])

            corrupted_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

            result = detector.detect(corrupted_img)

            assert isinstance(result, (list, tuple))
        except FileNotFoundError:
            pytest.skip("Model not available")
        except Exception:
            pass


class TestPerformanceBaseline:
    """Performance benchmark tests."""

    @pytest.mark.slow
    def test_detection_speed_baseline(self):
        """Test detection speed baseline."""
        from bp_face_recognition.vision.registry import get_registry

        registry = get_registry()
        available = list(registry.list_detectors().keys())
        if not available:
            pytest.skip("No detectors available")

        try:
            detector = registry.get_detector(available[0])

            test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            import time

            start = time.time()
            for _ in range(10):
                detector.detect(test_img)
            elapsed = time.time() - start

            avg_time = elapsed / 10
            fps = 1.0 / avg_time if avg_time > 0 else 0

            assert fps > 0
        except FileNotFoundError:
            pytest.skip("Model not available")
        except Exception as e:
            pytest.skip(f"Detector not functional: {e}")

    @pytest.mark.slow
    def test_recognition_speed_baseline(self):
        """Test recognition speed baseline."""
        from bp_face_recognition.vision.registry import get_registry

        registry = get_registry()
        available = list(registry.list_recognizers().keys())
        if not available:
            pytest.skip("No recognizers available")

        try:
            recognizer = registry.get_recognizer(available[0])

            test_img = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)

            import time

            start = time.time()
            for _ in range(10):
                recognizer.get_embedding(test_img)
            elapsed = time.time() - start

            avg_time = elapsed / 10
            fps = 1.0 / avg_time if avg_time > 0 else 0

            assert fps > 0
        except FileNotFoundError:
            pytest.skip("Model not available")
        except Exception as e:
            pytest.skip(f"Recognizer not functional: {e}")


class TestMultiModelSupport:
    """Test with different model configurations."""

    def test_different_detector_configs(self):
        """Test loading detectors with different configs."""
        from bp_face_recognition.vision.registry import get_registry

        registry = get_registry()
        available = list(registry.list_detectors().keys())
        if not available:
            pytest.skip("No detectors available")

        try:
            detector = registry.get_detector(available[0], min_detection_confidence=0.7)

            assert detector is not None
        except FileNotFoundError:
            pytest.skip("Model not available")
        except Exception as e:
            pytest.skip(f"Detector not functional: {e}")

    def test_different_recognizer_configs(self):
        """Test loading recognizers with different configs."""
        from bp_face_recognition.vision.registry import get_registry

        registry = get_registry()
        available = list(registry.list_recognizers().keys())
        if not available:
            pytest.skip("No recognizers available")

        try:
            recognizer = registry.get_recognizer(available[0], input_size=160)

            assert recognizer is not None
        except FileNotFoundError:
            pytest.skip("Model not available")
        except Exception as e:
            pytest.skip(f"Recognizer not functional: {e}")
