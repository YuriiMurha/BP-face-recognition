"""
Updated Factory Pattern using Model Registry

This module provides backward-compatible factory methods that now use
the configuration-driven registry system.
"""

import logging
from typing import Optional, Any, Dict

from bp_face_recognition.vision.registry import get_registry
from bp_face_recognition.vision.interfaces import FaceDetector, FaceRecognizer

logger = logging.getLogger(__name__)


class RecognizerFactory:
    """
    Factory class to instantiate different types of FaceRecognizers and FaceDetectors.
    Now uses configuration-driven registry system with backward compatibility.
    """

    @staticmethod
    def get_recognizer(
        recognizer_type: str = "custom_cnn_v1",
        model_path: Optional[str] = None,
        **kwargs: Any,
    ) -> FaceRecognizer:
        """
        Instantiate a recognizer based on type and configuration.

        Args:
            recognizer_type (str): Type of recognizer from config ('facenet_v1', 'custom_cnn_v1', 'tflite_v1').
            model_path (str, optional): Override model path from config
            **kwargs: Additional configuration overrides

        Returns:
            FaceRecognizer: An instance of requested recognizer.
        """
        try:
            registry = get_registry()

            # Try to get from registry first
            if recognizer_type in registry.list_recognizers():
                recognizer_kwargs = kwargs.copy()
                if model_path:
                    recognizer_kwargs["model_path"] = model_path

                return registry.get_recognizer(recognizer_type, **recognizer_kwargs)

            # Fallback to legacy names for backward compatibility
            legacy_mapping = {
                "facenet": "facenet_v1",
                "custom": "custom_cnn_v1",
                "tflite": "tflite_v1",
            }

            if recognizer_type in legacy_mapping:
                mapped_type = legacy_mapping[recognizer_type]
                logger.info(
                    f"Legacy name '{recognizer_type}' mapped to '{mapped_type}'"
                )

                recognizer_kwargs = kwargs.copy()
                if model_path:
                    recognizer_kwargs["model_path"] = model_path

                return registry.get_recognizer(mapped_type, **recognizer_kwargs)

            # If not found, try registry (which will raise appropriate error)
            return registry.get_recognizer(recognizer_type, **kwargs)

        except Exception as e:
            logger.error(f"Failed to create recognizer '{recognizer_type}': {e}")
            raise

    @staticmethod
    def get_detector(
        detector_type: str = "mediapipe_v1", **kwargs: Any
    ) -> FaceDetector:
        """
        Instantiate a face detector based on type and configuration.

        Args:
            detector_type (str): Type of detector from config ('mediapipe_v1', 'mtcnn_v1', 'haar_v1', 'dlib_hog_v1', 'face_recognition_lib_v1').
            **kwargs: Detector-specific configuration options.

        Returns:
            FaceDetector: An instance of requested detector.
        """
        try:
            registry = get_registry()

            # Try to get from registry first
            if detector_type in registry.list_detectors():
                return registry.get_detector(detector_type, **kwargs)

            # Fallback to legacy names for backward compatibility
            legacy_mapping = {
                "mtcnn": "mtcnn_v1",
                "mediapipe": "mediapipe_v1",
                "haar": "haar_v1",
                "dlib_hog": "dlib_hog_v1",
                "face_recognition": "face_recognition_lib_v1",
            }

            if detector_type in legacy_mapping:
                mapped_type = legacy_mapping[detector_type]
                logger.info(f"Legacy name '{detector_type}' mapped to '{mapped_type}'")
                return registry.get_detector(mapped_type, **kwargs)

            # If not found, try registry (which will raise appropriate error)
            return registry.get_detector(detector_type, **kwargs)

        except Exception as e:
            logger.error(f"Failed to create detector '{detector_type}': {e}")
            raise

    @staticmethod
    def get_optimized_detector(**kwargs: Any) -> FaceDetector:
        """
        Get best available detector with optimization for current hardware.
        Falls back gracefully if preferred detectors are not available.

        Args:
            **kwargs: Configuration options for detector.

        Returns:
            FaceDetector: Optimized detector instance.
        """
        registry = get_registry()
        global_settings = registry.get_global_settings()

        # Try detectors in order of preference
        preferred_order = ["mediapipe_v1", "mtcnn_v1", "haar_v1"]

        for detector_type in preferred_order:
            try:
                return RecognizerFactory.get_detector(detector_type, **kwargs)
            except Exception as e:
                logger.warning(f"Detector '{detector_type}' failed: {e}")
                continue

        # If all fail, try the first available detector
        available_detectors = list(registry.list_detectors().keys())
        if available_detectors:
            logger.warning(f"Trying fallback detector: {available_detectors[0]}")
            return RecognizerFactory.get_detector(available_detectors[0], **kwargs)

        raise RuntimeError("No detectors available in registry")

    @staticmethod
    def get_default_detector(**kwargs: Any) -> FaceDetector:
        """
        Get default detector as specified in configuration.

        Args:
            **kwargs: Configuration options for detector.

        Returns:
            FaceDetector: Default detector instance.
        """
        registry = get_registry()
        return registry.get_default_detector(**kwargs)

    @staticmethod
    def get_default_recognizer(**kwargs: Any) -> FaceRecognizer:
        """
        Get default recognizer as specified in configuration.

        Args:
            **kwargs: Configuration options for recognizer.

        Returns:
            FaceRecognizer: Default recognizer instance.
        """
        registry = get_registry()
        return registry.get_default_recognizer(**kwargs)

    @staticmethod
    def list_available_detectors() -> Dict[str, Dict[str, Any]]:
        """
        List all available detectors from registry.

        Returns:
            Dict of detector configurations
        """
        registry = get_registry()
        return registry.list_detectors()

    @staticmethod
    def list_available_recognizers() -> Dict[str, Dict[str, Any]]:
        """
        List all available recognizers from registry.

        Returns:
            Dict of recognizer configurations
        """
        registry = get_registry()
        return registry.list_recognizers()


# Backward compatibility aliases
Factory = RecognizerFactory  # Keep old name for compatibility
