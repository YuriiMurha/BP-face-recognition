"""
Vision Module - Computer Vision Components

This module provides face detection, recognition, and training capabilities
with a plugin-based architecture for dynamic model loading.
"""

from .interfaces import FaceDetector, FaceRecognizer
from .factory import RecognizerFactory, Factory
from .registry import ModelRegistry, get_registry, reload_registry

# Import detection classes for backward compatibility
try:
    from .detection.base import (
        BaseDetector,
        DetectionResult,
        create_detection_result_from_legacy,
    )
    from .recognition.base import BaseRecognizer

    DETECTION_AVAILABLE = True
except ImportError:
    BaseDetector = None
    DetectionResult = None
    create_detection_result_from_legacy = None
    BaseRecognizer = None
    DETECTION_AVAILABLE = False

__all__ = [
    "FaceDetector",
    "FaceRecognizer",
    "RecognizerFactory",
    "Factory",
    "ModelRegistry",
    "get_registry",
    "reload_registry",
]

if DETECTION_AVAILABLE:
    __all__.extend(
        [
            "BaseDetector",
            "DetectionResult",
            "create_detection_result_from_legacy",
            "BaseRecognizer",
        ]
    )
