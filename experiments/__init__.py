from ..vision.core.face_tracker import FaceTracker

# Backward compatibility - expose vision components from models package
from ..vision import (
    FaceDetector,
    FaceRecognizer,
    ModelRegistry,
    get_registry,
    BaseDetector,
    DetectionResult,
    BaseRecognizer,
    RecognizerFactory,
)
