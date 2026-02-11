"""Vision Detection Module - Updated for New Architecture"""

from ..interfaces import FaceDetector, FaceRecognizer
from .base import BaseDetector, DetectionResult, create_detection_result_from_legacy
from .mediapipe import MediaPipeDetector
from .mtcnn import MTCNNDetector
from .haar_cascade import HaarCascadeDetector
from .dlib_hog import DlibHOGDetector
from .face_recognition_lib import FaceRecognitionLibDetector

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "create_detection_result_from_legacy",
    "MediaPipeDetector",
    "MTCNNDetector",
    "HaarCascadeDetector",
    "DlibHOGDetector",
    "FaceRecognitionLibDetector",
]
