from typing import Optional, Union, Dict, Any
from bp_face_recognition.models.interfaces import FaceRecognizer, FaceDetector
from bp_face_recognition.models.methods.facenet_recognizer import FaceNetRecognizer


class RecognizerFactory:
    """
    Factory class to instantiate different types of FaceRecognizers and FaceDetectors.
    Supports easy switching between FaceNet, custom Keras models, TFLite models, and various detectors.
    """

    @staticmethod
    def get_recognizer(
        recognizer_type: str = "custom", model_path: Optional[str] = None
    ) -> FaceRecognizer:
        """
        Instantiate a recognizer based on type and path.

        Args:
            recognizer_type (str): Type of recognizer ('facenet', 'custom', 'tflite').
            model_path (str, optional): Path to model file.

        Returns:
            FaceRecognizer: An instance of requested recognizer.
        """
        # Late import to avoid circular dependencies and unnecessary overhead
        from bp_face_recognition.models.model import CustomFaceRecognizer

        if recognizer_type == "facenet":
            return FaceNetRecognizer(model_path=model_path)

        if recognizer_type == "custom":
            return CustomFaceRecognizer(model_path=model_path)

        if recognizer_type == "tflite":
            # Late import to avoid circular dependencies
            from bp_face_recognition.models.methods.tflite_recognizer import (
                TFLiteRecognizer,
            )

            return TFLiteRecognizer(model_path=model_path)

        raise ValueError(f"Unknown recognizer type: {recognizer_type}")

    @staticmethod
    def get_detector(detector_type: str = "mtcnn", **kwargs: Any) -> FaceDetector:
        """
        Instantiate a face detector based on type and configuration.

        Args:
            detector_type (str): Type of detector ('mtcnn', 'mediapipe', 'haar', 'dlib_hog', 'face_recognition').
            **kwargs: Detector-specific configuration options.

        Returns:
            FaceDetector: An instance of requested detector.
        """
        # Late imports to avoid circular dependencies
        from bp_face_recognition.models.methods.mtcnn_detector import MTCNNDetector
        from bp_face_recognition.models.methods.haar_cascade import HaarCascadeDetector
        from bp_face_recognition.models.methods.dlib_hog import DlibHOGDetector
        from bp_face_recognition.models.methods.mediapipe_detector import (
            MediaPipeDetector,
        )
        from bp_face_recognition.models.methods.face_recognition_detector import (
            FaceRecognitionLibDetector,
        )

        if detector_type == "mtcnn":
            return MTCNNDetector()

        if detector_type == "mediapipe":
            return MediaPipeDetector(
                min_detection_confidence=kwargs.get("min_detection_confidence", 0.5),
                use_gpu=kwargs.get("use_gpu", True),
            )

        if detector_type == "haar":
            return HaarCascadeDetector()

        if detector_type == "dlib_hog":
            return DlibHOGDetector()

        if detector_type == "face_recognition":
            return FaceRecognitionLibDetector()

        raise ValueError(f"Unknown detector type: {detector_type}")

    @staticmethod
    def get_optimized_detector(**kwargs: Any) -> FaceDetector:
        """
        Get the best available detector with optimization for the current hardware.
        Falls back to MTCNN if MediaPipe is not available.

        Args:
            **kwargs: Configuration options for the detector.

        Returns:
            FaceDetector: Optimized detector instance.
        """
        try:
            return RecognizerFactory.get_detector("mediapipe", **kwargs)
        except Exception as e:
            print(f"MediaPipe not available ({e}), falling back to MTCNN")
            return RecognizerFactory.get_detector("mtcnn", **kwargs)
