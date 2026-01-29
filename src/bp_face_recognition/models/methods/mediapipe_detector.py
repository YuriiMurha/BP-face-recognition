import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple
from bp_face_recognition.models.interfaces import FaceDetector


class MediaPipeDetector(FaceDetector):
    """
    MediaPipe face detector optimized for GPU acceleration.
    Implements BlazeFace model for high-speed face detection.
    """

    def __init__(self, min_detection_confidence=0.5, use_gpu=True):
        """
        Initialize MediaPipe detector.

        Args:
            min_detection_confidence: Minimum confidence threshold
            use_gpu: Enable GPU delegate for acceleration
        """
        self.use_gpu = use_gpu

        # Initialize MediaPipe Face Detection
        try:
            # Try new Tasks API first
            base_options = mp.tasks.BaseOptions()
            if use_gpu:
                base_options.delegate = mp.tasks.BaseOptions.Delegate.GPU

            options = mp.tasks.vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=min_detection_confidence,
            )

            self.detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        except Exception as e:
            print(f"MediaPipe Tasks API failed: {e}, falling back to legacy API")
            try:
                # Try to access solutions module
                if hasattr(mp, "solutions"):
                    self.detector = mp.solutions.face_detection.FaceDetection(
                        min_detection_confidence=min_detection_confidence,
                        model_selection=0,  # Short range model for speed
                    )
                    self.use_legacy_api = True
                else:
                    print("MediaPipe solutions module not available")
                    raise e
            except Exception as e2:
                print(f"MediaPipe legacy API also failed: {e2}")
                raise e2

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using MediaPipe.

        Args:
            image: Input image in BGR format (OpenCV default)

        Returns:
            List of bounding boxes as (x, y, w, h)
        """
        if image is None:
            return []

        if hasattr(self, "use_legacy_api") and self.use_legacy_api:
            # Use legacy FaceDetection API
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process with legacy API
            results = self.detector.process(rgb_image)

            boxes = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x, y, w, h = (
                        int(bbox.xmin),
                        int(bbox.ymin),
                        int(bbox.width),
                        int(bbox.height),
                    )
                    boxes.append((x, y, w, h))

            return boxes
        else:
            # Use Tasks API
            rgb_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            )

            # Detect faces
            detection_result = self.detector.detect(rgb_image)

            # Convert detections to bounding boxes
            boxes = []
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                # MediaPipe returns (x_min, y_min, width, height)
                boxes.append(
                    (
                        int(bbox.origin_x),
                        int(bbox.origin_y),
                        int(bbox.width),
                        int(bbox.height),
                    )
                )

            return boxes

    def detect_with_confidence(
        self, image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces using MediaPipe and return bounding boxes with confidence scores.

        Args:
            image: Input image in BGR format

        Returns:
            List of tuples: ((x, y, w, h), confidence)
        """
        if image is None:
            return []

        if hasattr(self, "use_legacy_api") and self.use_legacy_api:
            # Use legacy FaceDetection API
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process with legacy API
            results = self.detector.process(rgb_image)

            boxes_with_conf = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x, y, w, h = (
                        int(bbox.xmin),
                        int(bbox.ymin),
                        int(bbox.width),
                        int(bbox.height),
                    )
                    confidence = (
                        detection.score[0] if hasattr(detection, "score") else 0.5
                    )
                    boxes_with_conf.append(((x, y, w, h), float(confidence)))

            return boxes_with_conf
        else:
            # Use Tasks API
            rgb_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            )

            # Detect faces
            detection_result = self.detector.detect(rgb_image)

            # Convert detections to bounding boxes with confidence
            results = []
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                box = (
                    int(bbox.origin_x),
                    int(bbox.origin_y),
                    int(bbox.width),
                    int(bbox.height),
                )
                confidence = detection.categories[0].score
                results.append((box, float(confidence)))

            return results

    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, "detector"):
            self.detector.close()
