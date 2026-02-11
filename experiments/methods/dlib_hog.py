import cv2
import dlib  # type: ignore
import numpy as np
from typing import List, Tuple
from bp_face_recognition.models.interfaces import FaceDetector


class DlibHOGDetector(FaceDetector):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        # Ensure image is in 8-bit format for dlib
        if image.dtype != np.uint8:
            image = (
                (image * 255).astype(np.uint8)
                if image.max() <= 1.0
                else image.astype(np.uint8)
            )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        # faces is dlib.rectangles
        return [
            (face.left(), face.top(), face.width(), face.height()) for face in faces
        ]

    def detect_with_confidence(
        self, image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        if image.dtype != np.uint8:
            image = (
                (image * 255).astype(np.uint8)
                if image.max() <= 1.0
                else image.astype(np.uint8)
            )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # run returns (rectangles, scores, idx)
        faces, scores, _ = self.detector.run(gray, 1)

        results = []
        for face, score in zip(faces, scores):
            results.append(
                ((face.left(), face.top(), face.width(), face.height()), float(score))
            )
        return results
