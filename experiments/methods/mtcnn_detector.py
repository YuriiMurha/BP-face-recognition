import cv2
import numpy as np
from mtcnn import MTCNN
from typing import List, Tuple
from bp_face_recognition.models.interfaces import FaceDetector

class MTCNNDetector(FaceDetector):
    def __init__(self):
        """
        Initialize the MTCNN detector.
        """
        self.detector = MTCNN()

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using MTCNN.
        """
        if image is None:
            return []

        # MTCNN expects RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb_image)
        
        # MTCNN returns box as [x, y, width, height]
        return [tuple(face['box']) for face in faces]

    def detect_with_confidence(self, image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces using MTCNN and return bounding boxes with confidence scores.
        """
        if image is None:
            return []

        # MTCNN expects RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb_image)
        
        return [(tuple(face['box']), face['confidence']) for face in faces]
