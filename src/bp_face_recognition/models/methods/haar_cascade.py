import cv2
import numpy as np
from typing import List, Tuple
from bp_face_recognition.models.interfaces import FaceDetector


class HaarCascadeDetector(FaceDetector):
    def __init__(self):
        # Use type ignore because cv2 stubs are incomplete or data attribute dynamic
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )  # type: ignore

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        # faces is (n, 4) numpy array or tuple if empty
        if len(faces) == 0 or isinstance(faces, tuple):
            return []
        # Convert numpy int32 to python int for typing consistency
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
