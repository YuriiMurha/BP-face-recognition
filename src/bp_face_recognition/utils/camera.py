import cv2
from mtcnn import MTCNN
import numpy as np

class Camera:
    def __init__(self, device=0):
        self.cap = cv2.VideoCapture(device)
        self.detector = MTCNN()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def detect_faces(self, frame):
        detections = self.detector.detect_faces(frame)
        faces = []
        for det in detections:
            x, y, w, h = det['box']
            face = frame[y:y+h, x:x+w]
            if face.size > 0:
                face = face.astype(np.float32) / 255.0
                faces.append({'image': face, 'box': [x, y, x+w, y+h]})
        return faces

    def release(self):
        self.cap.release()