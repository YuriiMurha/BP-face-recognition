import cv2
import numpy as np
import logging

from database import FaceDatabase
from model import FaceTracker

class AttendanceApp:
    def __init__(self, camera_id=0, threshold=0.6):
        self.tracker = FaceTracker()
        self.db = FaceDatabase(db_type='postgres', conn_params={
            'dbname': 'faces_db', 'user': 'user', 'password': 'pass', 'host': 'localhost'
        })
        self.cap = cv2.VideoCapture(camera_id)
        self.threshold = threshold
        logging.basicConfig(filename='attendance.log', level=logging.INFO)

    def process_frame(self, frame):
        # Detect faces
        boxes_confidences = self.tracker.detect_faces(frame)
        processed_frame = frame.copy()

        known_embeddings = self.db.get_all_embeddings()
        for (box, confidence), idx in zip(boxes_confidences, range(len(boxes_confidences))):
            if confidence < 0.9:
                continue
            x, y, w, h = box
            face_crop = frame[y:y+h, x:x+w]
            embedding = self.tracker.get_embedding(face_crop)

            # Compare with database
            distances = [np.linalg.norm(embedding - emb) for _, emb in known_embeddings]
            if distances:
                min_dist = min(distances)
                match_idx = distances.index(min_dist)
                face_id, _ = known_embeddings[match_idx]
                label = str(face_id) if min_dist < self.threshold else "stranger"
            else:
                label = "stranger"

            # Handle new faces
            if label == "stranger":
                face_id = self.db.add_face(embedding)
                label = str(face_id)

            # Draw bounding box and label
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(processed_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Log detection
            self.db.log_detection(face_id, label)
            logging.info(f"Detected: ID={face_id}, Label={label}")

        return processed_frame

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            processed_frame = self.process_frame(frame)
            cv2.imshow('Attendance System', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AttendanceApp()
    app.run()