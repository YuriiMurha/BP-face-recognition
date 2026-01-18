import cv2
import numpy as np
import logging
from bp_face_recognition.database.database import FaceDatabase
from bp_face_recognition.models.model import FaceTracker
from bp_face_recognition.config.settings import settings

class AttendanceApp:
    def __init__(self, camera_id=0, threshold=0.6, db_type='csv'):
        self.tracker = FaceTracker()
        # Defaulting to CSV for easier initial testing if postgres isn't running
        self.db = FaceDatabase(db_type=db_type)
        self.cap = cv2.VideoCapture(camera_id)
        self.threshold = threshold
        
        # Log to configured logs directory
        log_file = settings.LOGS_DIR / 'attendance.log'
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(log_file), 
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def process_frame(self, frame):
        # Detect faces
        boxes_confidences = self.tracker.detect_faces(frame)
        processed_frame = frame.copy()

        known_embeddings = self.db.get_all_embeddings()
        for (box, confidence) in boxes_confidences:
            if confidence < 0.8: # Adjusted confidence threshold
                continue
            x, y, w, h = box
            # Ensure box is within frame boundaries
            x, y = max(0, x), max(0, y)
            face_crop = frame[y:y+h, x:x+w]
            
            if face_crop.size == 0:
                continue

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
        print(f"Starting {settings.APP_NAME}...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            processed_frame = self.process_frame(frame)
            cv2.imshow(settings.APP_NAME, processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AttendanceApp(db_type='csv')
    app.run()
