import cv2
import logging
from bp_face_recognition.database.database import FaceDatabase
from bp_face_recognition.models.recognition_service import RecognitionService
from bp_face_recognition.config.settings import settings


class AttendanceApp:
    def __init__(self, camera_id=0, threshold=0.6, db_type="csv"):
        self.service = RecognitionService(
            threshold=threshold, database=FaceDatabase(db_type=db_type)
        )
        self.cap = cv2.VideoCapture(camera_id)

        # Log to configured logs directory
        log_file = settings.LOGS_DIR / "attendance.log"
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def draw_results(self, frame, results):
        """Draws bounding boxes and labels on the frame."""
        processed_frame = frame.copy()
        for res in results:
            x, y, w, h = res["box"]
            label = res["label"]

            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                processed_frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )
        return processed_frame

    def run(self):
        print(f"Starting {settings.APP_NAME}...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 1. Pure Recognition Logic
            results = self.service.process_frame(frame)

            # 2. UI/Visualization Logic
            processed_frame = self.draw_results(frame, results)

            cv2.imshow(settings.APP_NAME, processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = AttendanceApp(db_type="csv")
    app.run()
