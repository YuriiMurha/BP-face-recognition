import cv2
import logging
from bp_face_recognition.services.pipeline_service import PipelineService
from bp_face_recognition.database.database import FaceDatabase
from bp_face_recognition.services.database_service import DatabaseService
from bp_face_recognition.config.settings import settings


class AttendanceApp:
    def __init__(self, camera_id=0, threshold=0.6, db_type="csv"):
        # Create database first
        face_db = FaceDatabase(db_type=db_type)
        db_service = DatabaseService(database=face_db)

        self.service = PipelineService(
            detector_type="mediapipe_v1",
            recognizer_type="efficientnetb0_webcam_gpu_quantized",
            recognition_threshold=threshold,
            database_service=db_service,
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
            result = self.service.process_image(frame)
            results = []
            if result.get("success") and result.get("recognition_result"):
                for face in result["recognition_result"].get("faces", []):
                    if "box" in face:
                        results.append(
                            {
                                "box": face["box"],
                                "label": face.get("identity", "Unknown"),
                            }
                        )

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
