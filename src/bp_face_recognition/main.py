import cv2
import logging
from bp_face_recognition.services.pipeline_service import PipelineService
from bp_face_recognition.database.database import FaceDatabase
from bp_face_recognition.services.database_service import DatabaseService
from bp_face_recognition.config.settings import settings, CameraSourceType
from bp_face_recognition.utils.camera_source import create_camera_manager


class AttendanceApp:
    def __init__(self, camera_id=0, threshold=0.5, db_type="csv", recognizer="metric_efficientnetb0_128d"):
        face_db = FaceDatabase(db_type=db_type)
        db_service = DatabaseService(database=face_db)

        self.service = PipelineService(
            detector_type="mediapipe_v1",
            recognizer_type=recognizer,
            recognition_threshold=threshold,
            database_service=db_service,
        )

        self.camera = create_camera_manager()

        # Normalize source type for display
        source_display = self.camera.config.source_type
        if source_display == CameraSourceType.USB:
            source_display = "webcam (USB phone)"
        device_idx = (
            self.camera.config.device_index
            if self.camera.config.device_index is not None
            else 0
        )

        log_file = settings.LOGS_DIR / "attendance.log"
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        logging.info(
            f"AttendanceApp started with camera source: {source_display}, device: {device_idx}"
        )
        logging.info(
            f"Detector: {self.service.face_tracker.detector_type}, Recognizer: {self.service.face_tracker.recognizer_type}, Threshold: {threshold}"
        )

    def draw_results(self, frame, results):
        processed_frame = frame.copy()
        for res in results:
            x, y, w, h = res["box"]
            label = res["label"]

            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                processed_frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
        return processed_frame

    def run(self):
        print(f"Starting {settings.APP_NAME}...")

        source_display = self.camera.config.source_type
        if source_display == CameraSourceType.USB:
            source_display = "webcam (USB phone)"
        device_idx = (
            self.camera.config.device_index
            if self.camera.config.device_index is not None
            else 0
        )

        print(f"Camera source: {source_display}")
        print(f"Camera device: {device_idx}")
        print(f"Detector: {self.service.face_tracker.detector_type}")
        print(f"Recognizer: {self.service.face_tracker.recognizer_type}")
        print(f"Threshold: {self.service.recognition_threshold}")

        if not self.camera.is_connected():
            print("Failed to connect to camera!")
            return

        print("Press 'q' to quit")

        frame_count = 0
        face_count = 0
        skip_frames = 3  # Process every Nth frame for recognition
        results = []
        
        # Get detector info
        detector_info = self.service.face_tracker.get_detector_info()
        print(f"Detector info: {detector_info}")
        logging.info(f"Detector info: {detector_info}")

        while True:
            frame = self.camera.read_frame()
            if frame is None:
                break

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Only run the heavy pipeline every few frames
            if frame_count % skip_frames == 0:
                result = self.service.process_image(frame_bgr)
                results = []

                if result.get("success"):
                    detection_result = result.get("detection_result", {})
                    num_detected = detection_result.get("num_faces", 0)
                    
                    # Log detection info every 10 frames
                    if frame_count % 30 == 0:
                        print(f"Frame {frame_count}: detected {num_detected} faces")
                        logging.info(f"Detection result: {detection_result}")

                    recognition_result = result.get("recognition_result")
                    reco_faces = (
                        recognition_result.get("faces", [])
                        if recognition_result
                        else []
                    )

                    for face in reco_faces:
                        if "box" in face:
                            label = face.get("identity", "Unknown")
                            conf = face.get("recognition_confidence", 0)
                            results.append(
                                {
                                    "box": face["box"],
                                    "label": f"{label} ({conf:.2f})",
                                }
                            )

                    face_count += len(results)

            # Always draw the last known results on the current frame
            processed_frame = self.draw_results(frame_bgr, results)

            cv2.imshow(settings.APP_NAME, processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1

        print(f"Processed {frame_count} frames, detected {face_count} total faces")
        logging.info(
            f"Session complete: {frame_count} frames processed, {face_count} faces detected"
        )

        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Face Recognition Attendance System")
    parser.add_argument("--recognizer", type=str, default="metric_efficientnetb0_128d",
                        help="Recognizer type: metric_efficientnetb0_128d, dlib_v1")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Recognition threshold (lower = stricter)")
    parser.add_argument("--db-type", type=str, default="csv",
                        help="Database type: csv, sqlite")

    args = parser.parse_args()

    print(f"DEBUG: Running {os.path.abspath(__file__)}")
    print(f"Using recognizer: {args.recognizer}")
    app = AttendanceApp(db_type=args.db_type, recognizer=args.recognizer, threshold=args.threshold)
    app.run()
