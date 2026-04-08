"""
Closed-Set Face Recognition Application

Classifies faces directly into one of N known identities from training.
No registration or database required — the model already knows all identities.

Usage:
    python closed_set_main.py --recognizer facenet_pu --threshold 0.7
"""

import cv2
import logging
from pathlib import Path

from bp_face_recognition.services.closed_set_pipeline_service import (
    ClosedSetPipelineService,
)
from bp_face_recognition.config.settings import settings, CameraSourceType
from bp_face_recognition.utils.camera_source import create_camera_manager


class ClosedSetApp:
    """Face recognition app using closed-set classification (no database)."""

    def __init__(
        self,
        recognizer: str = "facenet_pu",
        threshold: float = 0.7,
        detector: str = "mediapipe_v1",
    ):
        self.service = ClosedSetPipelineService(
            detector_type=detector,
            recognizer_type=recognizer,
            confidence_threshold=threshold,
        )

        self.camera = create_camera_manager()

        # Logging
        log_file = settings.LOGS_DIR / "closed_set.log"
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        source_display = self.camera.config.source_type
        if source_display == CameraSourceType.USB:
            source_display = "webcam (USB phone)"
        device_idx = (
            self.camera.config.device_index
            if self.camera.config.device_index is not None
            else 0
        )

        logging.info(
            f"ClosedSetApp started: camera={source_display}, device={device_idx}, "
            f"recognizer={recognizer}, threshold={threshold}"
        )

    def draw_results(self, frame, results):
        """Draw bounding boxes and labels on frame."""
        processed_frame = frame.copy()
        for res in results:
            x, y, w, h = res["box"]
            label = res["label"]

            # Green for recognized, red for unknown
            color = (0, 255, 0) if "Unknown" not in label else (0, 0, 255)

            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                processed_frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
        return processed_frame

    def run(self):
        """Run the closed-set recognition loop."""
        window_title = f"{settings.APP_NAME} [CLOSED-SET]"
        class_names = self.service.get_class_names()

        print(f"Starting {window_title}...")
        print(f"Mode: Closed-Set Classification ({len(class_names)} classes)")
        print(f"Known identities: {', '.join(class_names)}")
        print(f"Recognizer: {self.service.recognizer_type}")
        print(f"Confidence threshold: {self.service.confidence_threshold}")
        print("No registration needed — model classifies directly.")

        if not self.camera.is_connected():
            print("Failed to connect to camera!")
            return

        print("Press 'q' to quit")

        frame_count = 0
        face_count = 0
        skip_frames = 3
        results = []

        while True:
            frame = self.camera.read_frame()
            if frame is None:
                break

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if frame_count % skip_frames == 0:
                result = self.service.process_image(frame_bgr)
                results = []

                if result.get("success"):
                    recognition_result = result.get("recognition_result")
                    reco_faces = (
                        recognition_result.get("faces", [])
                        if recognition_result
                        else []
                    )

                    if frame_count % 30 == 0:
                        num_detected = result.get("detection_result", {}).get(
                            "num_faces", 0
                        )
                        print(f"Frame {frame_count}: detected {num_detected} faces")

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

            processed_frame = self.draw_results(frame_bgr, results)

            cv2.imshow(window_title, processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1

        print(f"Processed {frame_count} frames, detected {face_count} total faces")
        logging.info(
            f"Session complete: {frame_count} frames, {face_count} faces detected"
        )

        self.camera.release()
        cv2.destroyAllWindows()


def get_default_recognizer_from_config():
    """Read default recognizer from config file."""
    import yaml

    config_path = Path(__file__).parent.parent.parent / "config" / "models.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            return config.get("global", {}).get(
                "default_recognizer", "facenet_pu"
            )
    except Exception:
        return "facenet_pu"


if __name__ == "__main__":
    import argparse

    default_recognizer = get_default_recognizer_from_config()

    parser = argparse.ArgumentParser(
        description="Closed-Set Face Recognition (no registration needed)"
    )
    parser.add_argument(
        "--recognizer",
        type=str,
        default=default_recognizer,
        choices=["facenet_tl", "facenet_pu", "facenet_tloss"],
        help=f"Classifier model (default: {default_recognizer})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold (below = Unknown) (default: 0.7)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="mediapipe_v1",
        help="Face detector (default: mediapipe_v1)",
    )

    args = parser.parse_args()

    print(f"Using recognizer: {args.recognizer}")
    app = ClosedSetApp(
        recognizer=args.recognizer,
        threshold=args.threshold,
        detector=args.detector,
    )
    app.run()
