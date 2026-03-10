import os
import sys
import cv2
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from bp_face_recognition.utils.camera_source import create_camera_manager
from bp_face_recognition.config.settings import settings
from bp_face_recognition.services.pipeline_service import PipelineService
from bp_face_recognition.database.database import FaceDatabase
from bp_face_recognition.services.database_service import DatabaseService
from bp_face_recognition.vision.registry import get_registry


def get_default_recognizer():
    """Get default recognizer from models.yaml global settings."""
    try:
        registry = get_registry()
        return registry.get_global_settings().get(
            "default_recognizer", "metric_efficientnetb0_128d"
        )
    except Exception:
        return "metric_efficientnetb0_128d"


def main():
    parser = argparse.ArgumentParser(description="Register a person from camera")
    parser.add_argument("name", type=str, help="Name of the person to register")
    parser.add_argument(
        "--count", type=int, default=10, help="Number of samples to capture"
    )
    parser.add_argument(
        "--recognizer",
        type=str,
        default=None,
        help="Recognizer type (default: from models.yaml)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Recognition threshold (default: 0.7)",
    )
    args = parser.parse_args()

    name = args.name
    target_count = args.count

    # Use default from models.yaml if not specified
    recognizer_type = args.recognizer if args.recognizer else get_default_recognizer()
    threshold = args.threshold

    print(f"Registering '{name}'...")
    print(f"Recognizer: {recognizer_type}")
    print(f"Threshold: {threshold}")
    print(f"Goal: Capture {target_count} clear face samples.")

    # Initialize components
    face_db = FaceDatabase(db_type="csv")
    db_service = DatabaseService(database=face_db)

    # Use configurable recognizer from models.yaml default
    service = PipelineService(
        detector_type="mediapipe_v1",
        recognizer_type=recognizer_type,
        recognition_threshold=threshold,
        database_service=db_service,
    )

    camera = create_camera_manager()
    if not camera.is_connected():
        print("Error: Could not connect to camera.")
        return

    print("\nInstructions:")
    print("1. Look at the camera.")
    print("2. Press 'c' to capture a sample.")
    print("3. Press 'q' to cancel.")
    print("Try different angles and lighting for better accuracy.")

    samples = []

    while len(samples) < target_count:
        frame_rgb = camera.read_frame()
        if frame_rgb is None:
            continue

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        display_frame = frame_bgr.copy()

        # Just run detection for feedback
        result = service.face_tracker.track_faces(frame_bgr, update_history=False)

        for face in result.get("faces", []):
            x, y, w, h = face["box"]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                "Face Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            display_frame,
            f"Samples: {len(samples)}/{target_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Registration - Press 'c' to capture", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Registration cancelled.")
            camera.release()
            cv2.destroyAllWindows()
            return
        elif key == ord("c"):
            if result.get("num_faces") > 0:
                # Get the largest face embedding
                faces = result.get("faces", [])
                faces.sort(key=lambda x: x["box"][2] * x["box"][3], reverse=True)
                embedding = faces[0]["embedding"]

                if embedding is not None:
                    samples.append(embedding)
                    print(f"Captured sample {len(samples)}/{target_count}")
                else:
                    print("Failed to extract embedding. Try again.")
            else:
                print("No face detected! Make sure your face is visible.")

    print(f"\nCaptured {len(samples)} samples. Registering in database...")

    # Register using service for consistency
    success = db_service.register_person(name, samples)

    if success:
        print(f"SUCCESS! '{name}' is now registered.")
    else:
        print(f"FAILED to register '{name}'. Check logs for details.")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
