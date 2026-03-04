import uuid
import cv2


def capture_and_save_images(frame, img_path):
    img_path.mkdir(parents=True, exist_ok=True)
    imgname = img_path / f"{str(uuid.uuid1())}.jpg"
    cv2.imwrite(str(imgname), frame)


def main():
    from bp_face_recognition.config.settings import settings

    IMAGES_PATH = settings.DATA_DIR / "camera_captures"
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)

    from bp_face_recognition.utils.camera_source import create_camera_manager
    from bp_face_recognition.config.settings import CameraSourceType

    camera = create_camera_manager()

    # Normalize source type for display
    source_display = camera.config.source_type
    if source_display == CameraSourceType.USB:
        source_display = "webcam (USB phone)"

    device_idx = (
        camera.config.device_index if camera.config.device_index is not None else 0
    )

    print(f"Camera source: {source_display}")
    print(f"Camera device: {device_idx}")

    if not camera.is_connected():
        print("Failed to connect to camera!")
        return

    print("Press 'q' to quit, 'c' to capture image")

    while True:
        frame = camera.read_frame()
        if frame is None:
            print("Failed to read frame")
            break

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Camera Stream", frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            capture_and_save_images(frame_bgr, IMAGES_PATH)
            print(f"Image captured to {IMAGES_PATH}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
