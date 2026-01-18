import os
import uuid
import cv2


async def capture_and_save_images(frame, img_path):
    imgname = os.path.join(img_path, f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)

def main():
    IMAGES_PATH = os.path.join(r'C:\Users\yuram\Documents\BP\data/datasets/seccam_2', 'images')
    cam1 = cv2.VideoCapture("rtsp://147.232.24.197/live.sdp")
    cam2 = cv2.VideoCapture("rtsp://147.232.24.189/live.sdp")

    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        if not ret1 or not ret2:
            break
        cv2.imshow('frame1', frame1)
        cv2.imshow('frame2', frame2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            capture_and_save_images(frame1, IMAGES_PATH)
            capture_and_save_images(frame2, IMAGES_PATH)

    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
