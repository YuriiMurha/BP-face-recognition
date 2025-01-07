import os
import uuid
import cv2

IMAGES_PATH = os.path.join(r'C:\Users\yuram\Documents\BP\FaceDetection\data/seccam_2','images')
number_images = 30

cap = cv2.VideoCapture("rtsp://147.232.24.197/live.sdp")

imgnum = 0
i = 0
while (imgnum < number_images):
    ret, frame = cap.read()
    if i % 12 == 0:
        print('Collecting image {}'.format(imgnum))
        imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
        cv2.imwrite(imgname, frame)
        imgnum = imgnum + 1
    i = i + 1
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()