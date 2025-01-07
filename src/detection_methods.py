# detection_methods.py
import time
import cv2
import dlib
from mtcnn import MTCNN
import face_recognition


# Initialize models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
hog_face_detector = dlib.get_frontal_face_detector()
face_detector = MTCNN()
# facenet_model = tf.keras.models.load_model('path_to_facenet_model.h5')  # Replace with your model path


def detect_faces_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    start_time = time.time()
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    detection_time = time.time() - start_time

    return faces, detection_time


def detect_faces_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    start_time = time.time()
    faces = hog_face_detector(gray, 1)
    detection_time = time.time() - start_time

    return faces, detection_time


def detect_faces_facenet(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    start_time = time.time()
    faces = face_detector.detect_faces(rgb_image)
    detection_time = time.time() - start_time

    return faces, detection_time


def detect_faces_face_recognition(image):
    img_small = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    start_time = time.time()
    face_locations = face_recognition.face_locations(img_rgb)
    detection_time = time.time() - start_time

    return face_locations, detection_time
