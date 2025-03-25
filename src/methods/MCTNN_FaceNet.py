import tensorflow as tf
from mtcnn import MTCNN
import cv2
import time

# Load MTCNN face detector and FaceNet model
face_detector = MTCNN()
# facenet_model = tf.keras.models.load_model('C:\\Users\\yuram\\Documents\\BP\\facenet_keras.h5')  # Load FaceNet model

def detect_faces_facenet(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    start_time = time.time()
    faces = face_detector.detect_faces(rgb_image)
    detection_time = time.time() - start_time

    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Extract face region for FaceNet
        face_region = rgb_image[y:y+h, x:x+w]
        face_resized = cv2.resize(face_region, (160, 160))  # FaceNet input size
        face_resized = face_resized.astype('float32') / 255.0
        face_resized = tf.expand_dims(face_resized, axis=0)

        # Perform FaceNet embedding generation (optional for recognition)
        # embedding = facenet_model(face_resized)

    return image, detection_time, len(faces)

# Example usage
image_path =  'face-recognition\Basic\ImagesBasic\webcam_example.jpg'
output_image, detection_time, num_faces = detect_faces_facenet(image_path)
print(f"FaceNet - Detected {num_faces} faces in {detection_time:.4f} seconds")
cv2.imshow("Detected Faces", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
