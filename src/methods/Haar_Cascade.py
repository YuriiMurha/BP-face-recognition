import cv2
import time

# Load Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_haar(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    start_time = time.time()
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    detection_time = time.time() - start_time

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return image, detection_time, len(faces)

# Example usage
image_path = 'face-recognition\Basic\ImagesBasic\webcam_example.jpg'
output_image, detection_time, num_faces = detect_faces_haar(image_path)
print(f"Haar Cascade - Detected {num_faces} faces in {detection_time:.4f} seconds")
cv2.imshow("Detected Faces", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
